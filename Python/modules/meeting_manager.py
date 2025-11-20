import os, sys, datetime
import json
import time, copy
from queue import Queue, Empty
from threading import Lock
import cv2, asyncio
import shutil
import numpy as np
from modules.model_processor import ModelProcessor
from modules.change_detector import ChangeDetector
from modules.vdb_module import HybridSearcher
from modules.rerank_module import OpenVINORerankQwen3Rerank06B
from modules.roi_module import ROIProcessor

from modules.tokenlizer import tokenize_ocr


defined_prompt = '''这是一张在线会议软件的截图。请你主要聚焦于这张图片所传达的核心内容，忽略参会人员、聊天窗口等次要信息。分析并总结之后，按以下格式输出：\
[主要内容]
详细描述并总结这张图片的主要内容。请忽略参会人员、聊天窗口等次要信息。

[视图模式]
根据截图判断当前会议所处模式，必须是以下三种模式之一，仅输出模式结果：
- 摄像头模式：摄像头模式表现为主窗口为摄像头画面（例如主讲者、画廊、多发言者、Together 场景、会议室场景等）。
- 共享模式：主窗口为屏幕或PPT等共享内容，主体部分不包含摄像头画面或画面较小。
- 聊天模式：无摄像头画面且无共享内容，主体部分仅显示单个或多个头像。

[共享类型]
如果判断是摄像头模式或聊天模式，则认为共享类型为“非PPT”，直接输出即可。如果判断当前会议为“共享模式”，继续判断共享的类型，必须是以下两种类型之一，仅输出共享类型：
- PPT
- 非PPT

[PPT包含元素]
如果判断共享类型为“非PPT”，则直接输出NA。如果判断共享类型为“PPT”时，请从下面类型中挑选出该PPT中主要包含的元素，可以多选，仅从下面类别中选。
- NA
- Title，Agenda，text, illustration, background image, chart, diagram, table, video, summary, conclusion, SmartArt, Equations

[PPT标题]
如果判断共享类型为“PPT”时，请输出PPT标题内容（假设存在），如果判断共享类型为“非PPT”或不存在标题信息，则输出NA
- PPT标题
- NA
'''

defined_prompt = '''角色：资深会议总结专家  
任务：分析一张在线会议截图，仅提炼“核心信息”，忽略“参会人员、聊天窗口或头像”等次要部分。  
 
请严格按以下模板输出：
 
[主要内容]  
- 详细表述图片的主旨、关键点。若包含 PPT，请完整重现其文字信息与逻辑结构。
 
[视图模式]  
- 仅输出三选一：摄像头模式 / 内容共享模式 / 聊天听众模式
 
[共享类型]  
- 若视图为“内容共享模式”，判断为 PPT 或 非PPT；否则一律输出“非PPT”。
 
[PPT包含元素]  
- 若为 PPT，可选的内容元素有（可多选）：Title、Agenda、text、illustration、background image、chart、diagram、table、video、summary、conclusion、SmartArt、Equations  
 
[PPT标题]  
- 若为 PPT，输出标题文本；若无标题或非 PPT，则输出 NA
'''


defined_prompt = '''这是一张在线会议软件的截图。请详细描述并总结这张图片的主要内容。请忽略参会人员，聊天窗口等次要信息。'''


# defined_prompt = '''你是一位会议总结专家，请概括总结下面会议截图中的主要内容。
# 要求：
# - 对于图片中的文字，请详细输出文字内容
# - 对于图片中的图表，请笼统的概括总结图表内容，不需要详细输出内容
# '''


#initial_modules = set(sys.modules.keys())
# --------------------------- 全局单例模块 ---------------------------
GLOBAL_MODEL_PROCESSOR = ModelProcessor(prompt=defined_prompt)  # 全局模型处理器（单例）
GLOBAL_CHANGE_DETECTOR = ChangeDetector()  # 全局变化检测器（单例）
GLOBAL_HYBRID_SEARCHER = HybridSearcher()  # 全局搜索（单例）
GLOBAL_RERANK_RERANKER = OpenVINORerankQwen3Rerank06B()  # 全局排序（单例）
GLOBAL_ROI_PROCESSOR   = ROIProcessor()    # 全局ROI提取 (单例)
# # 找出新增的模块
# new_modules = set(sys.modules.keys()) - initial_modules

# # 打印动态加载的模块
# print("\n动态加载的模块:")
# for module_name in new_modules:
#     module = sys.modules[module_name]
#     print(f"- {module_name} ({getattr(module, '__file__', 'built-in')})")
# --------------------------- 全局会话存储 ---------------------------
active_sessions = {}
sessions_lock = Lock()


def timestamp() -> str:
    """获取用于日志记录的时间戳，格式：[YYYY-MM-DD HH:MM:SS]"""
    now = datetime.datetime.now()
    return now.strftime("[%Y-%m-%d %H:%M:%S]")

def get_executable_dir() -> str:
    """获取可执行文件所在的真实目录"""
    if getattr(sys, 'frozen', False):
        # 打包环境：通过 sys.executable 获取路径
        exe_path = os.path.realpath(sys.executable)
        # print('session path=',os.path.dirname(exe_path))
        return os.path.dirname(exe_path)
    else:
        # 开发环境：使用当前工作目录
        exec_dir = os.getcwd()
        # print('session path=', exec_dir)
        return exec_dir
    

SESSION_SAVE_DIR = os.path.join(get_executable_dir(), "./sessions")
os.makedirs(SESSION_SAVE_DIR, exist_ok=True)


class MeetingSession:
    def __init__(self, session_id, meeting_name):
        self.session_id = session_id
        self.session_status = False  # True - 会议分析已结束  False - 会议分析进行中
        self.meeting_name = meeting_name
        self.is_running = True
        self._stop_flag = False
        self.frame_queue_finished = False
        self.keyframe_queue_finished = False
        self.frame_queue = Queue(maxsize=10000)
        self.keyframe_queue = Queue(maxsize=10000)
        self.key_frames = []
        self.history_features = []  # 保存历史特征（供全局检测器使用）
        self.current_interval_start = 0  # 当前区间起始帧号
        self.total_frames_processed = 0  # 已处理总帧数
        self.local_path = ''             # 本地图片路径
        self.last_activity_time = time.time()
        self.session_start_time = time.time()
        self.session_end_time = time.time()
        self.frame_dict = {}      # 帧号 -> 文件全路径
        self.frame_analysis = {}  # 帧号 -> 多模态文本
        self.frame_ocr = {}       # 帧号 -> ocr文本
        self.lock = Lock()
        #self.prompt = '这是一张开会时的桌面截图，截图包含PPT的内容。请详细描述PPT相关的内容并进行总结。务必忽略PPT以外的信息。'  # for lenovo
        ##self.prompt = '这是一张开会时的桌面截图，截图包含PPT的内容。请详细描述PPT相关的内容并进行总结。若画面中间主要显示人脸并且不包含PPT内容，则输出“无”，务必忽略PPT以外的信息。'
        #self.prompt = 'This is a screenshot of your desktop during a meeting and the screenshot contains the content of a PPT. Please describe the content of the PPT in detail and summarize it. If the center of the screen mainly shows faces and does not contain the PPT, output “None”, and be sure to ignore information other than the PPT.'
        self.prompt = defined_prompt

    def parse_ppt_info_list(self, text):
        # 定义所有可能的字段及其顺序
        fields = ["主要内容", "视图模式", "共享类型", "PPT包含元素", "PPT标题"]
        
        # 初始化结果字典，所有字段默认值为"NA"
        result = {field: "NA" for field in fields}
        
        # 处理字符串中存在的字段
        import re
        
        # 用于匹配[字段名]内容模式的正则表达式
        pattern = r'\[([^\]]+)\]([^[]*)'
        matches = re.findall(pattern, text)
        
        for field_name, content in matches:
            # 去除内容前后的空白字符
            content = content.strip()
            
            # 如果内容是"NA"，则保持为字符串"NA"
            if content.upper() == "NA":
                result[field_name] = "NA"
            else:
                # 对"PPT包含元素"字段进行特殊处理，转换为列表
                if field_name == "PPT包含元素":
                    # 按逗号分割并去除每个元素周围的空白字符
                    elements = [element.strip() for element in content.split(',') if element.strip()]
                    result[field_name] = elements if elements else ["NA"]  # 确保至少有"NA"
                else:
                    result[field_name] = content
        
        return result

    def parse_ppt_info(self, text):
        # 定义所有可能的字段及其顺序
        fields = ["主要内容", "视图模式", "共享类型", "PPT包含元素", "PPT标题"]
        result = {field: "NA" for field in fields}
        result['PPT包含元素'] = []
        import re

        pattern = r'\[([^\]]+)\]([^[]*)'
        matches = re.findall(pattern, text)
        ele = ''
        for field_name, content in matches:
            content = content.strip()
            if content.upper() == "NA":
                result[field_name] = "NA"
            else:
                # 对"PPT包含元素"字段进行特殊处理，转换为小写英文列表
                if field_name == "PPT包含元素":
                    ele = copy.deepcopy(content)
                    # 按逗号分割并去除空白
                    elements = [element.strip() for element in content.split(',') if element.strip()]
                    
                    # 处理每个元素：保留字母并转换为小写
                    processed_elements = []
                    for element in elements:
                        # 保留所有字母并转换为小写
                        cleaned = ''.join(filter(str.isalpha, element)).lower()
                        if cleaned:  # 如果处理后不为空
                            processed_elements.append(cleaned)
                    # 确保至少有一个元素
                    result[field_name] = processed_elements #if processed_elements else ["NA"]
                else:
                    result[field_name] = content

        if '摄像头模式' in result['视图模式']:
            result['视图模式'] = '摄像头模式'
        elif '共享模式' in result['视图模式']:
             result['视图模式'] = '共享模式'
        elif '聊天模式' in result['视图模式']:
             result['视图模式'] = '聊天模式'
        
        if '摄像头模式' in result['视图模式'] or '聊天模式' in result['视图模式']:
            result['共享类型'] = '非PPT'
            result['PPT包含元素'] = []
            result['PPT标题'] = 'NA'

        field_mapping = {
            "主要内容": "content",
            "视图模式": "view_mode",
            "共享类型": "shared_type",
            "PPT包含元素": "ppt_elements",
            "PPT标题": "ppt_title"
        }
    
        # 初始化结果字典
        english_dict = {}
        
        # 遍历映射表进行字段转换
        for chinese_key, english_key in field_mapping.items():
            # 检查原始字典中是否存在该中文字段
            if chinese_key in result:
                english_dict[english_key] = result[chinese_key]
            else:
                # 若不存在则填充为None（可根据需求改为其他默认值）
                english_dict[english_key] = None
        english_dict['ori_ele'] = ele
        return english_dict

    def format_desc(self, desc: str) -> dict:
        import re
        """格式化描述字符串"""
        # 去除多余的空格和换行
        desc = re.sub(r'\s+', ' ', desc).strip()
        
        # 规范化标点符号
        desc = desc.replace('。', '。 ')
        desc = desc.replace('，', '， ')
        desc = desc.replace('：', '： ')
        
        # 处理特殊标记，如代码块
        desc = desc.replace('```', '')
        
        return self.parse_ppt_info(desc)
    
    def copy_image_with_keyframes(self, image_path: str) -> str:
        # 获取图片所在目录和文件名
        directory, filename = os.path.split(image_path)
        keyFrame_path = os.path.join(directory, self.session_id)
        if not os.path.exists(keyFrame_path):
            os.makedirs(keyFrame_path, exist_ok=True)
        name, ext = os.path.splitext(filename)
        # 构建新的文件名
        new_filename = f"{name}_keyFrame{ext}"
        new_image_path = os.path.join(keyFrame_path, new_filename)
        shutil.copy(image_path, new_image_path)
        
        return new_image_path

    def format_elapsed_time(self, start_time, end_time):
        # 计算时间差
        elapsed_time = end_time - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        
        time_parts = []
        if hours > 0:
            time_parts.append(f"{hours} 小时")
        if minutes > 0:
            time_parts.append(f"{minutes} 分钟")
        if seconds > 0:
            time_parts.append(f"{seconds} 秒")
        
        return " ".join(time_parts)

    def add_frame(self, image_path, frame_id):
        #with self.lock:
            if self._stop_flag:
                return False, "会议已停止"
            if self.frame_queue.full():
                return False, "队列已满"
            # 防止重复上传
            if frame_id in self.frame_dict.keys():
                return False, '上传失败，当前图片已上传(%s)' % image_path
            self.frame_queue.put({"path": image_path, "frame_id": frame_id})
            self.frame_dict[frame_id] = image_path
            self.last_activity_time = time.time()
            if not self.local_path:
                self.local_path = os.path.dirname(image_path)
            return True, "上传成功"

    def start_capture(self):
        self.session_start_time = time.time()
        self.session_status = False
        while self.is_running and not self._stop_flag:
            try:
                frame_data = self.frame_queue.get(timeout=1)
                self._process_frame(frame_data)
                self.frame_queue.task_done()
            except Empty:
                continue

        print(timestamp(),'接收到停止信号，待分析数据：', self.frame_queue.qsize())
        analysis_time = time.time()
        while True:
            try:
                frame_data = self.frame_queue.get_nowait()
                self._process_frame(frame_data)
                self.frame_queue.task_done()
            except Empty:
                break
        self.frame_queue_finished = True  # 标记队列任务完成
        print(timestamp(),'会议记录结束，等待分析结果...')
        # 等待关键帧队列结束分析
        while not self.keyframe_queue_finished:
            time.sleep(1)
        
        if not len(self.key_frames) and self.total_frames_processed == 0:
            self.is_running = False
            if self.session_id in active_sessions:
                del active_sessions[self.session_id]
            print(timestamp(),'会议记录结束，无效的会议(%s)' % self.session_id)
            return
        # 处理最后一个区间
        self._finalize_last_interval()
        ret = self._save_session()
        if ret:
            self.session_status = True
            self.session_end_time = time.time()
            print(timestamp(),'会议分析结束，可查询会议分析结果。')
            print(timestamp(),'会议时长：',  self.format_elapsed_time(self.session_start_time, analysis_time))
            print(timestamp(),'分析耗时：', self.format_elapsed_time(analysis_time, self.session_end_time))
        else:
            if self.session_id in active_sessions:
                del active_sessions[self.session_id]
        self.is_running = False

    # def start_analysis(self):
    #     self.keyframe_queue_finished = False
    #     while True:
    #         try:
    #             frame_data = self.keyframe_queue.get(timeout=1)
    #             #self._process_frame(frame_data)
    #             self._close_current_interval(frame_data["frame_id"]) 
    #             self.keyframe_queue.task_done()
    #         except Empty:
    #             if self.frame_queue_finished:
    #                 if self.keyframe_queue.qsize() == 0:
    #                     break
    #             continue
    #     self.keyframe_queue_finished = True
    def start_analysis(self):
        self.keyframe_queue_finished = False
        frame_batch = []  # 创建用于累积帧数据的列表
        while True:
            try:
                # 当累积数据达到4帧或队列为空时处理
                if len(frame_batch) >= 4 and self.keyframe_queue.qsize() == 0 or (self.frame_queue_finished and not self.keyframe_queue.qsize() and frame_batch):
                    if len(frame_batch):
                        frame_batch_crop = copy.deepcopy(frame_batch)
                        for item in frame_batch_crop:
                            item['path'] = GLOBAL_ROI_PROCESSOR.roi_predict(item['path'], self.session_id)
                        batch_result = GLOBAL_MODEL_PROCESSOR.inference(frame_batch_crop)
                        if batch_result:
                            for idx, item in enumerate(frame_batch):
                                self._close_current_interval(item["frame_id"], batch_result[idx]) 
                    frame_batch = []  # 清空列表
                # 如果列表为空且队列未完成，等待新数据
                #if not frame_batch and not self.frame_queue_finished:
                frame_data = self.keyframe_queue.get(timeout=1)
                frame_batch.append(frame_data)
                self.keyframe_queue.task_done()
                # # 检查是否有更多数据可立即获取（非阻塞）
                # while True:
                # try:
                #     frame_data = self.keyframe_queue.get_nowait()
                #     frame_batch.append(frame_data)
                #     self.keyframe_queue.task_done()
                    
                # 当累积数据达到8帧时立即处理
                if len(frame_batch) >= 8:
                    frame_batch_crop = copy.deepcopy(frame_batch)
                    for item in frame_batch_crop:
                        item['path'] = GLOBAL_ROI_PROCESSOR.roi_predict(item['path'], self.session_id)
                    batch_result = GLOBAL_MODEL_PROCESSOR.inference(frame_batch_crop)
                    if batch_result:
                        for idx, item in enumerate(frame_batch):
                            self._close_current_interval(item["frame_id"], batch_result[idx]) 
                    frame_batch = []  # 清空列表
                # except Empty:
                #     break  # 没有更多数据，退出内部循环
                
                
                # 检查退出条件
                if self.frame_queue_finished and not self.keyframe_queue.qsize() and not frame_batch:
                    break
            except Empty:
                # 仅在队列为空且标记为已完成时退出
                if self.frame_queue_finished and not self.keyframe_queue.qsize() and not frame_batch:
                    break
                continue
        self.keyframe_queue_finished = True

    def _process_frame(self, frame_data):
        time.sleep(0.5)

        from app import crop_ppt_vertical, crop_ppt_fixed
        image_path = frame_data["path"]
        #crop_ppt_vertical(image_path=image_path, output_path=image_path)
        #crop_ppt_fixed(image_path, image_path, 67, 67, 13, 14)
        # end

        image = cv2.imread(frame_data["path"])
        if image is None:
            print(timestamp(),'cannot read image file, please check image path format.', frame_data["path"])
            return
        current_feature = GLOBAL_CHANGE_DETECTOR._extract_feature(image)
        self.total_frames_processed += 1
        # 检测变化
        #print('当前处理帧：', frame_data['path'])
        while len(self.history_features) < GLOBAL_CHANGE_DETECTOR.history_size:
            self.history_features.append(current_feature)
        is_change, similaritys = GLOBAL_CHANGE_DETECTOR.detect_change(
            current_feature, self.history_features
        )
        #print('clip检测结果：', is_change, similaritys)
        #print('#####_process_frame(%s): is_change' % self.frame_queue.qsize(), is_change, frame_data["path"], similaritys)
        # 更新历史特征
        #with self.lock:
        self.history_features.append(current_feature)
        if len(self.history_features) > GLOBAL_CHANGE_DETECTOR.history_size:
            self.history_features.pop(0)

        # 提前检查场景是否变化
        if not is_change:
            # clip->no change, add OCR check
            #if similaritys[-1] < 0.98:
            if similaritys[-1] < 0.998: # 0.998
                current_frame_id = frame_data['frame_id']
                last_frame_id = current_frame_id - 1 if current_frame_id > 0 and current_frame_id-1 in self.frame_dict.keys() else current_frame_id
                current_frame_path = self.frame_dict[current_frame_id]
                last_frame_path = self.frame_dict[last_frame_id]
                ocr_time = time.time()
                #ret = GLOBAL_CHANGE_DETECTOR.detect_change_by_ov_ocr(current_frame_path, last_frame_path)
                ret = False
                current_frame_path_crop = GLOBAL_ROI_PROCESSOR.roi_predict(current_frame_path, self.session_id)
                last_frame_path_crop = GLOBAL_ROI_PROCESSOR.roi_predict(last_frame_path, self.session_id)
                if '_crop' not in current_frame_path_crop or '_crop' not in last_frame_path_crop:
                    if '_crop' in current_frame_path_crop and self.session_id in current_frame_path_crop:
                        try:
                            os.remove(current_frame_path_crop)
                        except Exception as e:
                            print(f"删除文件时出错: {current_frame_path_crop} \n {e}")
                    if '_crop' in last_frame_path_crop and self.session_id in last_frame_path_crop:
                        try:
                            os.remove(last_frame_path_crop)
                        except Exception as e:
                            print(f"删除文件时出错: {last_frame_path_crop} \n {e}")
                    current_frame_path_crop = current_frame_path
                    last_frame_path_crop = last_frame_path

                current_ocr = GLOBAL_CHANGE_DETECTOR.detect_image_ocr(current_frame_path_crop)
                last_ocr = GLOBAL_CHANGE_DETECTOR.detect_image_ocr(last_frame_path_crop)
                if '_crop' in current_frame_path_crop and self.session_id in current_frame_path_crop:
                    try:
                        os.remove(current_frame_path_crop)
                    except Exception as e:
                        print(f"删除文件时出错: {current_frame_path_crop} \n {e}")
                if '_crop' in last_frame_path_crop and self.session_id in last_frame_path_crop:
                    try:
                        os.remove(last_frame_path_crop)
                    except Exception as e:
                        print(f"删除文件时出错: {last_frame_path_crop} \n {e}")
                if len(current_ocr) < 20 and len(last_ocr) < 20:
                    #print('当前帧和上一帧OCR内容<20，丢弃')
                    ret = False
                elif GLOBAL_CHANGE_DETECTOR.cosine_similarity(current_ocr, last_ocr) < 0.9:  # 0.9
                    #print('OCR相似度满足要求，认为场景改变：', GLOBAL_CHANGE_DETECTOR.cosine_similarity(current_ocr, last_ocr), current_frame_path)
                    ret = True
                else:
                    pass
                    #print('OCR相似度满足不要求，认为场景无变化：', GLOBAL_CHANGE_DETECTOR.cosine_similarity(current_ocr, last_ocr), current_frame_path)
                ocr_end = time.time()
                print('O4 time=', ocr_end-ocr_time)
                is_change = ret

                # save ocr
                if current_frame_id not in self.frame_ocr.keys():
                    self.frame_ocr[current_frame_id] = current_ocr if current_ocr else ' '
                if last_frame_id not in self.frame_ocr.keys():
                    self.frame_ocr[last_frame_id] = last_ocr if last_ocr else ' '


                if min(similaritys) > 0.985:
                    is_change = False

        # 变化时闭合当前区间
        if is_change:
            # self._close_current_interval(frame_data["frame_id"])   
            current_frame_id = frame_data['frame_id']
            last_frame_id = current_frame_id - 1 if current_frame_id > 0 and current_frame_id-1 in self.frame_dict.keys() else current_frame_id
            last_frame_path = self.frame_dict[last_frame_id]
            last_frame_dict = {'frame_id': last_frame_id, 'path':last_frame_path}
            # 关键帧取前一区间的最后一张
            self.keyframe_queue.put(last_frame_dict)     
            # 检查关键帧对应的ocr是否存在，不存在则增加检测
            if last_frame_id not in self.frame_ocr.keys():
                last_frame_path_crop = GLOBAL_ROI_PROCESSOR.roi_predict(last_frame_path, self.session_id)
                last_ocr = GLOBAL_CHANGE_DETECTOR.detect_image_ocr(last_frame_path_crop)
                self.frame_ocr[last_frame_id] = last_ocr if last_ocr else ' '


    def _close_current_interval(self, current_frame_id, result):
        # result -> current-1帧的分析结果
        interval_end = current_frame_id# - 1
        mid_frame = interval_end #(self.current_interval_start + interval_end) // 2
        # 查找中间帧路径（实际需根据mid_frame_id匹配）
        mid_frame_path = self.frame_dict[mid_frame] # os.path.join(self.local_path, f"frame_{mid_frame}.jpg")
        analysis = result #GLOBAL_MODEL_PROCESSOR.process(mid_frame_path, self.prompt)
        analysis = str(analysis).strip().replace('\n', '')
        #print('result=', analysis)
        self.copy_image_with_keyframes(mid_frame_path)
        summary = self.format_desc(analysis)
        # 检查关键帧对应的ocr是否存在，不存在则增加检测
        if mid_frame not in self.frame_ocr.keys():
            mid_frame_path_crop = GLOBAL_ROI_PROCESSOR.roi_predict(mid_frame_path, self.session_id)
            mid_ocr = GLOBAL_CHANGE_DETECTOR.detect_image_ocr(mid_frame_path_crop)
            self.frame_ocr[mid_frame] = mid_ocr if mid_ocr else ' '
        else:
            mid_ocr = self.frame_ocr[mid_frame]
        self.key_frames.append({
            "file": mid_frame_path,
            "frame_id": mid_frame,
            "start": self.current_interval_start,
            "end": interval_end,
            "desc": analysis, #summary['content'],
            "asr": '',
            "ocr": mid_ocr,
            "ocr_tks": tokenize_ocr(analysis),
            #"ori_ele": summary['ori_ele'],
            "meta": {
                "view_mode": summary['view_mode'],
                "shared_type": summary['shared_type'],
                "ppt_elements": summary['ppt_elements'],
                "ppt_title": summary['ppt_title']
            }
            #"desc_ori": analysis
        })
        self.current_interval_start = current_frame_id + 1

    def _finalize_last_interval(self):
        if self.total_frames_processed == 0:
            return
        interval_end = self.total_frames_processed - 1
        mid_frame = interval_end #(self.current_interval_start + interval_end) // 2
        #mid_frame_path = os.path.join(self.local_path, f"frame_{mid_frame}.jpg")
        mid_frame_path = self.frame_dict[mid_frame]
        mid_frame_path_crop = GLOBAL_ROI_PROCESSOR.roi_predict(mid_frame_path, self.session_id)
        batch_result = GLOBAL_MODEL_PROCESSOR.inference(copy.deepcopy([{'path':mid_frame_path_crop}]))
        if batch_result:
            analysis = batch_result[0]
        else:
            return
        analysis = str(analysis).strip().replace('\n', '')
        self.copy_image_with_keyframes(mid_frame_path)
        summary = self.format_desc(analysis)
        # 检查关键帧对应的ocr是否存在，不存在则增加检测
        if mid_frame not in self.frame_ocr.keys():
            mid_frame_path_crop = GLOBAL_ROI_PROCESSOR.roi_predict(mid_frame_path, self.session_id)
            mid_ocr = GLOBAL_CHANGE_DETECTOR.detect_image_ocr(mid_frame_path_crop)
            self.frame_ocr[mid_frame] = mid_ocr if mid_ocr else ' '
        else:
            mid_ocr = self.frame_ocr[mid_frame]
        self.key_frames.append({
            "file": mid_frame_path,
            "frame_id": mid_frame,
            "start": self.current_interval_start,
            "end": interval_end,
            "desc": analysis, #summary['content'],
            "asr": '',
            "ocr": mid_ocr,
            "ocr_tks": tokenize_ocr(analysis),
            #"ori_ele": summary['ori_ele'],
            "meta": {
                "view_mode": summary['view_mode'],
                "shared_type": summary['shared_type'],
                "ppt_elements": summary['ppt_elements'],
                "ppt_title": summary['ppt_title']
            }
            #"desc_ori": analysis
        })

    def _save_session(self):
        save_data = {
            "session_id": self.session_id,
            "meeting_name": self.meeting_name,
            "frames": self.key_frames
        }
        save_path = os.path.join(SESSION_SAVE_DIR, f"{self.session_id}.json")
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            # 构造数据，存储到数据库
            final_data = copy.deepcopy(self.key_frames)
            for item in final_data:
                item['session_id'] = self.session_id
                item['meeting_name'] = self.meeting_name
                item['text_vec'] = GLOBAL_CHANGE_DETECTOR.bgem3_model.encode([item['desc']])[0][0]
                item['ocr_vec'] = GLOBAL_CHANGE_DETECTOR.bgem3_model.encode([item['ocr']])[0][0]
            GLOBAL_HYBRID_SEARCHER.upload('image', final_data)
            return True
        except BaseException as e:
            print(timestamp(),'***保存会话结果出错, 会议 id =', self.session_id)
            print(timestamp(),'***错误信息:', e)
            return False

    def monitor_timeout(self):
        while self.is_running:
            #with self.lock:
            if time.time() - self.last_activity_time > 20:
                self._stop_flag = True
                break
            else:
                time.sleep(1)

    def stop(self):
        with self.lock:
            self._stop_flag = True


    