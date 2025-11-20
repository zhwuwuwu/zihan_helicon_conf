import requests
import cv2
import os, re
import tempfile
import time
from datetime import datetime



def parse_json_file(file_path):
    import json
    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 解析JSON数据为Python列表
        data_list = json.loads(content)
        
        return data_list
    
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到")
    except json.JSONDecodeError:
        print("错误: 文件内容不是有效的JSON格式")
    except Exception as e:
        print(f"发生错误: {str(e)}")
    
    return None


def time_to_seconds(time_str):
    """将HH:MM:SS格式的时间转换为总秒数"""
    try:
        # 匹配时间格式，支持可选的小时部分
        #print('准备转换时间:',time_str)
        match = re.match(r'^(?:(\d+):)?(\d+):(\d+)$', time_str)
        if not match:
            raise ValueError(f"无效的时间格式: {time_str}")
        
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        
        return hours * 3600 + minutes * 60 + seconds
    except Exception as e:
        #print(f"时间转换错误: {e}")
        return None

def extract_text_by_frame(audio_list, start_frame, end_frame):
    # 计算时间区间（每帧5秒）
    frame_interval = 5  # 每帧间隔5秒
    start_seconds = start_frame * frame_interval
    end_seconds = end_frame * frame_interval
    
    # 存储符合条件的文本
    matched_texts = []
    
    for item in audio_list:
        # 验证必要字段
        required_fields = ['timeStamp', 'startTime', 'data']
        if not all(field in item for field in required_fields):
            print("跳过不完整的音频信息项")
            continue
        
        # 转换时间戳为秒数
        item_start = time_to_seconds(item['startTime'])
        item_end = time_to_seconds(item['timeStamp'])
        
        if item_start is None or item_end is None:
            #print("跳过时间格式错误的项")
            continue
        
        # 更新判断逻辑：
        # 1. 如果语音片段开始于当前区间内，纳入
        # 2. 如果语音片段结束于当前区间内，纳入
        # 3. 如果语音片段完全覆盖当前区间，纳入
        # 4. 如果当前区间的结束时间点落在语音片段中，纳入
        # 这确保了一个片段可以属于多个连续区间
        if (item_start <= end_seconds and item_end >= start_seconds):
            matched_texts.append(item['data'])
    
    text = '。'.join(matched_texts)

    # 返回结果字典，key为区间尾部帧号
    return text


def get_all_image_paths(directory, include_subdirectories=True):
    """
    读取指定目录下的所有图片文件路径（支持常见图片格式）
    
    参数:
        directory (str): 要扫描的目录路径
        include_subdirectories (bool): 是否包含子目录，默认为True
        
    返回:
        list: 包含所有图片文件全路径的列表，若目录不存在或无图片则返回空列表
        
    支持的图片格式: .jpg, .jpeg, .png, .bmp, .gif, .webp, .tiff, .tif
    """
    from pathlib import Path
    # 定义支持的图片文件扩展名（不区分大小写）
    image_extensions = {
        '.jpg', '.jpeg', '.png', '.bmp', '.gif', 
        '.webp', '.tiff', '.tif', '.JPG', '.JPEG',
        '.PNG', '.BMP', '.GIF', '.WEBP', '.TIFF', '.TIF'
    }
    
    # 初始化结果列表
    image_paths = []
    
    try:
        # 转换为Path对象并获取绝对路径
        dir_path = Path(directory).absolute()
        
        # 检查目录是否存在
        if not dir_path.exists() or not dir_path.is_dir():
            print(f"警告: 目录 {directory} 不存在或不是有效目录")
            return image_paths
        
        # 递归遍历目录（包括子目录）
        if include_subdirectories:
            for file_path in dir_path.rglob('*'):
                if file_path.suffix in image_extensions and file_path.is_file():
                    image_paths.append(str(file_path))
        # 仅遍历当前目录
        else:
            for file_path in dir_path.iterdir():
                if file_path.suffix in image_extensions and file_path.is_file():
                    image_paths.append(str(file_path))
                    
    except Exception as e:
        print(f"读取目录时发生错误: {str(e)}")
        return []
    
    return image_paths

class MeetingClient:
    def __init__(self, session_id=None, server_url="http://localhost:5000"):
        self.server_url = server_url
        self.session_id = session_id
        self.frame_id = 0

    def start_meeting(self, meeting_name=None):
        url = f"{self.server_url}/api/meeting/start"
        response = requests.post(url, json=({"meeting_name": meeting_name} if meeting_name else {}))
        result = response.json()
        if result["success"]:
            self.session_id = result["session_id"]
            print(f"会议启动成功，session_id: {self.session_id}")
            return True
        print(f"启动失败: {result['message']}")
        return False

    def upload_frame(self, image_path):
        if not self.session_id:
            print("请先启动会议")
            return False
        url = f"{self.server_url}/api/meeting/upload"
        response = requests.post(url, json={
            "session_id": self.session_id,
            "image_path": image_path,
            "frame_id": self.frame_id
        })
        self.frame_id += 1
        print(datetime.now(), 'upload frame:', response.json())
        return response.json()["success"]

    def stop_meeting(self):
        if not self.session_id:
            return
        url = f"{self.server_url}/api/meeting/stop"
        response = requests.post(url, json={"session_id": self.session_id})
        print(f"停止结果: {response.json()['message']}")

    def query_meeting(self, query='', limit=10, mode=0):
        if not self.session_id:
            return
        url = f"{self.server_url}/api/meeting/query"
        response = requests.post(url, json={"session_id": self.session_id, 'query': query if query else '你好', 'limit':limit, 'mode': mode})
        print(f"查询结果: {response.json()}")
        return response.json()

    def summary_meeting(self):
        if not self.session_id:
            return
        url = f"{self.server_url}/api/meeting/summary"
        response = requests.post(url, json={"session_id": self.session_id})
        print(f"会议总结: {response.json()}")
        return response.json()
    
    def update_asr(self, audio_data):
        if not self.session_id:
            return
        url = f"{self.server_url}/api/meeting/update/asr"
        import ast
        ret = self.summary_meeting()
        test_data = ret['data']
        for idx, item in enumerate(test_data['frames']):
            item['asr'] = extract_text_by_frame(audio_data, item['start'], item['end'])
        response = requests.post(url, json={"session_id": self.session_id, "data":test_data})
        print(f"ASR更新: {response.json()}")
        return response.json()

    def test_upload_video(self, video_path, save_dir='./client_frames'):
        if not self.start_meeting("百应技术解耦专项会"):
           return
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # cap = cv2.VideoCapture(video_path)
        # fps = int(cap.get(cv2.CAP_PROP_FPS))
        try:
            # while cap.isOpened():
            #     for i in range(16*5):
            #         ret, frame = cap.read()
            #         if not ret:
            #             break
            #     if not ret:
            #         break
            #     frame_path = os.path.join(save_dir, f"frame_{self.frame_id}.jpg")
            #     cv2.imwrite(frame_path, frame)
            #     print(self.frame_id, '保存图片:', os.path.abspath(frame_path))
            #     if not self.upload_frame(os.path.abspath(frame_path)):
            #        print("上传失败，终止测试")
            #        break
                #time.sleep(0.1)

            #images = get_all_image_paths(r'C:\Users\SAS\Downloads\multi-modal-meeting-system\Teams_20250725_172359')
            #images = get_all_image_paths(r'C:\Users\SAS\Downloads\4f1afec7-cae1-4732-a5f3-c2fcd7cd5439\4f1afec7-cae1-4732-a5f3-c2fcd7cd5439 - Copy')
            images = get_all_image_paths(r"C:\Users\SAS\Downloads\test")
            for idx, image in enumerate(images):
                #print('start:', datetime.now())
                if not self.upload_frame(os.path.abspath(image)):
                   print("上传失败，终止测试")
                   break
                #time.sleep(3)
        finally:
            #cap.release()
            self.stop_meeting()
            # 查询会议结果
            while True:
                ret = self.summary_meeting()
                if not ret['success']:
                    if '会议分析中' in ret['message']:
                        time.sleep(1)
                    else:
                        break
                else:
                    break


# 1、LSM的strategy是什么    answer：第二张      结果：top2/并列第一
# 2、LSM的典型使用场景有哪些  answer：第三张     结果：top5，其他图片包含LSM和场景描述
# 3、LSM转写功能是怎样的 answer：第四张（跟ASR有关系） 结果：top1
# 4、LSM主要功能有哪些 answer：第一张、第四张    结果：top1, top3
# 5、LSM的产品框架和架构是怎样的    answer：第四张 第五张   结果：top1,top2

if __name__ == "__main__":
    client = MeetingClient(session_id='f999392b-d384-4b89-a905-868b9438b0bb')
    #audio = parse_json_file(r'C:\Users\SAS\Downloads\multi-modal-meeting-system\Teams_20250725_172359\Audio.json')
    #client.update_asr(audio)

    client.test_upload_video(r"voc.mp4")  # 替换为本地视频路径


    #client.query_meeting(query='LSM的典型使用场景有哪些', limit=3)  # mode - 0-混合模式， 1-多模态模式， 2-ocr模式, 3-asr模式
    

    
    

