import cv2
import numpy as np
import time
import threading
import os, sys
import json
from queue import Queue, Empty
from flask import Flask, request, jsonify
from uuid import uuid4
from datetime import datetime
import argparse
# 导入所有模块
from modules.camera_module import CameraModule
from modules.screenshot_module import ScreenshotModule
from modules.meeting_manager import MeetingSession, active_sessions, sessions_lock, SESSION_SAVE_DIR
from modules.meeting_manager import GLOBAL_HYBRID_SEARCHER, GLOBAL_CHANGE_DETECTOR, GLOBAL_RERANK_RERANKER

from nlp.query import EsQueryer

# --------------------------- 全局配置 ---------------------------
ENABLE_UI = False                 # 默认关闭UI
ENABLE_CAMERA = False             # 默认关闭摄像头捕获


# --------------------------- Flask初始化 ---------------------------
flask_app = Flask(__name__)

# def get_temp_directory():
#     if hasattr(sys, '_MEIPASS'):
#         # 如果程序是打包后运行的，返回解压后的临时目录
#         print('_MEIPASS=', sys._MEIPASS)
#         return sys._MEIPASS
#     else:
#         return None
# if get_temp_directory():
#     sys.path.append(get_temp_directory())

# --------------------------- Flask接口实现 ---------------------------
import cv2
import numpy as np
from PIL import Image


def crop_ppt_fixed(input_path, output_path, top=0, bottom=0, left=0, right=0):
    """
    裁剪图片
    
    Args:
        input_path (str): 输入图片路径
        output_path (str): 输出图片路径
        top (int): 从顶部裁剪的像素数
        bottom (int): 从底部裁剪的像素数
        left (int): 从左侧裁剪的像素数
        right (int): 从右侧裁剪的像素数
    
    Returns:
        bool: 是否成功
    """
    try:
        # 读取图片
        image = cv2.imread(input_path)
        if image is None:
            print(f"无法读取图片: {input_path}")
            return False
        
        height, width = image.shape[:2]
        
        # 计算裁剪区域
        y1 = top
        y2 = height - bottom
        x1 = left  
        x2 = width - right
        
        # 检查裁剪参数是否合理
        if x2 <= x1 or y2 <= y1:
            print("裁剪参数错误，剩余区域无效")
            return False
        
        # 裁剪图片
        cropped = image[y1:y2, x1:x2]
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存图片
        cv2.imwrite(output_path, cropped)
        
        #print(f"裁剪成功: {width}x{height} -> {x2-x1}x{y2-y1}")
        return True
        
    except Exception as e:
        #print(f"裁剪失败: {e}")
        return False

def crop_ppt_vertical(image_path, output_path, threshold=30, min_height=20, scan_width_ratio=0.3):
    """
    垂直方向PPT内容裁剪函数（简化版）
    
    Args:
        image_path (str): 输入图片路径
        output_path (str): 输出图片路径
        threshold (int): 像素值阈值，默认30
        min_height (int): 最小保留高度（像素），默认20
        scan_width_ratio (float): 扫描宽度比例，默认0.3
    
    Returns:
        bool: 是否进行了裁剪
    """
    
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    
    # 检测上下边缘的裁剪位置
    top_crop, bottom_crop = detect_vertical_bounds(image_rgb, threshold, min_height, scan_width_ratio)
    
    # 判断是否需要裁剪
    needs_cropping = (top_crop > 0) or (bottom_crop < h - 1)
    crop_height = bottom_crop - top_crop + 1
    
    if not needs_cropping or crop_height < 2 * min_height:
        # 无需裁剪或裁剪后太小，保存原图
        pil_image = Image.fromarray(image_rgb)
        pil_image.save(output_path)
        return False
    
    # 执行裁剪并保存
    cropped = image_rgb[top_crop:bottom_crop+1, :]
    pil_image = Image.fromarray(cropped)
    pil_image.save(output_path)
    
    return True

def detect_vertical_bounds(image_rgb, threshold, min_height, scan_width_ratio):
    """
    检测垂直方向的裁剪边界
    """
    h, w = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # 计算扫描区域（图片中央部分）
    scan_width = int(w * scan_width_ratio)
    start_x = (w - scan_width) // 2
    end_x = start_x + scan_width
    
    # 提取中央区域并计算每行的平均像素值
    central_region = gray[:, start_x:end_x]
    row_means = np.mean(central_region, axis=1)
    
    # 从上边缘开始扫描
    top_crop = 0
    for i in range(h // 2):
        if row_means[i] > threshold and has_consecutive_valid_rows(row_means, i, threshold, 20, 'down'):
            top_crop = i
            break
    
    # 从下边缘开始扫描
    bottom_crop = h - 1
    for i in range(h - 1, h // 2 - 1, -1):
        if row_means[i] > threshold and has_consecutive_valid_rows(row_means, i, threshold, 20, 'up'):
            bottom_crop = i
            break
    
    # 验证裁剪区域的合理性
    if top_crop > 0 and top_crop < min_height:
        top_crop = 0
    
    if (h - 1 - bottom_crop) > 0 and (h - 1 - bottom_crop) < min_height:
        bottom_crop = h - 1
    
    return top_crop, bottom_crop

def has_consecutive_valid_rows(row_means, start_idx, threshold, min_consecutive, direction):
    """
    检查是否有足够的连续有效行
    """
    count = 0
    if direction == 'down':
        for i in range(start_idx, min(start_idx + min_consecutive * 2, len(row_means))):
            if row_means[i] > threshold:
                count += 1
                if count >= min_consecutive:
                    return True
            else:
                count = 0
    else:  # direction == 'up'
        for i in range(start_idx, max(start_idx - min_consecutive * 2, -1), -1):
            if row_means[i] > threshold:
                count += 1
                if count >= min_consecutive:
                    return True
            else:
                count = 0
    
    return count >= min_consecutive

#---------------------------------------------------------------------#


@flask_app.route('/api/meeting/start', methods=['POST'])
def start_meeting():
    data = request.json or {}
    print('客户端请求服务[start]:', data)
    meeting_name = data.get('meeting_name')
    if not meeting_name:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        meeting_name = f"视频会议_{now}"
    
    session_id = str(uuid4())
    with sessions_lock:
        active_sessions[session_id] = MeetingSession(
            session_id=session_id,
            meeting_name=meeting_name
        )
        threading.Thread(target=active_sessions[session_id].start_capture, daemon=True).start()
        threading.Thread(target=active_sessions[session_id].start_analysis, daemon=True).start()
        threading.Thread(target=active_sessions[session_id].monitor_timeout, daemon=True).start()

    return jsonify({
        "session_id": session_id,
        "success": True,
        "message": f"会议 {meeting_name} 已启动"
    })

@flask_app.route('/api/meeting/stop', methods=['POST'])
def stop_meeting():
    data = request.json or {}
    print('客户端请求服务[stop]:', data)
    session_id = data.get('session_id')
    if not session_id or session_id not in active_sessions:
        return jsonify({"success": False, "message": "无效的参数<session_id>"})

    session = active_sessions[session_id]
    session.stop()
    return jsonify({"success": True, "message": "会议停止请求已接收"})


@flask_app.route('/api/meeting/summary', methods=['POST'])
def meeting_summary():
    data = request.json or {}
    print('客户端请求服务[summary]:', data)
    session_id = data.get('session_id')
    if not session_id:
        print('服务端返回结果[summary]:', {"success": False, "message": "缺少必要参数<session_id>", "data": {}})
        return jsonify({"success": False, "message": "缺少必要参数<session_id>", "data": {}})

    if session_id in active_sessions:
        session = active_sessions[session_id]
        if not session.session_status:
            print('服务端返回结果[summary]:', {"success": False, "message": "会议分析中，请稍后", "data": {}}, '(%s, %s)' % (session.frame_queue.qsize(), session.keyframe_queue.qsize()))
            return jsonify({"success": False, "message": "会议分析中，请稍后", "data": {}})
        print('服务端返回结果[summary]:', {
            "success": True, 
            "message": "查询成功", 
            "data":
            {
                "session_id": session_id,
                "meeting_name": session.meeting_name,
                "frames": session.key_frames
            }
        })
        return jsonify({
            "success": True, 
            "message": "查询成功", 
            "data":
            {
                "session_id": session_id,
                "meeting_name": session.meeting_name,
                "frames": session.key_frames
            }
        })
    else:
        ret = GLOBAL_HYBRID_SEARCHER.search_by_sessionid(session_id=session_id)
        if not ret:
            print('服务端返回结果[summary]:', {"success": True, "message": "查询成功", "data": {}})
            return jsonify({"success": True, "message": "查询成功", "data": {}})
        data = {'session_id': session_id, 'meeting_name': ret[0]['meeting_name']}
        for item in ret:
            del item['session_id']
            del item['meeting_name']
        data['frames'] = ret
        print('服务端返回结果[summary]:', {
            "success": True, 
            "message": "查询成功", 
            "data": data
        })
        return jsonify({
            "success": True, 
            "message": "查询成功", 
            "data": data
        })


@flask_app.route('/api/meeting/query', methods=['POST'])
def query_meeting():
    data = request.json or {}
    print('客户端请求服务[query]:', data)
    session_id = data.get('session_id')
    query = data.get('query')
    limit = data.get('limit', 10) # 10
    mode = data.get('mode', 0)

    if not session_id or not query:
        print('服务端返回结果[query]:', {"success": False, "message": "缺少必要参数<session_id> <query>", "data": {}})
        return jsonify({"success": False, "message": "缺少必要参数<session_id> <query>", "data": {}})
    if session_id in active_sessions:
        session = active_sessions[session_id]
        if not session.session_status:
            print('服务端返回结果[query]:', {"success": False, "message": "会议分析中，请稍后", "data": {}}, '(%s, %s)' % (session.frame_queue.qsize(), session.keyframe_queue.qsize()))
            return jsonify({"success": False, "message": "会议分析中，请稍后", "data": {}})

    # 从数据库搜索图片
    query_embed_list = GLOBAL_CHANGE_DETECTOR.bgem3_model.encode([query])[0]
    #print(query_embed_list)
    query_dict_list = []
    for qe in query_embed_list:
        query_dict_list.append({'query': query, 'qv': qe.tolist()})
    ret = GLOBAL_HYBRID_SEARCHER.search_by_vector(session_id=session_id, queryList=query_dict_list, limit=limit, mode=mode)
    if not ret:
        print('服务端返回结果[query]:', {"success": True, "message": "查询成功", "data": {}})
        return jsonify({"success": True, "message": "查询成功", "data": {}})
    # rerank
    from modules.ocr_module import JSONConfigReader
    config = JSONConfigReader("config.json")
    enable_rerank = config.get("app.enable_rerank", False)
    index = []
    if enable_rerank:
        desc_list = []
        for item in ret:
            desc_list.append(item['desc'])
        rerank_ret = GLOBAL_RERANK_RERANKER.similarity(query, desc_list)
        index = np.argsort(rerank_ret * -1)[0:limit]
        #print('相似度:', rerank_ret, index)
    data = {'session_id': session_id, 'meeting_name': ret[0]['meeting_name']}
    for item in ret:
        del item['session_id']
        del item['meeting_name']
    rerank_list = [ret[i] for i in index[:limit]] if enable_rerank else ret[:limit]

    # token sim
    qryr = EsQueryer()
    keywords = qryr.question(query)
    ocr_tks = []
    for item in rerank_list:
        tks = item['ocr_tks']
        ocr_tks.append(tks)
    tkweight = 0.2
    vtweight = 0.8
    tksim = qryr.token_similarity(keywords, ocr_tks)
    vtsim = [item['score'] for item in rerank_list]
    sim = tkweight*np.array(tksim) + vtweight*np.array(vtsim)
    index = np.argsort(sim * -1)[0:limit]
    rerank_list = [rerank_list[i] for i in index[:limit]]

    print('\n客户端查询:', query,'搜索结果: %s 张' % len(rerank_list), 'mode=%s' % mode, 'limit=', limit, '(重排序=%s)' % index[:limit] if enable_rerank else '')
    for idx, item in enumerate(rerank_list):
        print('------------------------------------------------------------------------------------------------')
        print('第%s张:' % (idx+1), '分数:', item['score'])
        print('名称:', item['file'])
    print('------------------------------------------------------------------------------------------------')
    data['frames'] = rerank_list
    import copy
    copy_data = copy.deepcopy(data)
    for frame in copy_data['frames']:
        frame['desc'] = frame['desc'][0:10] + '...'
        frame['ocr'] = frame['ocr'][0:10] + '...'
        frame['ocr_tks'] = frame['ocr_tks'][0:10]
        frame['asr'] = frame['asr'][0:10] + '...'

    print('服务端返回结果[query]:', {
        "success": True, 
        "message": "查询成功", 
        "data": copy_data
    })
    return jsonify({
        "success": True, 
        "message": "查询成功", 
        "data": data
    })


@flask_app.route('/api/meeting/upload', methods=['POST'])
def upload_frame():
    data = request.json or {}
    #print('upload time:', datetime.now())
    session_id = data.get('session_id')
    image_path = data.get('image_path')
    frame_id = data.get('frame_id')

    if 'session_id' not in data.keys() or 'image_path' not in data.keys() or 'frame_id' not in data.keys():
        return jsonify({"success": False, "message": "缺少必要参数 <session_id> <image_path> <frame_id>"})
    if session_id not in active_sessions:
        return jsonify({"success": False, "message": "无效的session_id"})
    if not os.path.exists(image_path):
        return jsonify({"success": False, "message": "图片路径不存在: %s" % image_path})

    session = active_sessions[session_id]
    #print('upload time 1:', datetime.now())

    success, msg = session.add_frame(os.path.abspath(image_path), int(frame_id))
    #print('upload time 2:', datetime.now())
    print('客户端请求服务[upload]:', os.path.abspath(image_path), '(%s, %s)' % (session.frame_queue.qsize(), session.keyframe_queue.qsize()))
    #print('upload time 3:', datetime.now())
    return jsonify({"success": success, "message": msg})


@flask_app.route('/api/meeting/update/asr', methods=['POST'])
def update_asr():
    data = request.json or {}
    session_id = data.get('session_id')
    client_data = data.get('data')
    print('客户端请求服务[asr]:', data)

    if 'session_id' not in data.keys() or 'data' not in data.keys():
        return jsonify({"success": False, "message": "缺少必要参数 <session_id> <data>"})
    #if session_id not in active_sessions:
    #    return jsonify({"success": False, "message": "无效的session_id"})
    
    # update data to qdrant db
    if not client_data or 'session_id' not in client_data.keys() or 'frames' not in client_data.keys():
        return jsonify({"success": False, "message": "无效的data"})
    try:
        cnt = 0
        for frame in client_data['frames']: 
            asr_vec = GLOBAL_CHANGE_DETECTOR.bgem3_model.encode([frame['asr']])[0][0]
            ret =GLOBAL_HYBRID_SEARCHER.update_points(session_id, frame, asr_vec)
            if ret:
                cnt = cnt + 1
    except BaseException as e:
        print('客户端请求服务[asr]: 更新ASR信息出错',e)
        return jsonify({"success": False, "message": "更新ASR信息出错，查看服务器日志获取更多信息！"})
    
    if cnt:
        print('客户端请求服务[asr]: ASR更新完成，共 %s 条' % cnt)
        return jsonify({"success": True, "message": '更新ASR数据成功'})
    else:
        print('客户端请求服务[asr]: 无效更新，请检查session_id或ASR数据是否有效！')
        return jsonify({"success": False, "message": '无效更新，请检查session_id或ASR数据是否有效！'})

# --------------------------- 启动逻辑 ---------------------------
if __name__ == "__main__":
    flask_app.run(host='127.0.0.1', port=5000, debug=False)
    