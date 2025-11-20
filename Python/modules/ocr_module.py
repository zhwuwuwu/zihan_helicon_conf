import sys
import cv2
import numpy as np
import math
import time
import collections
from PIL import Image
from pathlib import Path
import tarfile
import os
import shutil
import requests
from typing import List, Dict, Tuple, Optional

import openvino as ov

import sys
import cv2
import numpy as np
import paddle
import math
import time
import collections
from PIL import Image
from pathlib import Path
import tarfile

import openvino as ov
import copy

# Import local modules

utils_file_path = Path('../utils/notebook_utils.py')
notebook_directory_path = Path('.')

if not utils_file_path.exists():
   # !git clone --depth 1 https://github.com/igor-davidyuk/openvino_notebooks.git -b moving_data_to_cloud openvino_notebooks
    utils_file_path = Path('./openvino_notebooks/notebooks/utils/notebook_utils.py')
    notebook_directory_path = Path('./openvino_notebooks/notebooks/405-paddle-ocr-webcam/')

# print("PTAHS")
# print(utils_file_path)
# print(notebook_directory_path)

sys.path.append(str(utils_file_path.parent))
sys.path.append(str(notebook_directory_path))

# files = list(notebook_directory_path.glob("**/*"))
# print("FILES:", files)

#import notebook_utils as utils
import pre_post_processing as processing

import json
from pathlib import Path

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

class JSONConfigReader:
    def __init__(self, config_path):
        self.config_path = os.path.join(get_executable_dir(), config_path)
        self.config_data = self._load_config()
    
    def _load_config(self):
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"配置文件不存在: {self.config_path}")
            return {}
        except json.JSONDecodeError as e:
            print(f"配置文件解析错误: {e}")
            return {}
    
    def get(self, key, default=None):
        """获取配置值，支持嵌套键，如 "database.host" """
        keys = key.split('.')
        value = self.config_data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value


class PaddleOCRWithOpenVINO:
    """基于OpenVINO的PaddleOCR文字识别类，支持图片文字检测与识别"""
    
    def __init__(self, models_dir: str = ".\\models\\paddle_ocr", download_models: bool = True):
        """
        初始化OCR识别器
        
        Args:
            models_dir: 模型存储目录
            download_models: 是否自动下载模型
        """
        config = JSONConfigReader("config.json")
        print(config.config_path)
        # 获取直接配置项
        device = config.get("app.ocr_device")
        print('OCR device =', device)
        self.device = device if device else "CPU"
        self.models_dir = Path(models_dir)
        self.det_model_path = self.models_dir / "ch_PP-OCRv3_det_infer/inference.pdmodel"
        self.rec_model_path = self.models_dir / "ch_PP-OCRv3_rec_infer/inference.pdmodel"
        self.fonts_dir = self.models_dir / "fonts"
        self.font_path = self.fonts_dir / "simfang.ttf"
        self.char_dict_path = self.fonts_dir / "ppocr_keys_v1.txt"
        #print(self.fonts_dir, self.char_dict_path)
        # 模型相关
        self.core = ov.Core()
        self.det_compiled_model = None
        self.rec_compiled_model = None
        self.char_dict = None
        
        # 下载模型和资源
        if download_models:
            self._download_resources()
        
        # 加载模型
        self._load_models()
        # 加载字符字典
        self._load_char_dict()

        # 后处理参数
        self.postprocess_params = {
            'name': 'PPOCRPostProcess',
            'label_file_path': str(self.char_dict_path),
            'use_space_char': True
        }

        print("==== INIT & LOAD FINISHED ====")
        
    def _download_resources(self):
        """下载模型和资源文件"""
        # 创建目录
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.fonts_dir.mkdir(parents=True, exist_ok=True)
        
        # 下载检测模型
        det_model_url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/paddle-ocr/ch_PP-OCRv3_det_infer.tar"
        self._download_and_extract(det_model_url, self.det_model_path)
        
        # 下载识别模型
        rec_model_url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/paddle-ocr/ch_PP-OCRv3_rec_infer.tar"
        self._download_and_extract(rec_model_url, self.rec_model_path)
        
        # 下载字体文件
        font_url = "https://raw.githubusercontent.com/Halfish/lstm-ctc-ocr/master/fonts/simfang.ttf"
        self._download_file(font_url, self.font_path)
        
        # 下载字符字典
        char_dict_url = "https://raw.githubusercontent.com/WenmuZhou/PytorchOCR/master/torchocr/datasets/alphabets/ppocr_keys_v1.txt"
        self._download_file(char_dict_url, self.char_dict_path)
        
        # 加载字符字典
        self._load_char_dict()
    
    def _download_and_extract(self, url: str, target_file: Path):
        """下载并解压模型文件"""
        if target_file.exists():
            #print(f"模型已存在: {target_file}")
            return
        
        print(f"下载模型: {url}")
        file_name = url.split("/")[-1]
        archive_path = self.models_dir / file_name
        
        # 下载文件
        self._download_file(url, archive_path)
        
        # 解压文件
        print(f"解压模型: {archive_path}")
        try:
            with tarfile.open(archive_path, 'r') as tar:
                tar.extractall(self.models_dir)
            #print(f"模型解压完成: {target_file.parent}")
        except Exception as e:
            print(f"解压失败: {e}")
            if archive_path.exists():
                archive_path.unlink()
            raise
    
    def _download_file(self, url: str, target_path: Path):
        """下载单个文件"""
        if target_path.exists():
            return
        
        #print(f"下载文件: {url} -> {target_path}")
        try:
            response = requests.get(url, stream=True, timeout=30)
            if response.status_code == 200:
                with open(target_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                #print(f"文件下载完成: {target_path}")
            else:
                print(f"下载失败，状态码: {response.status_code}")
                raise Exception(f"下载失败，状态码: {response.status_code}")
        except Exception as e:
            print(f"下载过程出错: {e}")
            if target_path.exists():
                target_path.unlink()
            raise
    
    def _load_char_dict(self):
        """加载字符字典"""
        # print("CHAR DICT PATH", self.char_dict_path)
        if not self.char_dict_path.exists():
            raise FileNotFoundError(f"字符字典不存在: {self.char_dict_path}")
        
        with open(self.char_dict_path, 'r', encoding='utf-8') as f:
            self.char_dict = [line.strip() for line in f.readlines()]

        # print("DICT", self.char_dict)
    
    def _load_models(self):
        """加载检测和识别模型"""
        # 加载检测模型
        if not self.det_model_path.exists():
            raise FileNotFoundError(f"检测模型不存在: {self.det_model_path}")
        
        #print("加载检测模型...")
        det_model = self.core.read_model(model=str(self.det_model_path))
        self.det_compiled_model = self.core.compile_model(
            model=det_model,
            device_name=self.device
        )
        self.det_input_layer = self.det_compiled_model.input(0)
        self.det_output_layer = self.det_compiled_model.output(0)
        #print("检测模型加载完成")
        
        # 加载识别模型
        if not self.rec_model_path.exists():
            raise FileNotFoundError(f"识别模型不存在: {self.rec_model_path}")
        
        #print("加载识别模型...")
        rec_model = self.core.read_model(model=str(self.rec_model_path))
        
        # 设置动态形状
        for input_layer in rec_model.inputs:
            input_shape = input_layer.partial_shape
            input_shape[3] = -1
            rec_model.reshape({input_layer: input_shape})
        
        self.rec_compiled_model = self.core.compile_model(
            model=rec_model,
            device_name=self.device
        )
        self.rec_input_layer = self.rec_compiled_model.input(0)
        self.rec_output_layer = self.rec_compiled_model.output(0)
        #print("识别模型加载完成")
        

    # Preprocess for text detection.
    def image_preprocess(self, input_image, size):
        """
        Preprocess input image for text detection

        Parameters:
            input_image: input image
            size: value for the image to be resized for text detection model
        """
        img = cv2.resize(input_image, (size, size))
        img = np.transpose(img, [2, 0, 1]) / 255
        img = np.expand_dims(img, 0)
        # NormalizeImage: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], is_scale: True}
        img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
        img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
        img -= img_mean
        img /= img_std
        return img.astype(np.float32)
    
    def post_processing_detection(self, frame, det_results):
        """
        Postprocess the results from text detection into bounding boxes

        Parameters:
            frame: input image
            det_results: inference results from text detection model
        """
        ori_im = frame.copy()
        data = {'image': frame}
        data_resize = processing.DetResizeForTest(data)
        data_list = []
        keep_keys = ['image', 'shape']
        for key in keep_keys:
            data_list.append(data_resize[key])
        img, shape_list = data_list

        shape_list = np.expand_dims(shape_list, axis=0)
        pred = det_results[0]
        if isinstance(pred, paddle.Tensor):
            pred = pred.numpy()
        segmentation = pred > 0.3

        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            mask = segmentation[batch_index]
            boxes, scores = processing.boxes_from_bitmap(pred[batch_index], mask, src_w, src_h)
            boxes_batch.append({'points': boxes})
        post_result = boxes_batch
        dt_boxes = post_result[0]['points']
        #print('dt box=', dt_boxes)
        dt_boxes = processing.filter_tag_det_res(dt_boxes, ori_im.shape)
        return dt_boxes


    # Preprocess for text recognition.
    def resize_norm_img(self, img, max_wh_ratio):
        """
        Resize input image for text recognition

        Parameters:
            img: bounding box image from text detection
            max_wh_ratio: value for the resizing for text recognition model
        """
        rec_image_shape = [3, 48, 320]
        imgC, imgH, imgW = rec_image_shape
        assert imgC == img.shape[2]
        character_type = "ch"
        if character_type == "ch":
            imgW = int((32 * max_wh_ratio))
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im


    def prep_for_rec(self, dt_boxes, frame):
        """
        Preprocessing of the detected bounding boxes for text recognition

        Parameters:
            dt_boxes: detected bounding boxes from text detection
            frame: original input frame
        """
        ori_im = frame.copy()
        img_crop_list = []
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            #print('tmp_box=', tmp_box)
            img_crop = processing.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)

        img_num = len(img_crop_list)
        # Calculate the aspect ratio of all text bars.
        width_list = []
        for img in img_crop_list:
            width_list.append(img.shape[1] / float(img.shape[0]))

        # Sorting can speed up the recognition process.
        indices = np.argsort(np.array(width_list))
        return img_crop_list, img_num, indices


    def batch_text_box(self, img_crop_list, img_num, indices, beg_img_no, batch_num):
        """
        Batch for text recognition

        Parameters:
            img_crop_list: processed detected bounding box images
            img_num: number of bounding boxes from text detection
            indices: sorting for bounding boxes to speed up text recognition
            beg_img_no: the beginning number of bounding boxes for each batch of text recognition inference
            batch_num: number of images for each batch
        """
        norm_img_batch = []
        max_wh_ratio = 0
        end_img_no = min(img_num, beg_img_no + batch_num)
        for ino in range(beg_img_no, end_img_no):
            h, w = img_crop_list[indices[ino]].shape[0:2]
            wh_ratio = w * 1.0 / h
            max_wh_ratio = max(max_wh_ratio, wh_ratio)
        for ino in range(beg_img_no, end_img_no):
            norm_img = self.resize_norm_img(img_crop_list[indices[ino]], max_wh_ratio)
            norm_img = norm_img[np.newaxis, :]
            norm_img_batch.append(norm_img)

        norm_img_batch = np.concatenate(norm_img_batch)
        norm_img_batch = norm_img_batch.copy()
        return norm_img_batch 

    def filter_ocr_results(self, rec_res, min_confidence=0.95, remove_special=True, remove_short=True):
        """
        过滤OCR识别结果
        
        Args:
            rec_res: OCR识别结果列表
            min_confidence: 最小置信度阈值，低于此值的结果将被过滤
            remove_special: 是否移除包含特殊字符的结果
            remove_short: 是否移除过短的文本（如单个字符）
            
        Returns:
            过滤后的字符串，用空格连接
        """
        filtered_texts = []
        
        for text, confidence in rec_res:
            # 处理特殊格式
            if isinstance(text, list):  # 处理类似 ['', 0.0] 的情况
                if len(text) >= 2:
                    text, confidence = text[0], text[1]
                else:
                    continue
            elif isinstance(confidence, str):  # 处理置信度为字符串的情况
                try:
                    confidence = float(confidence)
                except:
                    continue
            
            # 过滤空字符串和NaN
            if not text or confidence != confidence:  # 检查是否为NaN
                continue
            # 过滤低置信度结果
            if confidence < min_confidence:
                continue
            # 过滤特殊字符
            if remove_special and not text.isalnum() and not any(c.isalnum() for c in text):
                continue
            # 过滤过短的文本
            if remove_short and len(text.strip()) <= 3:
                continue
            # 清理文本
            cleaned_text = text.strip()
            if cleaned_text:
                filtered_texts.append(cleaned_text)
        
        return " ".join(filtered_texts)


    def run_paddle_ocr(self, image='', flip=False, use_popup=False, skip_first_frames=0):
        """
        Main function to run the paddleOCR inference:
        1. Create a video player to play with target fps (utils.VideoPlayer).
        2. Prepare a set of frames for text detection and recognition.
        3. Run AI inference for both text detection and recognition.
        4. Visualize the results.

        Parameters:
            source: The webcam number to feed the video stream with primary webcam set to "0", or the video path.
            flip: To be used by VideoPlayer function for flipping capture image.
            use_popup: False for showing encoded frames over this notebook, True for creating a popup window.
            skip_first_frames: Number of frames to skip at the beginning of the video.
        """
        # Create a video player to play with target fps.
        try:
            # Grab the frame.
            frame = cv2.imread(image)
            if frame is None:
                print("***frame is None")
                return ''
            # If the frame is larger than full HD, reduce size to improve the performance.
            #scale = 1280 / max(frame.shape)
            scale = 1920 / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(src=frame, dsize=None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_AREA)
            # Preprocess the image for text detection.
            test_image = self.image_preprocess(frame, 640)

            # Measure processing time for text detection.
            start_time = time.time()
            # Perform the inference step.
            det_results = self.det_compiled_model([test_image])[self.det_output_layer]
            stop_time = time.time()

            # Postprocessing for Paddle Detection.
            dt_boxes = self.post_processing_detection(frame, det_results)
            processing_times = []
            processing_times.append(stop_time - start_time)
            # Use processing times from last 200 frames.
            if len(processing_times) > 200:
                processing_times.popleft()
            processing_time_det = np.mean(processing_times) * 1000

            # Preprocess detection results for recognition.
            dt_boxes = processing.sorted_boxes(dt_boxes)
            batch_num = 6
            img_crop_list, img_num, indices = self.prep_for_rec(dt_boxes, frame)
            print("BOXES:", dt_boxes)
            print("==== DET TEXTUAL BBOX DONE ====")
            # print("IMG CROP LIST", img_crop_list)
            # For storing recognition results, include two parts:
            # txts are the recognized text results, scores are the recognition confidence level.
            rec_res = [['', 0.0]] * img_num
            txts = []
            scores = []
            full_text = ''
            #print('img num', img_num, batch_num)
            for beg_img_no in range(0, img_num, batch_num):

                # Recognition starts from here.
                norm_img_batch = self.batch_text_box(
                    img_crop_list, img_num, indices, beg_img_no, batch_num)
                
                # Run inference for text recognition.
                #print('type norm_img_batch', norm_img_batch.shape)
                #print('type self.rec_output_layer', type(self.rec_output_layer), self.rec_output_layer)
                rec_results = self.rec_compiled_model([norm_img_batch])[self.rec_output_layer]
                #print('type rec_results', type(rec_results), rec_results)
                # Postprocessing recognition results.
                postprocess_op = processing.build_post_process(processing.postprocess_params)
                rec_result = postprocess_op(rec_results)
                for rno in range(len(rec_result)):
                    rec_res[indices[beg_img_no + rno]] = rec_result[rno]
                print('rec_res=', rec_res)
                if rec_res:
                    txts = [rec_res[i][0] for i in range(len(rec_res))]
                    scores = [rec_res[i][1] for i in range(len(rec_res))]
                    #print('text=', txts)
                    #full_text += txts
                    content = self.filter_ocr_results(rec_res=rec_res)
                    full_text += content
                    #print('text=', content)
            print("==== REC DONE ====")
            return full_text

        # any different error
        except RuntimeError as e:
            print(e)
        # finally:
        #     print('done')

# 使用示例
if __name__ == "__main__":
    ocr = PaddleOCRWithOpenVINO(models_dir='.\\models\\paddle_ocr', download_models=False)
    image_path = "C:\\Users\\zihanwu\\Downloads\\ocr_test1.png"

    for i in range(1):
        start = time.time()
        text = ocr.run_paddle_ocr(image=image_path)
        print(f'ROUND{i} - TIME', time.time()-start)
    print("Extracted Text:", text)