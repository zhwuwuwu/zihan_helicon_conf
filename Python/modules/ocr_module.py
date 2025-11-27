# -*- coding: utf-8 -*-
import sys
import io

# 强制设置stdout和stderr为UTF-8编码
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

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
        
        # Step 1: 计算max_wh_ratio
        for ino in range(beg_img_no, end_img_no):
            h, w = img_crop_list[indices[ino]].shape[0:2]
            wh_ratio = w * 1.0 / h
            max_wh_ratio = max(max_wh_ratio, wh_ratio)
        
        # Step 2: 调整大小和归一化
        for ino in range(beg_img_no, end_img_no):
            norm_img = self.resize_norm_img(img_crop_list[indices[ino]], max_wh_ratio)
            norm_img = norm_img[np.newaxis, :]
            norm_img_batch.append(norm_img)

        norm_img_batch = np.concatenate(norm_img_batch)
        norm_img_batch = norm_img_batch.copy()
        
        return norm_img_batch 

    def visualize_ocr_results(self, image_path, dt_boxes, rec_res, output_path=None):
        """
        可视化OCR结果，在图像上绘制检测框和识别文本
        
        Args:
            image_path: 原始图像路径
            dt_boxes: 检测框列表（基于缩放后图像的坐标）
            rec_res: 识别结果列表 [(text, confidence), ...]
            output_path: 输出图像路径，如果为None则显示图像
        """
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        
        # 读取原始图像（未缩放）
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"无法读取图像: {image_path}")
            return
        
        # 转换为PIL图像以支持中文显示
        pil_image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # 加载中文字体
        try:
            # 尝试使用系统中文字体
            font_path = str(self.font_path) if hasattr(self, 'font_path') and self.font_path.exists() else None
            if font_path:
                font = ImageFont.truetype(font_path, 20)
                font_small = ImageFont.truetype(font_path, 16)
            else:
                # Windows系统字体路径
                import platform
                if platform.system() == 'Windows':
                    font = ImageFont.truetype('C:/Windows/Fonts/simhei.ttf', 20)
                    font_small = ImageFont.truetype('C:/Windows/Fonts/simhei.ttf', 16)
                else:
                    font = ImageFont.load_default()
                    font_small = ImageFont.load_default()
        except:
            # 如果加载失败，使用默认字体
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()
            print("警告: 无法加载中文字体，使用默认字体")
        
        # 为每个检测框绘制矩形和文本
        for idx, (box, (text, conf)) in enumerate(zip(dt_boxes, rec_res)):
            # 转换坐标为整数
            box = np.array(box).astype(np.int32)
            
            # 绘制检测框（绿色）- 将box转换为PIL格式的坐标列表
            box_points = [(int(pt[0]), int(pt[1])) for pt in box]
            draw.polygon(box_points, outline=(0, 255, 0), width=2)
            
            # 准备文本标签
            label = f"{text} ({conf:.2f})"
            
            # 在检测框上方绘制文本
            text_pos = (int(box[0][0]), max(0, int(box[0][1]) - 25))
            
            # 获取文本边界框
            bbox = draw.textbbox(text_pos, label, font=font_small)
            
            # 绘制文本背景（绿色）
            draw.rectangle(bbox, fill=(0, 255, 0))
            
            # 绘制文本（黑色）
            draw.text(text_pos, label, font=font_small, fill=(0, 0, 0))
            
            # 绘制索引号（红色）
            idx_pos = (int(box[0][0]), int(box[0][1]) + 5)
            draw.text(idx_pos, str(idx), font=font_small, fill=(255, 0, 0))
        
        # 转换回OpenCV格式
        vis_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # 保存或显示图像
        if output_path:
            cv2.imwrite(output_path, vis_image)
            print(f"可视化结果已保存到: {output_path}")
        else:
            cv2.imshow('OCR Results', vis_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return vis_image
    
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
            original_image = cv2.imread(image)
            if original_image is None:
                print("***frame is None")
                return ''
            
            # 保存原始图像尺寸
            original_height, original_width = original_image.shape[:2]
            
            # If the frame is larger than full HD, reduce size to improve the performance.
            #scale = 1280 / max(frame.shape)
            scale = 1920 / max(original_image.shape)
            frame = original_image
            if scale < 1:
                frame = cv2.resize(src=original_image, dsize=None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_AREA)
            
            # 保存缩放后的图像尺寸
            frame_height, frame_width = frame.shape[:2]
            
            # Preprocess the image for text detection.
            test_image = self.image_preprocess(frame, 640)

            # Measure processing time for text detection.
            start_time = time.time()
            # Perform the inference step.
            det_results = self.det_compiled_model([test_image])[self.det_output_layer]
            stop_time = time.time()

            # 【调试输出】检测模型推理结果统计
            print("\n===== DETECTION OUTPUT =====")
            print(f"Shape: {det_results.shape}")
            print(f"Stats: min={np.min(det_results):.4f}, max={np.max(det_results):.4f}, mean={np.mean(det_results):.4f}")
            positive_count = np.sum(det_results > 0.3)
            print(f"Pixels>0.3: {positive_count}/{det_results.size} ({100.0*positive_count/det_results.size:.1f}%)")
            print(f"Image: {original_width}x{original_height} -> {frame_width}x{frame_height}")
            print("============================\n")

            # Postprocessing for Paddle Detection.
            dt_boxes = self.post_processing_detection(frame, det_results)
            
            # 【关键修复】如果进行了缩放，需要将坐标映射回原始图像尺寸
            if original_width != frame_width or original_height != frame_height:
                scale_x = original_width / frame_width
                scale_y = original_height / frame_height
                
                print(f"[Python OCR] Restoring coordinates from scaled size ({frame_width}x{frame_height}) to original size ({original_width}x{original_height})")
                
                # 将所有检测框坐标映射回原始尺寸
                dt_boxes_original = []
                for box in dt_boxes:
                    box_original = []
                    for pt in box:
                        box_original.append([pt[0] * scale_x, pt[1] * scale_y])
                    dt_boxes_original.append(np.array(box_original))
                # 转换回numpy数组以保持与sorted_boxes兼容
                dt_boxes = np.array(dt_boxes_original)
            processing_times = []
            processing_times.append(stop_time - start_time)
            # Use processing times from last 200 frames.
            if len(processing_times) > 200:
                processing_times.popleft()
            processing_time_det = np.mean(processing_times) * 1000

            # Preprocess detection results for recognition.
            dt_boxes = processing.sorted_boxes(dt_boxes)
            batch_num = 6
            # 使用原始图像和已还原的坐标进行裁剪
            img_crop_list, img_num, indices = self.prep_for_rec(dt_boxes, original_image)
            print(f"Detected {len(dt_boxes)} text boxes")
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
                rec_results = self.rec_compiled_model([norm_img_batch])[self.rec_output_layer]
                
                # Postprocessing recognition results.
                postprocess_op = processing.build_post_process(processing.postprocess_params)
                rec_result = postprocess_op(rec_results)
                
                # 存储识别结果
                for rno in range(len(rec_result)):
                    original_idx = indices[beg_img_no + rno]
                    rec_res[original_idx] = rec_result[rno]
                if rec_res:
                    txts = [rec_res[i][0] for i in range(len(rec_res))]
                    scores = [rec_res[i][1] for i in range(len(rec_res))]
                    #print('text=', txts)
                    #full_text += txts
                    content = self.filter_ocr_results(rec_res=rec_res)
                    full_text += content
                    full_text += '\n'
            print("Recognition completed")
            
            # 输出详细识别结果
            print("\n" + "="*70)
            print("OCR识别结果详情")
            print("="*70)
            
            # 准备日志内容
            log_lines = []
            log_lines.append("\n" + "="*70)
            log_lines.append(f"图像: {image}")
            log_lines.append(f"检测到 {len(dt_boxes)} 个文本区域")
            log_lines.append("="*70 + "\n")
            
            # 可视化结果 - 过滤空结果以确保boxes和results一一对应
            if rec_res and dt_boxes:
                # 过滤出有效的识别结果并输出详情
                valid_boxes = []
                valid_results = []
                valid_idx = 0
                
                for box, (text, conf) in zip(dt_boxes, rec_res):
                    if text and len(text.strip()) > 0 and conf > 0:
                        valid_boxes.append(box)
                        valid_results.append((text, conf))
                        
                        # 输出到控制台
                        box_str = ", ".join([f"({int(pt[0])},{int(pt[1])})" for pt in box])
                        print(f"[{valid_idx}] 文本: {text} | 置信度: {conf:.4f} | 位置: [{box_str}]")
                        
                        # 写入日志
                        log_lines.append(f"[{valid_idx}] 文本: {text}")
                        log_lines.append(f"    置信度: {conf:.4f}")
                        log_lines.append(f"    边界框: [{box_str}]\n")
                        
                        valid_idx += 1
                
                print("-"*70)
                
                if valid_boxes and valid_results:
                    # 创建输出文件夹并生成输出文件名
                    from pathlib import Path
                    input_path = Path(image)
                    output_dir = input_path.parent / "output"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    output_path = str(output_dir / f"{input_path.stem}_ocr_result_py.jpg")
                    log_path = str(output_dir / f"{input_path.stem}_ocr_result_py.log")
                    
                    # 可视化
                    self.visualize_ocr_results(image, valid_boxes, valid_results, output_path)
                    
                    # 输出最终文本
                    print(f"\n最终文本 ({len(full_text)} 字符):\n{full_text}")
                    print("="*70 + "\n")
                    
                    # 写入日志文件
                    log_lines.append(f"最终文本: {full_text}")
                    log_lines.append("="*70)
                    
                    try:
                        with open(log_path, 'w', encoding='utf-8') as f:
                            f.write('\n'.join(log_lines))
                        print(f"日志已保存到: {log_path}")
                    except Exception as e:
                        print(f"警告: 无法保存日志文件: {e}")
            
            return full_text

        # any different error
        except RuntimeError as e:
            print(e)
        # finally:
        #     print('done')

# 使用示例
if __name__ == "__main__":
    ocr = PaddleOCRWithOpenVINO(models_dir='.\\models\\paddle_ocr', download_models=False)
    
    # 支持多个图像测试
    # test_images = [
    #     # "C:\\netshare\\test_imgs\\ocr_test1.png",
    #     # "C:\\netshare\\test_imgs\\ocr_test2.png",
    #     # "C:\\netshare\\test_imgs\\ocr_test3.png",
    #     # "C:\\netshare\\test_imgs\\ocr_test4.png",
    #     # 可以添加更多图像路径
    # ]
    test_images = []
    input_path = Path("C:\\netshare\\test_imgs\\group3").expanduser()
    if input_path.is_dir():
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
        # 排除output文件夹中的文件
        test_images = [str(p) for p in sorted(input_path.rglob('*')) 
                      if p.suffix.lower() in image_exts and 'output' not in p.parts]
    elif input_path.is_file():
        test_images = [str(input_path)]
    else:
        print(f" 输入路径不存在: {input_path}")
    if not test_images:
        print(" 未提供图像或目录，请在命令行参数中指定。")
    
    print(f"========== 开始测试 {len(test_images)} 个图像 ==========\n")
    
    for idx, image_path in enumerate(test_images):
        print(f"\n{'='*60}")
        print(f"测试图像 [{idx+1}/{len(test_images)}]: {image_path}")
        print(f"{'='*60}")
        
        if not os.path.exists(image_path):
            print(f" 警告: 图像不存在，跳过: {image_path}\n")
            continue
        
        try:
            start = time.time()
            text = ocr.run_paddle_ocr(image=image_path)
            elapsed = time.time() - start
            
            print(f"\n{'─'*60}")
            print(f"✓ 处理完成 - 耗时: {elapsed:.3f}秒")
            print(f"识别文本: {text if text else '(未识别到文本)'}")
            print(f"{'─'*60}\n")
            
        except Exception as e:
            print(f"\n✗ 错误: 处理图像时出错: {e}\n")
    
    print(f"\n{'='*60}")
    print("所有测试完成")
    print(f"{'='*60}")