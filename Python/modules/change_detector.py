import numpy as np
import torch
from PIL import Image
from cn_clip.clip import load_from_name
import cn_clip.clip as clip
import time

#from paddleocr import PPStructureV3
#from FlagEmbedding import BGEM3FlagModel

from modules.ocr_module import PaddleOCRWithOpenVINO
from modules.bgem3_module import Model_BGE_M3, OpenVINOEmbedding_ov

class ChangeDetector:
    def __init__(self, similarity_threshold=0.93, history_size=5):
        self.threshold = similarity_threshold
        self.history_size = history_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = CHINESECLIP(device=self.device)
        #self.bgem3_model = Model_BGE_M3(use_ov=True)
        self.bgem3_model = OpenVINOEmbedding_ov()# 
        self.ocr_model = PaddleOCRWithOpenVINO()
        # self.ocr_model = PPStructureV3(
        #             use_doc_orientation_classify=False,
        #             use_doc_unwarping=False
        #         )

    def smart_interval_extract(self, text, target_length):
        """
        根据目标结果长度自动计算间隔，均匀提取文本信息
        
        参数:
        text: 原始文本内容
        target_length: 需要获取的结果长度
        
        返回:
        提取后的文本片段和实际使用的间隔
        """
        if not text or target_length <= 0:
            return '', 0
        
        text_length = len(text)
        
        # 如果目标长度大于等于原文长度，直接返回全文
        if target_length >= text_length:
            return text#, 1
        
        # 计算理论最佳间隔
        ideal_interval = max(1, text_length / target_length)
        
        # 尝试理想间隔及其附近值，找到最接近目标长度的结果
        candidates = [
            max(1, int(ideal_interval)),
            max(1, int(ideal_interval) - 1),
            max(1, int(ideal_interval) + 1)
        ]
        
        best_interval = 1
        min_diff = float('inf')
        
        for interval in candidates:
            # 计算使用当前间隔时能提取的文本长度
            actual_length = (text_length + interval - 1) // interval
            
            # 计算与目标长度的绝对差
            diff = abs(actual_length - target_length)
            
            if diff < min_diff or (diff == min_diff and interval < best_interval):
                min_diff = diff
                best_interval = interval
        
        # 使用最佳间隔提取文本
        result = text[0:text_length:best_interval]
        
        return result#, best_interval


    def detect_change(self, current_feature, history_features):
        # 初始阶段（历史帧不足时）视为无变化
        if len(history_features) < self.history_size:
            return False, None
        similarities = [np.dot(current_feature, f) for f in history_features]
        #print('detect_change:',all(s < self.threshold for s in similarities), similarities)
        return all(s < self.threshold for s in similarities), similarities
    
    def detect_image_ocr(self, image):
        ocr_text = self.ocr_model.run_paddle_ocr(image=image)
        if len(ocr_text) < 8000:
            return ocr_text
        else:
            return ocr_text[:8000]
    
    def detect_change_by_ov_ocr(self, filepath1, filepath2):
        start = time.time()
        ocr_text1 = self.ocr_model.run_paddle_ocr(image=filepath1)
        end = time.time()
        print('O1 time = ', end-start)
        ocr_text2 = self.ocr_model.run_paddle_ocr(image=filepath2)
        end1 = time.time()
        print('O2 time = ', end1-end)
        if len(ocr_text1) < 20 and len(ocr_text2) < 20:
            return False
        
        bert_sim = self.cosine_similarity(ocr_text2, ocr_text1)
        end2 = time.time()
        print('O3 time = ', end2-end1)
        if bert_sim < 0.9:
            # print("OCR相似度补缺: ", bert_sim)
            # print('文件1:', filepath1)
            # print('文件2:', filepath2)
            # print('ocr1=', ocr_text1)
            # print('ocr2=', ocr_text2)
            return True  # 如果 OCR 相似度小于 0.95，则认为是场景变化
        else:
            return False


    def detect_change_by_ocr(self, filepath1, filepath2):
        output = self.ocr_model.predict(
                input=filepath1,
            )
        for res in output:
            res_ = res._to_str()
            res = res_['res']

            # 提取识别的文本
            recognized_texts = res.get('overall_ocr_res', [])
            recognized_texts = recognized_texts.get('rec_texts',[])
            #print("Recognized Texts:")

            ocr_text1 = "\n".join(recognized_texts)  # 将文本拼接在一起，用换行分隔
            #print(" ".join(recognized_texts))   
        
        output = self.ocr_model.predict(
            input=filepath2,
        )
        for res in output:
            res_ = res._to_str()
            res = res_['res']

            # 提取识别的文本
            recognized_texts = res.get('overall_ocr_res', [])
            recognized_texts = recognized_texts.get('rec_texts',[])
            #print("Recognized Texts:")
            ocr_text2 = "\n".join(recognized_texts)  # 将文本拼接在一起，用换行分隔
            print(" ".join(recognized_texts))
        #print("------------jaccard_similarity: ", jaccard_similarity(ocr_text2, ocr_text1))
        if len(ocr_text1) < 20 and len(ocr_text2) < 20:
            return False
        bert_sim = self.cosine_similarity(ocr_text2, ocr_text1)
        if bert_sim < 0.9:
            # print("OCR相似度补缺: ", bert_sim)
            # print('文件1:', filepath1)
            # print('文件2:', filepath2)
            return True  # 如果 OCR 相似度小于 0.95，则认为是场景变化
        else:
            return False

    def _extract_feature(self, image):
        pil_image = Image.fromarray(image)
        image_input = self.clip_model.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feature = self.clip_model.model.encode_image(image_input)
            feature = feature / feature.norm(dim=1, keepdim=True)
        return feature.cpu().numpy().flatten()
    
    def bert_similarity(self, doc1, doc2):
        # embeddings_1 = self.bgem3_model.encode(doc1, 
        #                         batch_size=1, 
        #                         max_length=1024, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
        #                         )['dense_vecs']
        # embeddings_2 = self.bgem3_model.encode(doc2)['dense_vecs']
        embeddings = self.bgem3_model.infer([doc1, doc2])
        similarity = embeddings[0] @ embeddings[1].mT
        return similarity
    
    def cosine_similarity(self, doc1, doc2):
        start = time.time()
        embeddings = self.bgem3_model.infer([doc1, doc2])
        if isinstance(embeddings[0], list):
            embeddings = embeddings[0]
        else:
            embeddings = embeddings[0].tolist()
        #print('embedding time=', time.time()-start)
        text_vec = embeddings[0]
        img_vec = embeddings[1]
        #print(type(embeddings), type(text_vec), type(img_vec))
        #print(embeddings)
        try:
            start = time.time()
            text_array = np.array(text_vec)
            img_array = np.array(img_vec)
            dot_product = np.dot(text_array, img_array)
            norm_text = np.linalg.norm(text_array)
            norm_img = np.linalg.norm(img_array)
            #print('embedding calc time=', time.time()-start)
            ret = dot_product / (norm_text * norm_img) if (norm_text * norm_img) != 0 else 0
            #print('ret =', ret)
            return ret
        except Exception as e:
            print(f"*** BGE Embedding: Similarity calculation error: {e} ***")
            return 0


class CHINESECLIP():
    def __init__(self, key='', model_name='ViT-B-16', device='cpu', **kwargs):

        self.device = device
        self.clip_model_name = model_name #"openai/clip-vit-large-patch14" #CLIP image encoder
        # 加载 CLIP 模型
        self.model, self.preprocess = load_from_name(self.clip_model_name, device=self.device, download_root='./models')
        self.model.eval()
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.text_embeddings_size = 512
        self.image_embeddings_size = 512
        # print('CHINESECLIP:', 'Text dim:', self.text_embeddings_size, 'Image dim:', self.image_embeddings_size)

    
    # 计算文本嵌入向量的函数
    def encode_text(self, queryList):
        if not self.model:
            print('*** CHINESECLIP: Text embedding failed, model is null ***')
            return []

        # 将输入文本列表tokenize
        inputs = clip.tokenize(queryList).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        text_embeddings = []
        if not isinstance(text_features, list) or not isinstance(text_features[0], list):
            for text in text_features:
                if not isinstance(text, list):
                    text = text.tolist()
                    text_embeddings.append(text)
                else:
                    text_embeddings.append(text)
        
        return text_embeddings

    
    # # 计算图片嵌入向量的函数
    def encode_image(self, imageList):
        if not self.model or not self.preprocess:
            print('*** CHINESECLIP: Image embedding failed, model or processor is null ***')
            return []
        
        img_embeddings = []
        with torch.no_grad():
            for idx, img in enumerate(imageList):
                image = Image.open(img)
                #print(f'Image embedding {idx}', img)
                inputs = self.preprocess(image).unsqueeze(0).to(self.device)
                image_features = self.model.encode_image(inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True).detach()
                if not isinstance(image_features[0], list):
                    image_features = image_features[0].tolist()
                
                img_embeddings.append(image_features)
            
        return img_embeddings
    
    # 计算consine相似度
    def cosine_similarity(self, text_vec, img_vec):
        # 计算两个向量的点积
        dot_product = np.dot(text_vec, img_vec)
        
        # 计算向量的范数
        norm_vec1 = np.linalg.norm(text_vec)
        norm_vec2 = np.linalg.norm(img_vec)
        
        # 计算余弦相似度
        cosine_sim = dot_product / (norm_vec1 * norm_vec2)
        
        return cosine_sim



if __name__ == '__main__':
    model = ChangeDetector()
    query = '气质美'
    image_list = [r'C:\Users\SAS\Downloads\multi-modal-meeting-system\client_frames\48e0a312-6978-416d-b8eb-80bfa3cf7aa6\frame_1160_keyFrame.jpg',
                r'C:\Users\SAS\Downloads\multi-modal-meeting-system\client_frames\48e0a312-6978-416d-b8eb-80bfa3cf7aa6\frame_1161_keyFrame.jpg',
                r'C:\Users\SAS\Downloads\multi-modal-meeting-system\client_frames\48e0a312-6978-416d-b8eb-80bfa3cf7aa6\frame_1157_keyFrame.jpg'
                ]
    image = model.clip_model.encode_image(image_list)
    enc_list = model.clip_model.encode_text([query])
    for i in image:
        print('sim:', model.clip_model.cosine_similarity(enc_list[0],i))