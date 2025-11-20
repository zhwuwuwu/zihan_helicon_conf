import os
import sys
CUR_FILE_PATH=os.path.dirname(os.path.abspath(__file__))
sys.path.append(CUR_FILE_PATH + "/../utils/")
sys.path.append(CUR_FILE_PATH + "/../")

import torch
import numpy as np
# from model.model_reranker_base import RerankModelBase
# from model.model_fea import ModelBase
# from model.model_minicpm_v26_emb import get_question
# from model.model_txt_emb import Model_TxtEmb
from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, TextStreamer
# from qwen_vl_utils import process_vision_info
# import time
# from model.video_to_imgs import video_to_fns

from utils.utils_model import get_model_full_path, get_logits_fea, TaskType, cache_desc, get_logits_fea_common
# from my_utils.utils_com import set_device

# OV
from optimum.intel.openvino import OVModelForVisualCausalLM
from PIL import Image 


class Model_Qwen2_VL_2B_OV():
    def __init__(self, language='zn'):
        self.__device = f"CPU"
        self.__model_id = ".\\models\\qwen2.5\\INT8"
        if language not in ['zn', 'en']:
            print("==Fail, Model_Qwen2_VL_2B_OV only support ['zn', 'en] at present.")
            exit()
        self.__language = language
        self.__is_en = language == 'en'
        # if task_type in [TaskType.Reranker, TaskType.FeaExtract]:
        #     print(f"== RerankModelBase::RerankModelBase: {self.__model_id}")
        # else:
        #     print(f"== Error: Don't support task type: {task_type}, Fail:{__file__}")
        #     exit()

        model_path='.\\models\\qwen2.5\\INT8'#get_model_full_path(self.__model_id)
        print(f"== '{self.__model_id}' path: {model_path}")
    
        self.__tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.__ov_model = OVModelForVisualCausalLM.from_pretrained(model_path, trust_remote_code=True)

        target_token_strings = [' Y', ' N']
        self.__token_yn_id = self.__tokenizer(target_token_strings)['input_ids']
        # print("token_yn_id=", self.__token_yn_id)

        # BG3-M3
        # if task_type == TaskType.FeaExtract:
        #     self.__cache_desc = cache_desc(self.__model_id + get_question(self.__language == 'en'), self.__language)
        #     self.__model_emb = Model_TxtEmb()

    def infer_rerank(self, desc_list:list[str], img_list:list[str]):
        messages_list = []
        sim=[]
        for idx, img_path in enumerate(img_list):
            if self.__language == 'zn':
                prompt_text = '''请回答以下问题，务必只能回复一个词 "Y"或 "N"：
                            图片和"'''+ desc_list[idx] +'''"是否相关？'''
            elif self.__language == 'en':
                prompt_text = '''Please answer the following question, and be sure to answer only one word "Y" or "N":
                    Are the pictures related to "'''+ desc_list[idx] +'''"? '''
            else:
                print(f"==Fail. don't support language: {self.__language}")
                exit()

            generation_args = { 
                "max_new_tokens": 1, 
                # "streamer": TextStreamer(self.__tokenizer, skip_prompt=True, skip_special_tokens=True),
                "return_dict_in_generate":True,
                "output_logits":True
            }
            # messages_list.append(messages)

            image = Image.open(img_path)
            inputs = self.__ov_model.preprocess_inputs(text=prompt_text, image=image, 
                                                       tokenizer=self.__tokenizer, 
                                                       config=self.__ov_model.config)
            outputs = self.__ov_model.generate(**inputs, **generation_args)
            logits = outputs['logits'][0]
            score = get_logits_fea_common(logits.numpy().tolist(), self.__token_yn_id)
            # generate_ids = outputs['sequences'][:, inputs['input_ids'].shape[1]:]
            # response = self.__tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
            # print("== response=", response)
            sim.append(score)

        return sim

    def infer_rerank_desc(self, desc_list1:list[str], desc_list2:list[str]):
        print("== Fail. not support description and description rerank.")
        pass

    def similarity(self, emb_q, emb_c):
        return emb_q@emb_c.T

    def infer_q(self, txt:list[str], img:list[str]=None):
        return self.__model_emb.infer(txt)

    def infer_c(self, imgs:list[str])->tuple[list, list[str]]:
        # batch and single, result is different. so just process batch=1
        return None, None
    
    def infer_c_video(self, video_fn):
        if self.__cache_desc.exist_fns([video_fn]):
            output_text = self.__cache_desc.load_des(imgs=[video_fn])
        else:
            img_fns = video_to_fns(video_fn)
            if img_fns is None:
                return None
            # print("img_fns=",img_fns)
            question = "Describe this video."
            question = '''本视频有多张时间连续的画面。 
            按照时间先后的顺序进行逐帧精细化分析，必须详细捕捉前后多帧间主角的状态及姿态变化细节，分析维度需涵盖：
            [主角分析]：聚焦主角形态演变，记录其从初始状态到最终状态的形状、大小、颜色等变化。
            完成所有帧分析后，以因果逻辑链提炼视频核心内容，总结人物姿态变化过程为叙事核心，着重阐述起身动作发生的原因、动作过程中的关键细节、该动作引发的后续事件，同时涵盖核心事件的发展脉络、关键人物其他动作动机、物品功能演变及场景转换对事件推进的作用。若画面存在无人物或物品的特殊帧，需重点分析环境要素变化对人物起身动作这一核心情节的铺垫作用。 
            必须要根据主角外貌特征，说明主角像什么动物种类。
            若画面里面有文字，则需要识别画面里面的文字。'''
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": img_fns,
                            "fps": 1.0,
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ]
            
            # Preparation for inference
            text = self.__processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.__processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.__device)

            # Inference
            generated_ids = self.__model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.__processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            # print(output_text)
            self.__cache_desc.save_des([video_fn], output_text)

        # embedding
        candi_embs = self.__model_emb.encode(output_text, 
                batch_size=12, 
                max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                )['dense_vecs']

        return torch.tensor(candi_embs).to(torch.float32), output_text

# def unit_test():
#     model = Model_Qwen_600M()
#     res = model.infer_rerank_desc(desc_list1=["图片中有个黄色的中华田园犬在奔跑", "图片中有个黄色的中华田园犬在玩球。"], desc_list2=["小狗在奔跑。","人在奔跑。"])
#     print(f"== Res: {res}")

def unit_test_rerank_qwen2_vl_2b(enable_ov=False):
    print("*** unit_test_rerank_qwen2_vl_2b *** enable_ov:", enable_ov)
    if not enable_ov:
        set_device(1)
    model = Model_Qwen2_VL_2B_OV() #if enable_ov else Model_Qwen2_VL_2B()
    img_root='/mnt/data_nvme1n1p1/xiping_workpath/ai_nas/ai_nas_verify/'
    inp_imgs = [img_root+"/test_data/facebook/my_data/images/cat_1.jpg"]*2
    inp_dess = ["a cat", "a dog"]
    similarity = model.infer_rerank(inp_dess, inp_imgs)

    # for id in range(20):
    #     t1= time.time()
    #     similarity = model.infer_rerank(inp_dess, inp_imgs)
    #     t2= time.time()
    #     print(f"== Infer [{id}] tm: {t2-t1:.3f} s")

    print(f"== similarity shape: {np.array(similarity).size}, type: {type(similarity)}, data type: {np.array(similarity).dtype}")

    print(f"== inputs:\n{inp_dess}\nVS\n{[os.path.split(fn)[-1] for fn in inp_imgs]}")
    print(f"== rerank similarity: {similarity}")
    print("== Done.")

# def unit_test_fea_extract_qwen2_vl_2b():
#     set_device(0)
#     model = Model_Qwen2_VL_2B(task_type=TaskType.FeaExtract, language='zn')
#     img_root='/mnt/data_nvme1n1p1/xiping_workpath/ai_nas/ai_nas_verify/'
#     inp_imgs = [img_root+"/test_data/facebook/my_data/images/cat_1.jpg"]
#     fea_c, desc = model.infer_c(inp_imgs)

#     for id in range(3):
#         t1= time.time()
#         emb, desc = model.infer_c(inp_imgs)
#         t2= time.time()
#         print(f"== Infer [{id}] tm: {t2-t1:.3f} s")
#         # print(f" 1: {desc[0]}")

#     print(f"== emb shape: {np.array(emb).size}, type: {type(emb)}, data type: {np.array(emb).dtype}")
#     print(f"== desc: {desc}")

#     fea_q = model.infer_q(["一只黄色的小猫在注视前方。","一只小狗在奔跑。"])
#     score = model.similarity(fea_q, fea_c)
#     print(f"== score: {score}")
#     print("== Done.")

# def unit_test_video_fea_extract_qwen2_vl_2b():
#     set_device(0)
#     model = Model_Qwen2_VL_2B(task_type=TaskType.FeaExtract, language='zn')
#     img_root='/mnt/data_nvme1n1p1/xiping_workpath/ai_nas/ai_nas_verify/'
#     inp_video = img_root+'test_data/video/video_199/video/1048982_x16.mp4'
#     inp_video = img_root+'test_data/video/video_199/video/e7ea0e7a4967629fc1cf65a7b4760686.mp4'
#     desc = model.infer_c_video(inp_video)

#     for id in range(3):
#         t1= time.time()
#         desc = model.infer_c_video(inp_video)
#         t2= time.time()
#         print(f"== Infer [{id}] tm: {t2-t1:.3f} s")
#         # print(f" 1: {desc[0]}")

#     print(f"== desc: {desc}")
#     print("== Done.")

if __name__ == "__main__":
    # unit_test()
    #unit_test_rerank_qwen2_vl_2b(enable_ov=False)
    unit_test_rerank_qwen2_vl_2b(enable_ov=True)
    # unit_test_fea_extract_qwen2_vl_2b()
    # video_to_fns('../test_data/video/video_199/video/1048982_x16.mp4')
    # unit_test_video_fea_extract_qwen2_vl_2b()