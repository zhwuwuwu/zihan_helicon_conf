import os
import sys
CUR_FILE_PATH=os.path.dirname(os.path.abspath(__file__))
sys.path.append(CUR_FILE_PATH + "/../utils/")
sys.path.append(CUR_FILE_PATH + "/../")

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
#from FlagEmbedding import BGEM3FlagModel
#from model.model_base_txt_emb import Model_BaseTxtEmb

# from transformers import (
#     AutoModel, AutoConfig,
#     AutoTokenizer, PreTrainedTokenizer
# )
from tqdm import tqdm, trange
import numpy as np

from modules.utils_model import get_model_full_path
from modules.utils_com import set_device, get_cur_device, load_json_file

import openvino as ov
from openvino import opset8 as opset
from openvino import Core, Model, Type, Shape, op


# def get_temp_directory():
#     if hasattr(sys, '_MEIPASS'):
#         # 如果程序是打包后运行的，返回解压后的临时目录
#         print('_MEIPASS=', sys._MEIPASS)
#         return sys._MEIPASS
#     else:
#         return None
# if get_temp_directory():
#     sys.path.append(get_temp_directory())


def new_const_1_dim(val):
    return op.Constant(Type.i32, Shape([1]), [val])

def get_vocab_zen():
    token_table_zn = load_json_file("./model_weights/BAAI/bge-base-zh-v1.5/tokenizer.json")
    token_table_en = load_json_file("./model_weights/BAAI/bge-large-en-v1.5/tokenizer.json")
    vocab_zn = list(token_table_zn["model"]["vocab"].keys())
    vocab_en = list(token_table_en["model"]["vocab"].keys())
    vocab_zen = list(set(vocab_en).union(set(vocab_zn)))
    # print(f"vocab_zn size: {len(vocab_zn)}")
    # print(f"vocab_en size: {len(vocab_en)}")
    # print(f"vocab_zen size: {len(vocab_zen)}")
    return vocab_zen


class Model_BGE_M3():
    def __init__(self, language='zn', use_ov=False, export_ov=False):
        self.__model_id_emb = 'BAAI/bge-m3'
        #self.__device = get_cur_device()
        self.__device = 'CPU'
        model_path='./models/bge-m3'#get_model_full_path(self.__model_id_emb)
        print(f"== '{self.__model_id_emb}' path: {model_path}")
        print(f"== current device: {self.__device}")

        self.__ov_export_output_dir=model_path
        self.__use_ov=use_ov
        self.__export_ov=export_ov
        if use_ov and export_ov:
            print("== Warning. Can't set use_ov and export_ov to True at the same time. Just keep export_ov=True, use_ov=False.")
            self.__use_ov = False

        self.__use_native_api = True # call AutoModel and AutoTokenizer.
        # self.__use_native_api = False
        use_fp16=True
        if self.__use_native_api:
            self.__pooling_method: str = "cls"
            self.__normalize_embeddings = True

            # OpenVINO
            if self.__use_ov:
                print("== Enable OpenVINO to inference, self.__device changed to 'cpu'.")
                self.__device =  self.__device
                core = ov.Core()
                # core.set_property(properties={ov.properties.enable_mmap: True})
                self.__ov_model = core.read_model(self.__ov_export_output_dir+"/bge_m3_int8.xml")
                # self.__ov_model = self.__update_ov_model(self.__ov_model, "last_hidden_state")
                # ov.save_model(self.__ov_model, "openvino_model_updated_i8.xml")
                self.__ov_compiled_model = ov.compile_model(self.__ov_model,  self.__device)

                self.__compiled_tokenizer = core.compile_model(self.__ov_export_output_dir+"/openvino_tokenizer.xml",  self.__device) # Or "GPU", "NPU"
                self.__tokenizer_infer_request = self.__compiled_tokenizer.create_infer_request()

            # else:
            #     self.__tokenizer = AutoTokenizer.from_pretrained(
            #         model_path,
            #         trust_remote_code=True,
            #         cache_dir=None
            #     )
   
            #     self.__model = AutoModel.from_pretrained(
            #         model_path,
            #         cache_dir=None,
            #         trust_remote_code=True,
            #         low_cpu_mem_usage = True
            #     )
            #     self.vocab_size = self.__model.config.vocab_size

            # 另外2中提取特征的方式
            # colbert_dim = -1
            # self.__colbert_linear = torch.nn.Linear(
            #     in_features=self.__model.config.hidden_size,
            #     out_features=self.__model.config.hidden_size if colbert_dim <= 0 else colbert_dim
            # )
            # self.__sparse_linear = torch.nn.Linear(
            #     in_features=self.__model.config.hidden_size,
            #     out_features=1
            # )
            # colbert_model_path = os.path.join(model_path, 'colbert_linear.pt')
            # sparse_model_path = os.path.join(model_path, 'sparse_linear.pt')
            # if os.path.exists(colbert_model_path) and os.path.exists(sparse_model_path):
            #     print('== loading existing colbert_linear and sparse_linear---------')
            #     colbert_state_dict = torch.load(colbert_model_path, map_location='cpu', weights_only=True)
            #     sparse_state_dict = torch.load(sparse_model_path, map_location='cpu', weights_only=True)
            #     self.__colbert_linear.load_state_dict(colbert_state_dict)
            #     self.__sparse_linear.load_state_dict(sparse_state_dict)

            # if not self.__use_ov:
            #     # self.__model.embeddings.word_embeddings.weight
            #     quantization=False
            #     if quantization:
            #         # 量化后，无法推理，提示错误：（是pytorch本身的bug）
            #         # AssertionError: Embedding quantization is only supported with float_qparams_weight_only_qconfig
            #         for _, mod in self.__model.named_modules():
            #             if isinstance(mod, torch.nn.Embedding):
            #                 mod.qconfig = torch.ao.quantization.float_qparams_weight_only_qconfig
            #         self.__model = torch.quantization.quantize_dynamic(
            #             self.__model,
            #             # 指定要量化的层类型。nn.Embedding 应该在这里。
            #             # nn.EmbeddingBag 是 nn.Embedding 在量化后的常用实现，也需要指定。
            #             {nn.Embedding, nn.EmbeddingBag},
            #             dtype=torch.qint8
            #         )

            #     if use_fp16 and self.__device != 'cpu': 
            #         self.__model.half()
            #     self.__model.to(self.__device)
            #     self.__model.eval()
        # else:
        #     # torch_dtype=torch.float16
        #     # low_cpu_mem_usage=True
        #     # torch_load_kwargs={"mmap": True, "weights_only": True}
        #     self.__model_emb = BGEM3FlagModel(model_path, use_fp16=True, devices=self.__device) # mmap = True

    def __update_ov_model(self, ov_model:ov.Model, output_name):
        print(f"== Update last layer for ov model. new output_name={output_name}")
        org_output = ov_model.get_results()[0]
        org_output_input = org_output.input(0).get_source_output()

        # Remove old output node.
        ov_model.remove_result(org_output)

        # idx start from 0, so need to reduce 1.
        indics = new_const_1_dim(0)

        # Get last seq fea, same to input[:,0,:]
        last_logits = opset.gather(org_output_input, indics, new_const_1_dim(1))

        result_new = opset.result(last_logits, name=output_name)

        return ov.Model([result_new], [n.get_node() for n in ov_model.inputs], "model_updated")

    def infer(self, txt:list[str]):
        if self.__use_native_api:
            all_dense_embeddings, all_lexical_weights, all_colbert_vecs = [], [], []
            if self.__use_ov:
                length_sorted_idx = list(range(len(txt))) # OV: not sort.
                input_tensor = ov.Tensor(txt)
                self.__tokenizer_infer_request.set_input_tensor(input_tensor)
                self.__tokenizer_infer_request.infer()

                input_ids = self.__tokenizer_infer_request.get_tensor("input_ids").data
                attention_mask = self.__tokenizer_infer_request.get_tensor("attention_mask").data

                last_hidden_state = self.__ov_compiled_model({
                    "input_ids":input_ids, 
                    "attention_mask":attention_mask
                    })
                last_hidden_state = torch.from_numpy(last_hidden_state['last_hidden_state'])
            # else:
            #     batch_size=12
            #     max_length=8192

            #     all_inputs = []
            #     sentences = txt
            #     for start_index in trange(0, len(sentences), batch_size, desc='pre tokenize',
            #                             disable=len(sentences) < 256):
            #         sentences_batch = sentences[start_index:start_index + batch_size]
            #         inputs_batch = self.__tokenizer(
            #             sentences_batch,
            #             truncation=True,
            #             max_length=max_length
            #         )
            #         inputs_batch = [{
            #             k: inputs_batch[k][i] for k in inputs_batch.keys()
            #         } for i in range(len(sentences_batch))]
            #         all_inputs.extend(inputs_batch)

            #     # sort by length for less padding
            #     length_sorted_idx = np.argsort([-len(x['input_ids']) for x in all_inputs])
            #     all_inputs_sorted = [all_inputs[i] for i in length_sorted_idx]

            #     # encode
            #     for start_index in tqdm(range(0, len(sentences), batch_size), desc="Inference Embeddings",
            #                             disable=len(sentences) < 256):
            #         inputs_batch = all_inputs_sorted[start_index:start_index + batch_size]
            #         inputs_batch = self.__tokenizer.pad(
            #             inputs_batch,
            #             padding=True,
            #             return_tensors='pt'
            #         ).to(self.__device)
            #         last_hidden_state = self.__model(**inputs_batch, return_dict=True).last_hidden_state

            #         if self.__export_ov:
            #             print("== Start to export tokenizer to OV.")
            #             ov_tokenizer_model, ov_detokenizer_model = convert_tokenizer(self.__tokenizer, with_detokenizer=True)
            #             print("== Start to export model to OV.")
            #             ov_model = ov.convert_model(
            #                 self.__model,
            #                 # input=sentenses
            #                 example_input={
            #                     "input_ids": inputs_batch['input_ids'],
            #                     "attention_mask": inputs_batch['attention_mask'],
            #                 }
            #             )

            #             print("== Start to compress OV weights.")
            #             # compress_weights is memory inplace.
            #             ov_model_4bit = compress_weights(copy.deepcopy(ov_model), mode=CompressWeightsMode.INT4_SYM)
            #             ov_model_8bit = compress_weights(copy.deepcopy(ov_model_4bit), mode=CompressWeightsMode.INT8)

            #             print("== Start to save OV model and tokenizer")
            #             if not os.path.exists(self.__ov_export_output_dir): os.mkdir(self.__ov_export_output_dir)
            #             ov.save_model(ov_model, self.__ov_export_output_dir+"/bge_m3.xml")
            #             ov.save_model(ov_model_8bit, self.__ov_export_output_dir+"/bge_m3_int8.xml")
            #             ov.save_model(ov_model_4bit, self.__ov_export_output_dir+"/bge_m3_int4_sym.xml")

            #             tokenizer_xml_path = self.__ov_export_output_dir+"/openvino_tokenizer.xml"
            #             detokenizer_xml_path = self.__ov_export_output_dir+"/openvino_detokenizer.xml"
            #             ov.save_model(ov_tokenizer_model, tokenizer_xml_path)
            #             ov.save_model(ov_detokenizer_model, detokenizer_xml_path)

            # ======== Post processing ========
            # _dense_embedding
            if self.__pooling_method == "cls":
                dense_vecs = last_hidden_state[:, 0]
            if self.__normalize_embeddings:
                # n=x/sqrt(x1*x1+x2*x2+...)
                dense_vecs = F.normalize(dense_vecs, dim=-1)

            all_dense_embeddings.append(dense_vecs.cpu().detach().numpy())
            all_dense_embeddings = np.concatenate(all_dense_embeddings, axis=0)
            # adjust the order of embeddings
            all_dense_embeddings = all_dense_embeddings[np.argsort(length_sorted_idx)]
            return torch.tensor(all_dense_embeddings).cpu().to(torch.float32)
        # else:
        #     candi_embs = self.__model_emb.encode(txt, 
        #                                         batch_size=12, 
        #                                         max_length=8192,
        #                                         )['dense_vecs']
        #     return torch.tensor(candi_embs).to(torch.float32)

#from openvino_tokenizers import convert_tokenizer
#from nncf import compress_weights, CompressWeightsMode
def cvt_bgem3_to_ov():
    model = Model_BGE_M3(export_ov=True)
    embs1 = model.infer(["图片中有个黄色的中华田园犬在奔跑", "图片中有个黄色的中华田园犬在玩球。"])

# 量化成功，但是无法在原来模型使用。
def quantization_embedding():
    model_path='./model_weights/BAAI/bge-m3/'
    model_emb = BGEM3FlagModel(model_path, use_fp16=False, devices='cpu')
    model_emb.model.eval()
    model = model_emb.model
    # model_emb.tokenizer

    original_model_path = "embedding_model_original.pth"
    torch.save(model, original_model_path) # state_dict()
    original_size = os.path.getsize(original_model_path) / (1024 * 1024) # MB
    print(f"原始模型大小: {original_size:.2f} MB")
    
    for _, mod in model.named_modules():
        if isinstance(mod, torch.nn.Embedding):
            mod.qconfig = torch.ao.quantization.float_qparams_weight_only_qconfig

    quantized_model_dynamic = torch.quantization.quantize_dynamic(
        model,
        # 指定要量化的层类型。nn.Embedding 应该在这里。
        # nn.EmbeddingBag 是 nn.Embedding 在量化后的常用实现，也需要指定。
        {nn.Embedding, nn.EmbeddingBag},
        dtype=torch.qint8
    )

    # 保存量化后的模型
    quantized_dynamic_model_path = "embedding_model_quantized_dynamic.pth"
    torch.save(quantized_model_dynamic, quantized_dynamic_model_path)
    quantized_dynamic_size = os.path.getsize(quantized_dynamic_model_path) / (1024 * 1024) # MB
    print(f"动态量化后的模型大小: {quantized_dynamic_size:.2f} MB")
    print(f"大小压缩比: {original_size / quantized_dynamic_size:.2f}x")

def unit_test(use_ov=False):
    import time
    gpu_id=0
    #print_gpu_memory_usage(gpu_id)
    #rss, vms = get_current_process_memory_usage()
    #print(f"== Init RSS = {rss/(1024*1024):.2f} MB, VMS = {vms/1024/1024:.2f} MB.")
    #set_device(gpu_id)
    model = Model_BGE_M3(use_ov=use_ov)
    for i in range(20):
        t1 = time.time()
        embs1 = model.infer(["oaigqnan;f爱国噶溶解傲娇发发olds发生发觉啊艾佛i瑟夫会额外i怀柔区骄傲呢adsigajfeoaofaofaig图片中有个黄色的中华田园犬在奔跑", "图片中有个黄色的中华田园犬在玩球。"])
        t2 = time.time()
        print(f"  == {i} tm = {(t2-t1)*1000:.3f} ms")

    embs2 = model.infer(["小狗在奔跑。","人在奔跑。"])
    #rss, vms = get_current_process_memory_usage()

    print(f"== embs: {embs1[0].shape}")
    print(f"== embs1 vs embs2: {embs1@embs2.T}")

    #print(f"== RSS = {rss/(1024*1024):.2f} MB, VMS = {vms/1024/1024:.2f} MB.")
    #print_gpu_memory_usage(gpu_id)

from langchain_community.embeddings import OpenVINOBgeEmbeddings
import threading
from modules.ocr_module import JSONConfigReader
class OpenVINOEmbedding():
    _model = None
    _model_lock = threading.Lock()
    config = JSONConfigReader("config.json")
    _device = config.get("app.m3_device")
    _device = _device if _device else "CPU"
    def __init__(self, key="", model_name='./models/bge-m3-int8', **kwargs):
        """
        If you have trouble downloading HuggingFace models, -_^ this might help!!

        For Linux:
        export HF_ENDPOINT=https://hf-mirror.com

        For Windows:
        Good luck
        ^_-

        """
        print("calling into OpenVINO embeding model\n", 'M3 device=', self._device)
        if not OpenVINOEmbedding._model:
            with OpenVINOEmbedding._model_lock:
                import torch
                if not OpenVINOEmbedding._model:
                    try:
                        model_kwargs = {"device": self._device}
                        encode_kwargs = {"normalize_embeddings": True}
                        print('load ebd from', model_name)
                        OpenVINOEmbedding._model = OpenVINOBgeEmbeddings(
                            model_name_or_path=model_name,
                            model_kwargs=model_kwargs,
                            encode_kwargs=encode_kwargs,
                        )
                        print("calling into OpenVINO embeding model 2\n",OpenVINOEmbedding._model)
                    except Exception as e:
                        print("calling into OpenVINO embeding model 3 - Exception\n", e)
                        return 
        print("create OpenVINO embedding model success\n")
        self._model = OpenVINOEmbedding._model


    def encode(self, texts: list, batch_size=32):
        arr = []
        tks_num = 0
        
        for txt in texts:
            try:
                res = self._model.embed_query(txt)
                arr.append(res)
                tks_num += len(res)
            except Exception as e:
                print("embed_query crash error:",str(e))
        return np.array(arr), tks_num

    def encode_queries(self, text):
        res = self._model.embed_query(text)
        return np.array(res)[0]#, len(res)
    
    def infer(self, texts):
        return self.encode(texts=texts)

import ctypes
from ctypes import c_void_p, c_uint32, c_int32, c_char_p, POINTER, c_float, c_int
class OpenVINOEmbedding_ov:
    handle = None
    _model_handle = None
    _model_lock = threading.Lock()
    config = JSONConfigReader("config.json")
    _device = config.get("app.m3_device")
    _device = _device if _device else "CPU"
    #_device = "GPU" # INFERENCE_DEVICES['embedding']['Openvino']

    def __init__(self, so_path='./models/bge-m3/bgem3ov.dll', weights="./models/bge-m3", compress_type=2):
        """
        :param so_path: libbgem3ov.so
        :param weights: embedding model
        :param device: ['CPU', 'GPU']
        :param compress_type: # 1=f16; 2=int8; 3=int4
        """
        #so_path = so_path or DEFAULT_SO_PATHS[EMBED_SERVICE_CONFIG["set_for_package"]]
        self.lib = ctypes.CDLL(so_path)
        self.lib.createModel_BGE_M3_OV.restype = c_void_p
        self.lib.createModel_BGE_M3_OV.argtypes = [c_char_p, c_char_p, c_int32]
        self.lib.releaseModel_BGE_M3_OV.restype = None
        self.lib.releaseModel_BGE_M3_OV.argtypes = [c_void_p]

        self.lib.inference_BGE_M3_OV.restype = None
        self.lib.inference_BGE_M3_OV.argtypes = [c_void_p, POINTER(c_char_p), c_uint32,
                                                 ctypes.POINTER(ctypes.POINTER(ctypes.c_float))]
        self.lib.getFeaDim_BGE_M3_OV.restype = c_int32
        self.lib.getFeaDim_BGE_M3_OV.argtypes = [c_void_p]

        print("loading embedding model=", so_path)
        print(f"embedding _device: {self._device}")
        try:
            # init/create
            if not self._model_handle:
                with self._model_lock:
                    if not self._model_handle:
                        self._model_handle = self.__create(weights, self._device, compress_type)

            self.handle = self._model_handle
            self.__fea_dim = self.lib.getFeaDim_BGE_M3_OV(self.handle)
        except BaseException as e:
            print("create bge-m3 failed:", e)

    def __create(self, model_path, device, compress_type):
        return self.lib.createModel_BGE_M3_OV(c_char_p(model_path.encode('utf-8')), c_char_p(device.encode('utf-8')),
                                              c_int32(compress_type), )

    # Destructors
    def __del__(self):
        if self.handle:
            self.lib.releaseModel_BGE_M3_OV(self.handle)

    def release(self):
        """释放模型资源"""
        if self.handle:
            self.lib.releaseModel_BGE_M3_OV(self.handle)
            self._model_handle = None

    def convert_c_2d_float_to_np_array(self, c_2d_ptr, rows, cols):
        if not c_2d_ptr:
            return []

        python_2d_list = []
        for r in range(rows):
            # c_2d_ptr[r] gives you a POINTER(c_float) to the start of the current row
            c_row_ptr = c_2d_ptr[r]

            row_list = []
            for c in range(cols):
                # c_row_ptr[c] directly accesses the element at index c in that row
                row_list.append(c_row_ptr[c])
            python_2d_list.append(row_list)

        return np.array(python_2d_list)

    def inference(self, txt_list: list[str]):
        input_txt_array = (ctypes.c_char_p * len(txt_list))(*[txt.encode('utf-8') for txt in txt_list])

        c_2d_array_of_ptrs = (ctypes.POINTER(ctypes.c_float) * len(txt_list))()
        for r_idx in range(len(txt_list)):
            fea_array = (ctypes.c_float * self.__fea_dim)()
            c_2d_array_of_ptrs[r_idx] = ctypes.cast(fea_array, ctypes.POINTER(ctypes.c_float))

        # Inference.
        self.lib.inference_BGE_M3_OV(self.handle, input_txt_array, len(txt_list), c_2d_array_of_ptrs)

        out_feas = self.convert_c_2d_float_to_np_array(c_2d_array_of_ptrs, len(txt_list), self.__fea_dim)
        # print("\n===== DEBUG OUTPUT ======")
        # print(f"Output type: {type(out_feas)}")
        # print(f"Output shape: {out_feas.shape if hasattr(out_feas, 'shape') else 'N/A'}")
        # print(f"Output dtype: {out_feas.dtype if hasattr(out_feas, 'dtype') else 'N/A'}")
        # print(f"Sample first row: {out_feas[0] if len(out_feas) > 0 else 'Empty'}")
        # print(f"len(out_feas): {len(out_feas)}")
        # print(f"out_feas.shape[0]: {out_feas.shape[0]}")
        # print(f"out_feas.shape[1]: {out_feas.shape[1]}")
        # print("========================\n")
        return out_feas

    def encode(self, texts: list, batch_size=32):
        arr = []
        tks_num = 0

        for txt in texts:
            try:
                res = self.inference([txt])
                arr.append(res[0])
                tks_num += res.shape[1]
            except Exception as e:
                print("embed_query crash error:", str(e))
        return [arr]

    def encode_queries(self, text):
        res = self.inference([text])
        return res[0], res.shape[1]

    def infer(self, texts):
        return self.encode(texts=texts)


import copy
if __name__ == "__main__":
    # quantization_embedding()
    # cvt_bgem3_to_ov()
    # unit_test(use_ov=False)
    #unit_test(use_ov=True)

    # langchain
    # embed = OpenVINOEmbedding(model_name='./models/bge-m3-int8')  #BAAI/bge-m3
    # ret = embed.encode(['一只小狗', '一直小猫'])
    # print('ret=',len(ret[0]))

    # ov
    embed = OpenVINOEmbedding_ov()  #BAAI/bge-m3
    ret = embed.encode(['一只小狗', '一直小猫'])
    print('ret=',len(ret[0]))