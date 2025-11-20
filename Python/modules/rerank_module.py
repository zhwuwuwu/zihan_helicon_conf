import re, os
import requests, threading
from abc import ABC
import numpy as np
import torch

from tqdm import trange
from transformers import AutoTokenizer
import openvino as ov
from openvino import opset8 as opset
from openvino import Core, Model, Type, Shape, op
def new_const_1_dim(val):
    return op.Constant(Type.i32, Shape([1]), [val])
from modules.ocr_module import JSONConfigReader

class OpenVINORerankQwen3Rerank06B():
    _model = None
    _model_lock = threading.Lock()
    _device = 'GPU'
    config = JSONConfigReader("config.json")
    enable_rerank = config.get("app.enable_rerank", False)
    
    def __init__(self, key='', model_name='./models/rerank_model', model_type='i8'):
        # if not self.enable_rerank:
        #     return
        print(f"== Reranker model_name = {model_name}")
        #model_type = model_type
        if model_type not in ['f16', 'i8']:
            print(f"== Error: Can't support rerank_weight_type[{model_type}], avaible value['f16', 'i8']")
            exit()
        ov_model_xml = model_name+f"/openvino_model_{model_type}.xml"
        print(f"== ov_model_xml={ov_model_xml}")
        if not os.path.exists(ov_model_xml):
            print(f"== Error: No exist model fn: {ov_model_xml}")
            exit()
        
        self.__tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.__token_false_id = self.__tokenizer.convert_tokens_to_ids("no")
        self.__token_true_id = self.__tokenizer.convert_tokens_to_ids("yes")

        core = ov.Core()
        self.__ov_model = core.read_model(ov_model_xml)
        print(f"== OpenVINO device = Intel: {self._device}")

        self.__ov_model = self.__update_model(self.__ov_model, output_name='logits')
        print("== Enable update model.")
        # ov.save_model(self.__ov_model, "openvion_model_updated.xml")

        self.__ov_compiled_model = ov.compile_model(self.__ov_model, self._device)

        self.__max_length = 1024
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"        
        
        self.__prefix_tokens = self.__tokenizer.encode(prefix, add_special_tokens=False)
        self.__suffix_tokens = self.__tokenizer.encode(suffix, add_special_tokens=False)
        self.__task = 'Given a web search query, retrieve relevant passages that answer the query'

        print("== Loading Reranker model success\n")
    
    def __update_model(self, ov_model:ov.Model, output_name):
        org_output = ov_model.get_result()
        org_output_input = org_output.input(0).get_source_output()

        # Remove old output node.
        ov_model.remove_result(org_output)

        # Take old output node's input as new input.
        input_ids_shape = opset.shape_of(org_output_input)

        # Get middle value from [-1,-1,FeaDim], for example: get 2 from shape[1,2,3]
        last_seq_id = opset.gather(input_ids_shape, new_const_1_dim(1), new_const_1_dim(0))

        # idx start from 0, so need to reduce 1.
        indics = opset.add(last_seq_id, op.Constant(Type.i64, Shape([1]), [-1]))

        # Get last seq fea, same to input[:,-1,:]
        last_logits = opset.gather(org_output_input, indics, new_const_1_dim(1))

        # # Infer to get reshaped target shape.
        # # fea_dim = opset.gather(input_ids_shape, new_const_1_dim(2), new_const_1_dim(0))
        # # fea_dim= opset.add(fea_dim, op.Constant(Type.i64, Shape([1]), [-1]))
        # # neg_one = opset.constant([-1], dtype=Type.i64)
        # # target_shape = opset.concat([neg_one, fea_dim], axis=0)
        # target_shape = opset.constant([-1, 151668], dtype=ov.Type.i64)

        # # Reshape from [-1, 1, FeaDim] to [-1, FeaDim]
        # reshape = opset.reshape(last_logits, target_shape, special_zero=True)

        result_new = opset.result(last_logits, name=output_name)

        return ov.Model([result_new], [n.get_node() for n in ov_model.inputs], "model_updated")

    def __format_instruction(self, instruction, query, doc):
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
        return output

    def __process_inputs(self, pairs):        
        inputs = self.__tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=self.__max_length - len(self.__prefix_tokens) - len(self.__suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.__prefix_tokens + ele + self.__suffix_tokens
        # inputs = self.__tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.__max_length)
        inputs = self.__tokenizer.pad(inputs, return_tensors="pt", max_length=self.__max_length)
        # for key in inputs:
        #     inputs[key] = inputs[key].to(self.__model.device)
        return inputs

    def __compute_logits(self, inputs, **kwargs):
        input_ids = inputs['input_ids']
        attention_mask = inputs["attention_mask"]

        batch_size, sequence_length = input_ids.shape
        # Most common approach: Simple arange for each sequence, then reshape
        # This works well if you are right-padding and the model handles padding with attention_mask
        position_ids = np.arange(sequence_length, dtype=np.int64)[np.newaxis, :]
        # If batch size > 1, replicate for the batch
        position_ids = np.repeat(position_ids, batch_size, axis=0)

        outputs = self.__ov_compiled_model({
            "input_ids":input_ids, 
            "attention_mask":attention_mask,
            "position_ids":position_ids
            })
        
        logits = torch.from_numpy(outputs['logits'])
        batch_scores = logits[:, -1, :]

        true_vector = batch_scores[:, self.__token_true_id]
        false_vector = batch_scores[:, self.__token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def infer_rerank(self, desc_list:list[str], img_list:list[str]):
        print("== Fail. not support description and image rerank.")
        pass

    def infer_rerank_desc(self, desc_list1:list[str], desc_list2:list[str]):
        assert(len(desc_list1) == len(desc_list2))
        pairs = [self.__format_instruction(self.__task, query, doc) for query, doc in zip(desc_list1, desc_list2)]

        # Tokenize the input texts
        inputs = self.__process_inputs(pairs)
        # get_nv_gpu_memory_usage(prefix="After process inputs:")
        scores = self.__compute_logits(inputs)

        return scores

    def similarity_no_batch(self, query: str, texts: list):
        desc_list1 = [query] * len(texts)
        desc_list2 = texts

        scores = self.infer_rerank_desc(desc_list1, desc_list2)

        #token_count = sum(num_tokens_from_string(t) for t in texts)
        return np.array(scores)#, token_count

    def similarity(self, query: str, texts: list):
        #token_count = sum(num_tokens_from_string(t) for t in texts)
        res = []
        batch_size = 4
        num_batches = (len(texts) + batch_size - 1) // batch_size
        import time
        start_time = time.time()

        for batch_index in range(num_batches):
            start_index = batch_index * batch_size
            end_index = min((batch_index + 1) * batch_size, len(texts))
            batch_texts = texts[start_index:end_index]

            desc_list1 = [query] * len(batch_texts)
            desc_list2 = batch_texts
            batch_scores = self.infer_rerank_desc(desc_list1, desc_list2)

            batch_res = [{"id": start_index + i, "score": score}
                         for i, score in enumerate(batch_scores)]
            res.extend(batch_res)

        #print('\nTotal Rerank=', len(texts), '\nRerank time:', time.time() - start_time, 's\n')

        res.sort(key=lambda x: x["id"], reverse=False)
        ans = [float(doc["score"]) for i, doc in enumerate(res)]
        return np.array(ans)#, token_count
    

if __name__ == '__main__':
    rerank = OpenVINORerankQwen3Rerank06B(model_type='f16')
    result = rerank.similarity('一只小狗', ['一只猫', '一只猪', '一只狗'])
    index = np.argsort(result * -1)[0:10]
    print(result)
    print(index)