import os
import sys
CUR_FILE_PATH=os.path.dirname(os.path.abspath(__file__))
sys.path.append(CUR_FILE_PATH + "/../utils/")
sys.path.append(CUR_FILE_PATH + "/../")
from utils_model import get_model_full_path

from simple_tokenizer import SimpleTokenizer as _Tokenizer
from viclip import ViCLIP
import torch
import numpy as np
import cv2

export_ov = False
# export_ov = True

clip_candidates = {'viclip':None, 'clip':None}

def get_clip(name='viclip'):
    global clip_candidates
    m = clip_candidates[name]
    if m is None:
        if name == 'viclip':
            tokenizer = _Tokenizer()
            vclip = ViCLIP(tokenizer)
            m = (vclip, tokenizer)
        else:
            raise Exception('the target clip model is not found.')
    
    return m

def get_text_feat_dict(texts, clip, tokenizer, text_feat_d={}):
    for t in texts:
        feat = clip.get_text_features(t, tokenizer, text_feat_d)
        text_feat_d[t] = feat
    return text_feat_d

def get_vid_feat(frames, clip):
    return clip.get_vid_features(frames)

def _frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break

v_mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
def normalize(data):
    return (data/255.0-v_mean)/v_std

def frames2tensor(vid_list, fnum=8, target_size=(224, 224), device=torch.device('cuda')):
    assert(len(vid_list) >= fnum)
    step = len(vid_list) // fnum
    vid_list = vid_list[::step][:fnum]
    vid_list = [cv2.resize(x[:,:,::-1], target_size) for x in vid_list]
    vid_tube = [np.expand_dims(normalize(x), axis=(0, 1)) for x in vid_list]
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
    return vid_tube

def retrieve_text(frames, texts, name='viclip', topk=5, device=torch.device('cuda')):
    clip, tokenizer = get_clip(name)
    clip = clip.to(device)
    frames_tensor = frames2tensor(frames, device=device)
    vid_feat = get_vid_feat(frames_tensor, clip)

    text_feat_d = {}
    text_feat_d = get_text_feat_dict(texts, clip, tokenizer, text_feat_d)
    text_feats = [text_feat_d[t] for t in texts]
    text_feats_tensor = torch.cat(text_feats, 0)
    
    probs, idxs = clip.get_predict_label(vid_feat, text_feats_tensor, top=topk)

    ret_texts = [texts[i] for i in idxs.numpy()[0].tolist()]
    return ret_texts, probs.numpy()[0]

class MyViClip():
    def __init__(self, model=None, language='zn', device='cuda.0'):
        self.__tokenizer = _Tokenizer()
        self.__clip = ViCLIP(self.__tokenizer)
        self.__clip = self.__clip.to(device)
        pass

    def infer_frames(self, frames):
        return self.__clip.get_vid_features(frames)
    
    def export_ov_infer_frames(self, frames):
        print(f"== Start to export openvino video model.")
        import openvino as ov
        from nncf import compress_weights, CompressWeightsMode
        from torch.export import Dim

        # self.__clip.forward = self.__clip.vision_encoder
        model = self.__clip.vision_encoder
        model.to('cpu')
        model.eval()

        input = frames.permute(0, 2, 1, 3, 4).contiguous().to("cpu")
    
        # Specify that the first dimension of each input is that batch size
        batch = Dim("batch")
        t_seq = Dim("t_seq")
        hight = Dim("hight")
        width = Dim("width")
        dynamic_shapes = {"x": {0: batch}}
        exported_model = torch.export.export(model, (input,))
        print(f"== torch.export.export done. ------------>")
        ov_model = ov.convert_model(exported_model, example_input={
            "x":input,
            "masking_prob":0.0}
            )

        output_dir='model_weights/OpenGVLab/ViCLIP/export_ov/vision_encoder/'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        outp_fn = output_dir + "openvino_model.xml"
        ov.save_model(ov_model, outp_fn)
        print(f"== Convert vision_encoder done. save to {output_dir}")
        
        import copy
        ov_model_int8 = compress_weights(copy.deepcopy(ov_model), mode=CompressWeightsMode.INT8)
        outp_fn = output_dir + "openvino_model_int8.xml"
        ov.save_model(ov_model_int8, outp_fn)
        print(f"== Convert vision_encoder done. save to {output_dir}")

        ov_model_int4 = compress_weights(copy.deepcopy(ov_model), mode=CompressWeightsMode.INT4_ASYM)
        outp_fn = output_dir + "openvino_model_int4_asym.xml"
        ov.save_model(ov_model_int4, outp_fn)
        print(f"== Convert vision_encoder done. save to {output_dir}")

    def infer_txts(self, texts:list[str]):
        text_feat = []
        for t in texts:
            feat = self.__clip.get_text_features_2(t)
            text_feat.append(feat)
        return text_feat

    def export_ov_infer_txts(self, text):
        print(f"== Start to export openvino video model.")
        import openvino as ov
        from torch.export import Dim
        from nncf import compress_weights, CompressWeightsMode

        model = self.__clip.text_encoder
        model.to('cpu')
        model.eval()

        print(f"== Start to convert text tokenizer model.")

        print(f"== Start to convert text_encoder model.")
        dummy_input = torch.randint(low=0, high=10000, size=(1, 32), dtype=torch.int32)  # Random token IDs
        ov_model = ov.convert_model(model, example_input={
            "text": dummy_input
            })

        output_dir='model_weights/OpenGVLab/ViCLIP/export_ov/text_encoder/'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        outp_fn = output_dir + "openvino_model.xml"
        ov.save_model(ov_model, outp_fn)
        print(f"== Convert text model done. save to {outp_fn}")

        import copy
        ov_model_int8 = compress_weights(copy.deepcopy(ov_model), mode=CompressWeightsMode.INT8)
        outp_fn = output_dir + "openvino_model_int8.xml"
        ov.save_model(ov_model_int8, outp_fn)
        print(f"== Compress to int8 and save to {output_dir}")

        ov_model_int4 = compress_weights(copy.deepcopy(ov_model), mode=CompressWeightsMode.INT4_ASYM)
        outp_fn = output_dir + "openvino_model_int4_asym.xml"
        ov.save_model(ov_model_int4, outp_fn)
        print(f"== Compress to int4 and save to {output_dir}")
    
    def similarity(self, vid_feat, txt_fea, top=5):
        label_probs = (100.0 * vid_feat @ txt_fea.T).softmax(dim=-1)
        top_probs, top_labels = label_probs.cpu().topk(top, dim=-1)
        return top_probs, top_labels

import openvino as ov
from utils_com import check_intel_gpu_openvino
class ViClipOV():
    def __init__(self, model=None, language='zn', device=None):
        if device is None:
            self.__device = "GPU" if check_intel_gpu_openvino() else "CPU"
        else:
            self.__device = device

        model_path = "./"
        # depends on origianl model's tokenizer. only need this file[bpe_simple_vocab_16e6.txt.gz].
        self.__tokenizer = _Tokenizer(bpe_path=model_path+"bpe_simple_vocab_16e6.txt.gz")
        self.__max_txt_l = 32

        ov_model_path = model_path + "export_ov/"
        vision_encoder_path = ov_model_path + "vision_encoder/openvino_model.xml"
        vision_encoder_path = ov_model_path + "vision_encoder/openvino_model_int8.xml"
        # vision_encoder_path = ov_model_path + "vision_encoder/openvino_model_int4_asym.xml"

        text_encoder_path = ov_model_path + "text_encoder/openvino_model.xml"
        text_encoder_path = ov_model_path + "text_encoder/openvino_model_int8.xml"
        # text_encoder_path = ov_model_path + "text_encoder/openvino_model_int4_asym.xml"

        core = ov.Core()
        ov_model = core.read_model(vision_encoder_path)
        self.__ov_vision_encoder = ov.compile_model(ov_model, device)
        
        ov_model2 = core.read_model(text_encoder_path)
        self.__ov_text_encoder = ov.compile_model(ov_model2, device)
        self.__ov_text_encoder_irq = self.__ov_text_encoder.create_infer_request()

    def __frames2tensor(self, vid_list, fnum=8, target_size=(224, 224)):
        assert(len(vid_list) >= fnum)
        step = len(vid_list) // fnum
        vid_list = vid_list[::step][:fnum]
        vid_list = [cv2.resize(x[:,:,::-1], target_size) for x in vid_list]
        vid_tube = [np.expand_dims(normalize(x), axis=(0, 1)) for x in vid_list]
        vid_tube = np.concatenate(vid_tube, axis=1).astype(np.float32)
        vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
        vid_tube = np.transpose(vid_tube, (0, 2, 1, 3, 4))
        # vid_tube = vid_tube.permute(0, 2, 1, 3, 4).contiguous()
        vid_tube = ov.Tensor(vid_tube)
        # vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
        return vid_tube

    def infer_frames(self, frames):
        frame_tensors = self.__frames2tensor(frames)
        output = self.__ov_vision_encoder(frame_tensors)
        clip_feat = torch.from_numpy(output['mm:0'])
        clip_feat /= clip_feat.norm(dim=-1, keepdim=True) 
        return clip_feat.tolist()[0]

    def __tokenize(self, texts, context_length=77, truncate=True):
        if isinstance(texts, str):
            texts = [texts]
        
        from pkg_resources import packaging
        sot_token = self.__tokenizer.encoder["<|startoftext|>"]
        eot_token = self.__tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self.__tokenizer.encode(text) + [eot_token] for text in texts]
        if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
            result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        else:
            result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate:
                    tokens = tokens[:context_length]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result
    
    def infer_txts(self, texts:list[str]):
        text_feat = []
        for t in texts:
            txt_token = self.__tokenize(t, context_length=self.__max_txt_l)
            input = ov.Tensor(txt_token.cpu().numpy())
            output = self.__ov_text_encoder_irq.infer(input)
            
            text_features = torch.from_numpy(output[0])
            text_features /= text_features.norm(dim=-1, keepdim=True)  
            text_feat.append(text_features.tolist()[0])
        return text_feat
    
    def similarity(self, vid_feat, txt_fea, top=5):
        # return (100.0 * vid_feat @ txt_fea.T).softmax(dim=-1)
        label_probs = (100.0 * vid_feat @ txt_fea.T).softmax(dim=-1)
        top_probs, top_labels = label_probs.cpu().topk(top, dim=-1)
        return top_probs, top_labels
    
    def cosine_similarity(self, text_vec, img_vec):
        """
        计算余弦相似度（保持与原接口一致）

        参数:
            text_vec: 文本向量（列表）
            img_vec: 图像向量（列表）

        返回:
            余弦相似度值
        """
        try:
            text_array = np.array(text_vec)
            img_array = np.array(img_vec)
            dot_product = np.dot(text_array, img_array)
            norm_text = np.linalg.norm(text_array)
            norm_img = np.linalg.norm(img_array)
            return dot_product / (norm_text * norm_img) if (norm_text * norm_img) != 0 else 0
        except Exception as e:
            print(f"*** viclip: Similarity calculation error: {e} ***")
            return 0

def unit_test(frames, text_candidates):
    if 0: # original pytorch model.
        texts, probs = retrieve_text(frames, text_candidates, name='viclip', topk=5)
        for t, p in zip(texts, probs):
            print(f'text: {t} ~ prob: {p:.4f}')    
        # text: A man in a gray sweater plays fetch with his dog in the snowy yard, throwing a toy and watching it run. ~ prob: 0.8264
        # text: A playful dog and its owner wrestle in the snowy yard, chasing each other with joyous abandon. ~ prob: 0.1587
        # text: A pet dog excitedly runs through the snowy yard, chasing a toy thrown by its owner. ~ prob: 0.0141
        # text: A person dressed in a blue jacket shovels the snow-covered pavement outside their house. ~ prob: 0.0006
        # text: A playful dog slides down a snowy hill, wagging its tail with delight. ~ prob: 0.0002
    
    device='cpu' if export_ov else 'cuda'
    viclip = MyViClip(device=device)

    frame_tensors = frames2tensor(frames, device=device)
    vid_feat = viclip.infer_frames(frames=frame_tensors)

    if export_ov:
        viclip.export_ov_infer_frames(frames=frame_tensors)

    text_feats = viclip.infer_txts(texts=text_candidates)

    if export_ov:
        viclip.export_ov_infer_txts(text=text_candidates)

    text_feats_tensor = torch.cat(text_feats, 0)
    probs, idxs  = viclip.similarity(vid_feat=vid_feat, txt_fea=text_feats_tensor)
    print(f"== Pytorch scores = {probs.numpy()[0]}, idxs={idxs}")

def unit_test_ov(frames, text_candidates):
    print(f"== OpenVINO Version: {ov.get_version()}")
    device='GPU'
    ov_viclip = ViClipOV(device=device)
    vid_feat = ov_viclip.infer_frames(frames)
    print('vid type', type(vid_feat), len(vid_feat))
    #print(vid_feat)
    text_feats = ov_viclip.infer_txts(texts=text_candidates)
    

    #text_feats_tensor = torch.cat(text_feats, 0)
    print('text type', type(text_feats[0]), len(text_feats[0]))
    for idx, item in enumerate(text_feats):
        score = ov_viclip.cosine_similarity(text_vec=item, img_vec=vid_feat)
        print(idx, score)
    #print(f"== OpenVINO scores = {probs.numpy()[0]}, idxs={idxs}")

if __name__ == "__main__":
    video = cv2.VideoCapture('./example1.mp4')
    frames = [x for x in _frame_from_video(video)]

    text_candidates = ["A playful dog and its owner wrestle in the snowy yard, chasing each other with joyous abandon.",
        "A man in a gray coat walks through the snowy landscape, pulling a sleigh loaded with toys.",
        "A person dressed in a blue jacket shovels the snow-covered pavement outside their house.",
        "A pet dog excitedly runs through the snowy yard, chasing a toy thrown by its owner.",
        "A person stands on the snowy floor, pushing a sled loaded with blankets, preparing for a fun-filled ride.",
        "A man in a gray hat and coat walks through the snowy yard, carefully navigating around the trees.",
        "A playful dog slides down a snowy hill, wagging its tail with delight.",
        "A person in a blue jacket walks their pet on a leash, enjoying a peaceful winter walk among the trees.",
        "A man in a gray sweater plays fetch with his dog in the snowy yard, throwing a toy and watching it run.",
        "A person bundled up in a blanket walks through the snowy landscape, enjoying the serene winter scenery."
        ]

    #unit_test(frames, text_candidates)
    unit_test_ov(frames, text_candidates)
