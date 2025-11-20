#
#  Copyright (C) 2025 Intel Corporation
#
#  This software and the related documents are Intel copyrighted materials, 
#  and your use of them is governed by the express license under which they 
#  were provided to you ("License"). Unless the License provides otherwise, 
#  you may not use, modify, copy, publish, distribute, disclose or transmit 
#  his software or the related documents without Intel's prior written permission.
#
#  This software and the related documents are provided as is, with no express 
#  or implied warranties, other than those that are expressly stated in the License.
#

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from nltk.tokenize import word_tokenize
import jieba
import re
import copy, sys

# 下载nltk的punkt语料库，用于英文分词，如果已经下载过可注释掉这行
import nltk
import os

CUR_FILE_PATH=os.path.dirname(os.path.abspath(__file__))
sys.path.append(CUR_FILE_PATH + "/../utils/")
sys.path.append(CUR_FILE_PATH + "/../")

# 指定下载路径
download_path = ".\\nltk_data"  # 请替换为你希望的下载路径
os.environ['NLTK_DATA'] = '.\\nltk_data'
# 设置NLTK_DATA环境变量
#nltk.data.path.append(download_path)
# 设置下载目录
#nltk.download('punkt', download_dir=download_path)




def remove_extra_newlines_and_combine(text):
    # 删除多余的空行
    text_no_extra_newlines = re.sub(r'\n\s*\n', '\n', text)
    # 删除所有的换行符
    combined_text = text_no_extra_newlines.replace('\n', ' ')
    # 移除多余的空格
    final_text = re.sub(r'\s+', ' ', combined_text).strip()
    return final_text

def split_text_into_chunks(text, max_length=4800, limit_length=4800):
    # 调用remove_extra_newlines_and_combine函数处理文本
    cleaned_text = remove_extra_newlines_and_combine(text)

    # 初始化变量
    chunks = []
    current_chunk = ""

    # 分割符号
    delimiters = ".!？。；？！,，"

    for sentence in re.split(f"([{delimiters}])", cleaned_text):
        if len(current_chunk) + len(sentence) + 1 > max_length:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    # 合并chunks
    merged_chunks = []
    temp_chunk = ""

    for chunk in chunks:
        if len(temp_chunk) + len(chunk) <= limit_length:
            temp_chunk += " " + chunk
        else:
            if temp_chunk:
                merged_chunks.append(temp_chunk.strip())
            temp_chunk = chunk

    if temp_chunk:
        merged_chunks.append(temp_chunk.strip())

    return merged_chunks




def make_colon_as_title(sections):
    if not sections:
        return []
    if isinstance(sections[0], type("")):
        return sections
    i = 0
    while i < len(sections):
        txt, layout = sections[i]
        i += 1
        txt = txt.split("@")[0].strip()
        if not txt:
            continue
        if txt[-1] not in ":：":
            continue
        txt = txt[::-1]
        arr = re.split(r"([。？！!?;；]| \.)", txt)
        if len(arr) < 2 or len(arr[1]) < 32:
            continue
        sections.insert(i - 1, (arr[0][::-1], "title"))
        i += 1

# def naive_merge(sections, chunk_token_num=128, delimiter="\n。；！？"):
#     if not sections:
#         return []
#     if isinstance(sections[0], type("")):
#         sections = [(s, "") for s in sections]
#     cks = [""]
#     tk_nums = [0]

#     def add_chunk(t, pos):
#         nonlocal cks, tk_nums, delimiter
#         tnum = num_tokens_from_string(t)
#         if not pos: pos = ""
#         if tnum < 8:
#             pos = ""
#         # Ensure that the length of the merged chunk does not exceed chunk_token_num  
#         if tk_nums[-1] > chunk_token_num:

#             if t.find(pos) < 0:
#                 t += pos
#             cks.append(t)
#             tk_nums.append(tnum)
#         else:
#             if cks[-1].find(pos) < 0:
#                 t += pos
#             cks[-1] += t
#             tk_nums[-1] += tnum

#     for sec, pos in sections:
#         add_chunk(sec, pos)
#         continue
#         s, e = 0, 1
#         while e < len(sec):
#             if sec[e] in delimiter:
#                 add_chunk(sec[s: e + 1], pos)
#                 s = e + 1
#                 e = s + 1
#             else:
#                 e += 1
#         if s < e:
#             add_chunk(sec[s: e], pos)

#     return cks

def tokenize(d, t, eng):
    d["content_with_weight"] = t
    t = re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", t)
    d["content_ltks"] =  jieba.lcut(t)


def tokenize_chunks(chunks, doc, eng, pdf_parser=None):
    res = []
    # wrap up as es documents
    for ck in chunks:
        if len(ck.strip()) == 0:continue
        d = copy.deepcopy(doc)
        tokenize(d, ck, eng)
        res.append(d)
    return res

def tokenize_ocr(text):
    sections = []
    language = detect_language(text)
    sections = split_text_into_chunks(text)
    sections = [(l, "") for l in sections if l]
    make_colon_as_title(sections)
    sections = [s.split("@") for s, _ in sections]
    sections = [(pr[0], "@" + pr[1]) if len(pr) == 2 else (pr[0], '') for pr in sections ]
    chunks = []
    for text, iq in sections:
        chunks.append(text)
    eng = language == "en"
    res=[]
    for ck in chunks:
        if len(ck.strip()) == 0:continue
        t = re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", ck)
        res.extend(jieba.lcut(t))
    res = [item for item in res if item != ' ']
    return res

def chunk(filename, binary=None, from_page=0, to_page=1, callback=None, **kwargs):
    """
        Supported file formats are docx, pdf, txt.
        Since a book is long and not all the parts are useful, if it's a PDF,
        please setup the page ranges for every book in order eliminate negative effects and save elapsed computing time.
    """
    task_info = kwargs.get("task_info", {})
    if task_info and 'enable_llm' in task_info.keys():
        enable_llm = task_info['enable_llm']
        if not enable_llm:
            res = []
            d = {}
            d["content_with_weight"] = ''
            d["content_ltks"] = ''
            d['chunk_id']   = 0
            d['image']      = task_info['image']
            d['kb_id']      = task_info['kb_id']
            d['doc_id']     = task_info['doc_id']
            d['file']       = task_info['file']
            d['label']      = task_info['label']
            d['summary']    = task_info['summary']
            d['segment_id'] = task_info['segment_id']
            res.append(d)
            return res

    doc = { }
    sections = []
    language = 'zh'
    #logger.info("start to tokenlize file %s", filename)
    if re.search(r"\.txt$", filename, re.IGNORECASE):
        #logger.info("Start to parse for text.")
        txt = ""
     
        try:
            with open(filename, "r", encoding="utf8") as f:
                while True:
                    l = f.readline()
                    if not l:
                        break
                    txt += l
        except Exception as e:
            #logger.error(f"reading summary file {filename} failure  error {str(e)}")
            if txt == "" :
                txt = filename
        finally:
            language = detect_language(txt)
            sections = split_text_into_chunks(txt)
            sections = [(l, "") for l in sections if l]
            #logger.info("Finish parsing.")
    else:
        raise NotImplementedError(
            "file type not supported yet(doc, docx, pdf, txt supported)")

    make_colon_as_title(sections)

    sections = [s.split("@") for s, _ in sections]
    sections = [(pr[0], "@" + pr[1]) if len(pr) == 2 else (pr[0], '') for pr in sections ]
    # chunks = naive_merge(
    #     sections, kwargs.get(
    #         "chunk_token_num", 256), kwargs.get(
    #         "delimer", "\n。；！？"))
    chunks = []
    for text, iq in sections:
        chunks.append(text)
        #logger.info('chunk len=%s', len(text))
        # logger.info(text)
    #logger.info('chunk num=%s',len(chunks))

    eng = language == "en"
    res=[]
    res.extend(tokenize_chunks(chunks, doc, eng, None))

    # fill task info data before upload
    task_info = kwargs.get("task_info", {})
    if task_info:
        for idx, d in enumerate(res):
            #d['chunk_num'] = len(res)
            d['chunk_id']   = idx
            d['image']      = task_info['image']
            d['kb_id']      = task_info['kb_id']
            d['doc_id']     = task_info['doc_id']
            d['file']       = task_info['file']
            d['label']      = task_info['label']
            d['summary']    = task_info['summary']
            d['segment_id'] = task_info['segment_id']
    
    return res

def detect_language(text):
    """
    简单判断文本的语言类型，这里通过判断是否包含中文字符来区分中英文
    :param text: 要判断语言的文本内容
    :return: 'zh' 表示中文，'en' 表示英文
    """
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return 'zh'
    return 'en'

def tokenize_text(text, language):
    """
    对文本进行分词处理，支持中文和英文
    :param text: 要分词的文本内容
    :param language: 文本的语言类型，'zh' 表示中文，'en' 表示英文
    :return: 分词后的词语/单词列表
    """
    if language == 'zh':
        tokens = jieba.lcut(text)
    elif language == 'en':
        tokens = word_tokenize(text)
    else:
        print(f"暂不支持对 {language} 语言的文本进行分词，请检查文本内容！")
        return []
    return tokens
def read_and_chunk_text_file(file_path, chunk_size=1024):
    """
    读取文本文件，将其分割成块，并对每个块进行分词处理
    :param file_path: 文本文件的路径
    :param chunk_size: 每个块的大小（以字节为单位）
    :return: 分词后的块列表
    """
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在，请检查路径！")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # 去除换行符
    text = text.replace('\n', '')
    
    language = detect_language(text)
    tokens = tokenize_text(text, language)
    
    # 将分词后的文本分割成块，确保每个块包含一个完整的句子
    chunks = []
    current_chunk = []
    for token in tokens:
        current_chunk.append(token)
        if token in  ['。', '.', '！', '!', '？', '?'] :
            if len(' '.join(current_chunk)) >= chunk_size:  
                # 如果遇到句号、感叹号或问号，或者当前块的大小达到chunk_size，说明一个句子结束
                chunks.append(current_chunk)
                current_chunk = []
            else:
                continue
              
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks



if __name__ == "__main__":

    # input = '''·智能视频优化 ·智能语音处理 ·隐私与安全保障 Wenwen ww19 Liu Jun Jun7 ShiWhat is Lenovo Smart Meeting ·核心智能体验覆盖： ·智能视频优化 ·智能语音处理 ·隐私与安全保障 ·其它提升协同办公效率的功能 Guangchao GC1 Per ·未来探索更多垂直场景的可行性 Wenwen ww19 Liu Jun Jun7 ShiWhat is Lenovo Smart Meeting 联想PC在个人办公场景中的生产力，其核心功能包括： ·预装于联想PC中，在协同办公场景中为联想及三方软件提供支持 ·兼容主流在线会议软件，提供增强的智能会议体验 ·为联想PC上的其它软件提供智能协同的能力与服务 ·核心智能体验覆盖： ·智能视频优化 ·智能语音处理 ·隐私与安全保障 ·其它提升协同办公效率的功能 Guangchao GC1 Per ·与联想软硬件生态系统一起，提供增强的体验 ·未来探索更多垂直场景的可行性 *需要与法务团队确认，是否需要修改软件名称，以避免使用meeting一词 Wenwen ww19 Liu Jun Jun7 ShiWhat is Lenovo Smart Meeting 联想智会（Lenovo Smart Meeting*）是一款智能助手软件，旨在提供高效、安全的协同体验，同时提升 联想PC在个人办公场景中的生产力，其核心功能包括： ·预装于联想PC中，在协同办公场景中为联想及三方软件提供支持 ·兼容主流在线会议软件，提供增强的智能会议体验 ·为联想PC上的其它软件提供智能协同的能力与服务 ·核心智能体验覆盖： ·智能视频优化 ·智能语音处理 ·隐私与安全保障 ·其它提升协同办公效率的功能 Guangchao GC1 Per ·与联想软硬件生态系统一起，提供增强的体验 ·未来探索更多垂直场景的可行性 *需要与法务团队确认，是否需要修改软件名称，以避免使用meeting一词 Wenwen ww19 Liu Jun Jun7 Shi'''
    # tks = tokenize_ocr(input)
    # print(tks)

    from nlp.query import EsQueryer
    qryr = EsQueryer()
    keywords = qryr.question('今天天气不错哈！')
    print(keywords)


    # file_path = r"E:\MetaSearch\测试图片\nature\100_IMG_20240623_200015.txt"  # 替换为实际的txt文本文件路径
    # import sys

    # def dummy(prog=None, msg=""):
    #     pass
    # print("file name=",sys.argv[1] )
    # tokens = chunk(sys.argv[1], from_page=1, to_page=10, callback=dummy)
    # #tokens = tokenize_text_file(file_path)
    # print('tokens=', tokens)

    # # 使用示例
    # file_path = r"E:\MetaSearch\测试图片\nature\100_IMG_20240623_200015.txt"  # 替换为实际的txt文本文件路径
    # chunks = read_and_chunk_text_file(file_path, 384)

    # print("len of chunks", len(chunks))
    # print(chunks)


