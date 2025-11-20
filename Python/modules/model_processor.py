import os
import ctypes
from ctypes import c_void_p, c_uint32, c_char_p, POINTER, c_float, c_char

class ModelProcessor:
    def __init__(self, so_path='.\\models\\qwen2p5vl.dll', weights='.\\models\\qwen2p5-3b', flag = 1, prompt=''):
        self.lib = ctypes.CDLL(so_path)
        self.lib.createModelQwen2vl.restype = c_void_p
        self.lib.createModelQwen2vl.argtypes = [c_uint32, c_char_p, c_uint32]
        self.lib.releaseModelQwen2vl.restype = None
        self.lib.releaseModelQwen2vl.argtypes = [c_void_p]
        self.lib.inferenceSummaryQwen2vl.restype = None
        self.lib.inferenceSummaryQwen2vl.argtypes = [c_void_p, POINTER(c_char_p), c_char_p, POINTER(c_char_p), POINTER(c_uint32), c_uint32, c_uint32]

        # init/create
        self.handle = self.create(8, weights, flag)

        self.gen_count = 800
        self.char_list = ['\0'] * 4096
        self.max_batches = int(8)
        self.output_strings = []
        for i in range(self.max_batches):
            self.output_strings.append(str(self.char_list))
        self.output_len = [c_uint32(0)] * self.max_batches
        self.prompt = '''这是一张在线会议软件的截图。请详细描述并总结这张图片的主要内容。请忽略参会人员，聊天窗口等次要信息。'''
        if prompt:
            self.prompt = prompt

    def create(self, llm_batch_size, model_path, flag):
        return self.lib.createModelQwen2vl(c_uint32(llm_batch_size), c_char_p(model_path.encode('utf-8')), flag)

    def release(self):
        if self.handle:
            self.lib.releaseModelQwen2vl(self.handle)

    def inference(self, input_list, prompt=''):
        if not len(input_list):
            return []
        if prompt:
            self.prompt = prompt
        input_files = []
        for item in input_list:
            input_files.append(item['path'])
        input_files_array = (ctypes.c_char_p * len(input_files))(*[f.encode('utf-8') for f in input_files])
        text_array = (ctypes.c_char_p * len(self.output_strings))(*[tt.encode('utf-8') for tt in self.output_strings])
        len_array = (c_uint32 * len(self.output_len))(*self.output_len)

        self.lib.inferenceSummaryQwen2vl(self.handle, input_files_array, c_char_p(self.prompt.encode('utf-8')), text_array, len_array, len(input_files), self.gen_count)

        # 将字节码转为文本内容并存储在结果列表中
        results = []
        for i in range(len(input_files)):
            byte_to_read = len_array[i]
            byte_data = text_array[i][0:byte_to_read]
            try:
                # 尝试使用UTF-8解码，如果失败则使用默认错误处理
                text_content = byte_data.decode('utf-8')
            except UnicodeDecodeError:
                # 使用替换模式处理解码错误（�表示无法解码的字符）
                text_content = byte_data.decode('utf-8', errors='replace')
            # if text_content:
            #     text_content = str(text_content).replace('\n', '')
            results.append(text_content)
        if len(results) == len(input_files):
            return results
        else:
            return []



import os
import glob

def get_image_paths(directory):
    # Define the image extensions you want to include
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff']

    # Initialize an empty list to store the full paths
    image_paths = []

    # Iterate over each extension and find matching files
    for extension in image_extensions:
        # Use glob to find all files matching the extension
        image_paths.extend(glob.glob(os.path.join(directory, extension)))

    return image_paths




if __name__ == '__main__':
    # Example usage
    directory_path = r'C:\Users\SAS\Downloads\multi-modal-meeting-system\Teams_20250730_113641\32c067d9-2043-4652-bbf3-84ad957d68cd'
    all_image_paths = get_image_paths(directory_path)
    #print(all_image_paths)
    from ocr_module import PaddleOCRWithOpenVINO
    ocr = PaddleOCRWithOpenVINO(models_dir='.\\models\\paddle_ocr')
    qwen2p5vl_mdl = ModelProcessor()
    prompt_ori = '''这是一张在线会议软件的截图。请详细描述并总结这张图片的主要内容。请忽略参会人员，聊天窗口等次要信息。'''
    prompt_new = '这是一张在线会议软件的截图。你必须根据事实详细全面描述并总结该图片的基础内容，不要虚构内容。请忽略工具栏介绍等次要信息，可借助辅助信息补全总结。辅助信息：'
    image_list = all_image_paths #['1753847132023_00106.jpg']
    for image in image_list:
        text = ocr.run_paddle_ocr(image=image)
        print(text)
        result_ori = qwen2p5vl_mdl.inference([{'path':image}], prompt_ori)
        print(result_ori)
        prompt_new = prompt_new + text
        prompt_new = prompt_new[:800]
        print(len(prompt_new))
        result_new = qwen2p5vl_mdl.inference([{'path':image}], prompt_new)
        print(result_new)

        with open('output.txt', 'a', encoding='gbk') as file:
            file.write('--------------------------------------------\n')
            file.write(image + '\n')
            file.write(result_ori[0] + '\n')
            file.write('=====\n')
            file.write(result_new[0] + '\n')

    
    # input_files = '1753847132023_00106.jpg'
    # from ocr_module import PaddleOCRWithOpenVINO
    # ocr = PaddleOCRWithOpenVINO(models_dir='.\\models\\paddle_ocr')
    # image_path = input_files
    # text = ocr.run_paddle_ocr(image=image_path)
    # print("Extracted Text:", text)

    # prompt = '这是一张在线会议软件的截图。你必须根据事实详细全面描述并总结该图片的基础内容，不要虚构内容。请忽略工具栏介绍等次要信息，可借助辅助信息补全总结。辅助信息：' + text
    # prompt = prompt[:800]
    # #print(prompt)
    # print(len(prompt))
    # qwen2p5vl_mdl = ModelProcessor(prompt=prompt)
    # ptrtype = c_char * 4096
    # char_list = ['\0'] * 4096
    # max_batches = int(8)
    # output_strings = []
    # for i in range(max_batches):
    #     output_strings.append(str(char_list))
    # output_len = [c_uint32(0)] * max_batches
    # #
    # # ['这张图片展示了一个在线会议软件的界面，标题为“New Product Experience Monitoring - Topic Design”。
    # # 主要内容包括一个产品描述区域和一个流程图。\n\n1. **产品描述区域**：\n   - 包含了产品的名称、版本号（V2.0）、
    # # 功能说明以及一些技术细节。例如，提到在Windows 10上运行的产品，并且提到了某些技术细节如“API”、“API-FX”等。\n  
    # #  - 还提到了一些关于性能优化的信息，比如“性能优化到5%”。\n\n2. **流程图**：\n   - 流程图详细展示了从某个初始状态开始的一系列步骤或操作。
    # # 每个步骤都用箭头连接起来表示顺序关系。\n   - 流程中包含了一些关键点和注意事项，如确保所有步骤都按照正确的顺序进行，
    # # 并注意某些特定的操作点。\n\n总结：这张图片主要展示了新产品的体验监控主题设计内容及其相关的流程图。它详细介绍了产品的名称、
    # # 版本号以及一些关键技术细节，并通过流程图的形式清晰地展示了从初始状态到最终结果的整个过程中的关键步骤和注意事项。']
    # #
    # generation = 500
    
    # #prompt = '''这是一张在线会议软件的截图。请详细描述并总结这张图片的主要内容。请忽略参会人员，聊天窗口等次要信息。直接输出内容信息，不要在开头添加额外介绍。'''
    
    # for i in range(1):
    #     result = qwen2p5vl_mdl.inference([{'path':input_files}])

    # print('result =', result)
    # qwen2p5vl_mdl.release()

    