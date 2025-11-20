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

import os
from enum import Enum

class FileStatus(Enum):
    NEW = "new"
    VLM_IN_PROGRESS = "vlm_in_progress"
    VLM_DONE = "vlm_done"
    EMBED_IN_PROGRESS = "embed_in_progress"
    EMBED_DONE = "embed_done"
    READY_FOR_QUERY = "ready_for_query"
    UNSUPPORTED  = "unsupport"

class FileTypeEx(Enum):
    IMAGE = "image"
    VIDEO = "video"
    DOCUMENT = "document"
    AUDIO = "audio"
    UNKNOWN = "unknown"


PROJECT_BASE = os.getenv("HELICON_SEARCH_PROJECT_BASE")
if PROJECT_BASE:
    PROJECT_BASE = PROJECT_BASE.strip()



class FileTypeClassifierEx:
    def __init__(self):
        # 初始化各类文件后缀名列表
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.svg', '.webp']
        self.video_extensions = ['.mov', '.avi', '.mp4', '.mkv', '.flv', '.wmv']
        self.document_extensions = ['.doc', '.docx', '.pdf', '.txt', '.xls', '.xlsx', '.ppt', '.pptx']
        self.audio_extensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg']

    def get_file_type(self, file_extension):
        """
        根据文件后缀名判断文件类型
        """
        file_extension = file_extension.lower()  # 将后缀名转换为小写

        if file_extension in self.image_extensions:
            return FileTypeEx.IMAGE.value
        elif file_extension in self.video_extensions:
            return FileTypeEx.VIDEO.value
        elif file_extension in self.document_extensions:
            return FileTypeEx.DOCUMENT.value
        elif file_extension in self.audio_extensions:
            return FileTypeEx.AUDIO.value
        else:
            return FileTypeEx.UNKNOWN.value

    # 判断文件是否是图像或视频
    def is_image_or_video(self, file):
        ext = os.path.splitext(file)[1].lower()
        return ext in self.image_extensions or ext in self.video_extensions

    # 遍历目录并收集所有图像和视频文件
    def get_all_media_files(self, dir_path, media_files=None):
        """
                  遍历指定目录及其子目录，收集所有图像和视频文件的完整路径

        :param dir_path: 要遍历的根目录路径
        :param media_files: 用于收集文件路径的列表，默认为None（内部会初始化为空列表）
        :return: 包含所有图像和视频文件完整路径的列表
        """
        if media_files is None:
            media_files = []

        try:
            for root, dirs, files in os.walk(dir_path):
                # 遍历当前目录下的所有文件
                for file in files:
                    full_path = os.path.join(root, file)
                    try:
                        if self.is_image_or_video(full_path):
                            media_files.append(full_path)
                    except Exception as e:
                        print(f"在判断文件 {full_path} 是否为图像或视频文件时出现异常: {e}")

                # 递归遍历当前目录下的所有子目录
                #for sub_dir in dirs:
                #    sub_dir_path = os.path.join(root, sub_dir)
                #    self.get_all_media_files(sub_dir_path, media_files)
        except OSError as os_err:
            print(f"遍历目录 {dir_path} 时出现操作系统相关错误: {os_err}")

        return media_files
    def get_home_cache_dir():
        dir = os.path.join(os.path.expanduser('~'), ".metasearch")
        try:
            os.mkdir(dir)
        except OSError as error:
            pass
        return dir

    def get_project_base_directory(*args):
        global PROJECT_BASE
        if PROJECT_BASE is None:
            PROJECT_BASE = os.path.abspath(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    os.pardir,
                    os.pardir,
                )
            )
        else:
            PROJECT_BASE = PROJECT_BASE.strip()

        if args:
            return os.path.join(PROJECT_BASE, *args)
        return PROJECT_BASE


def remove_files(files):
    for file in files:
        try:
            if os.path.exists(file):
                os.remove(file)
                print(f"File {file} is unuseful and deleted done.")
            else:
                print(f"File {file} is not exist.")
        except Exception as e:
            print(f"Failed to remove file {file}: {e}")
