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

import base64
import uuid
import tiktoken
def get_uuid():
    return uuid.uuid1().hex

def encode_image_to_base64(file_path):
    """
    Encodes an image file to Base64.

    :param file_path: Path to the image file
    :return: Base64 encoded string
    """
    try:
        with open(file_path, "rb") as image_file:
            base64_encoded = base64.b64encode(image_file.read()).decode('utf-8')
        return base64_encoded
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    try:
        num_tokens = len(encoder.encode(string))
        return num_tokens
    except Exception as e:
        pass
    return 0

def rmBackslash(value):
    # 去掉两端空格和多余的反斜杠
    value = value.strip()  # 去掉两端空格
    while value.endswith('/') or value.endswith('\\'):
        value = value[:-1]
    return value
def rmSpace(s):
    # 去掉多余的空格
    return s.replace(" ", "")

def truncate(string: str, max_len: int) -> str:
    """Returns truncated text if the length of text exceed max_len."""
    return encoder.decode(encoder.encode(string)[:max_len])

def time_to_seconds(time_str):
    """
    将时间格式 00:00:04.333 转换为秒。
    
    :param time_str: 时间字符串，例如 '00:00:04.333'
    :return: 转换后的秒数 (float)
    """
    # 分割时间为小时、分钟和秒.毫秒
    parts = time_str.split(':')
    if len(parts) != 3:
        raise ValueError("Invalid time format. Expected format: hh:mm:ss.milliseconds")
    
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])  # 秒包含小数部分
    
    # 转换为总秒数
    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds
