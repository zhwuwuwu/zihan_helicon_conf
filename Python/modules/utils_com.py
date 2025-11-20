import torch
import os
import json

def set_device(dev_id, use_env=False):
    if torch.cuda.is_available():
        if use_env:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(dev_id)
        else:
            # torch.set_default_device('cpu')
            if torch.cuda.is_available():
                if dev_id < torch.cuda.device_count():
                    torch.cuda.set_device(dev_id)
                else:
                    print(f"== Warning: dev_id[{dev_id}] should be < device_count[{torch.cuda.device_count()}], set dev_id to 0.")
                    torch.cuda.set_device(0)
        print(f"== Using CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}, device_id={torch.cuda.current_device()}")
    else:
        print(f"== CUDA is not available. Default CPU.")

import openvino as ov
def check_intel_gpu_openvino():
    core = ov.Core()
    available_devices = core.available_devices
    
    intel_gpu_found = False
    for device in available_devices:
        if "GPU" in device: # OpenVINO lists Intel GPUs as 'GPU', 'GPU.0', 'GPU.1', etc.
            intel_gpu_found = True
            print(f"== Intel GPU found via OpenVINO: {core.get_property(device, 'FULL_DEVICE_NAME')}")
    
    if not intel_gpu_found:
        print("== No Intel GPU found via OpenVINO.")
    return intel_gpu_found

def get_cur_device():
    if torch.cuda.is_available():
        return f'cuda:{torch.cuda.current_device()}'
    elif check_intel_gpu_openvino():
        return 'GPU'
    else:
        return 'CPU'

def list_files_in_directory(directory, ext='.txt'):
    txt_files = []
    try:
        # List all entries in the directory
        entries = os.listdir(directory)
        for entry in entries:
            full_path = os.path.join(directory, entry)
            # Check if it's a file and ends with .txt (case-insensitive check included)
            if os.path.isfile(full_path) and entry.lower().endswith(ext):
                txt_files.append(full_path) # Or just entry if you want only filenames
    except FileNotFoundError:
        print(f"Error: Directory not found at {directory}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return txt_files

def load_json_file(filepath):
    """
    Loads data from a JSON file.

    Args:
        filepath (str): The path to the JSON file.

    Returns:
        dict or list: The parsed JSON data. Returns None if an error occurs.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"== Error: The file '{filepath}' was not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"== Error: Could not decode JSON from '{filepath}'. Details: {e}")
        return None
    except Exception as e:
        print(f"== Error: An unexpected error occurred while reading '{filepath}'. Details: {e}")
        return None