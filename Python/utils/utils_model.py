import os
import torch
import uuid
import enum
import numpy as np

class TaskType(enum.Enum):
    FeaExtract = 1
    Reranker = 2
    Translate = 3
    
def get_model_path():
    return os.path.dirname(os.path.abspath(__file__)) + '/../model_weights/'

def get_model_full_path(model_id):
    return get_model_path() + model_id + os.sep

def get_logits_fea(last_token_logits, tokenizer,last_score=None):
    target_token_strings = ['Y', 'N'] # <--- *** ASSUMPTION - VERIFY THIS ***
    #print(f"Attempting to find logits for tokens: {target_token_strings}")

    try:
        token_ids = {}
        problematic_tokens = {}
        for token_str in target_token_strings:
            # Use encode, ensuring it adds neither BOS nor EOS unless intended,
            # and that it returns a single ID for the target token string.
            encoded = tokenizer.encode(token_str, add_special_tokens=False)
            if len(encoded) == 1:
                token_ids[token_str] = encoded[0]
            else:
                problematic_tokens[token_str] = encoded
                print(f"Warning: Token '{token_str}' did not encode to a single ID: {encoded}. Check the exact token string (spaces matter!).")

        if not token_ids:
            raise ValueError(f"Could not find single token IDs for any of the target tokens: {target_token_strings}. Please verify the exact token strings (e.g., with leading spaces) used by the Qwen2-VL-2B-Instruct tokenizer.")

        #print(f"Found token IDs: {token_ids}")

        # Extract the logit values for these specific token IDs
        extracted_logits = {}
        vocab_size = last_token_logits.shape[0]
        for token_str, token_id in token_ids.items():
            if 0 <= token_id < vocab_size:
                extracted_logits[token_str] = last_token_logits[token_id].item() # Get Python float
                if last_score != None:
                    print(f'== Score: {last_score[token_id].item()}')
            else:
                print(f"Warning: Token ID {token_id} for '{token_str}' is out of vocab bounds ({vocab_size}).")

        # Print the extracted logits
        #print("\n--- Extracted Logits ---")
        #for token_str, logit_value in extracted_logits.items():
            #print(f"Logit for token '{token_str}' (ID: {token_ids[token_str]}): {logit_value:.4f}")

        # (Optional) Compare and calculate relative probabilities
        if len(extracted_logits) == 2: # Assuming two target tokens like Yes/No
            logit_vals = list(extracted_logits.values())
            token_strs = list(extracted_logits.keys())
            #print(f"\nComparison: Logit('{token_strs[0]}') {' > ' if logit_vals[0] > logit_vals[1] else ' < ' if logit_vals[0] < logit_vals[1] else ' = '} Logit('{token_strs[1]}')")

            # Calculate relative probability using Softmax over these two logits
            prob_tensor = torch.softmax(torch.tensor(logit_vals), dim=0)
            #print(f"Relative Probability (considering only these two tokens):")
            #print(f"  P('{token_strs[0]}') = {prob_tensor[0].item():.4f}")
            #print(f"  P('{token_strs[1]}') = {prob_tensor[1].item():.4f}")
            sim = prob_tensor[0].item()
    except ValueError as e:
        print(f"\nError finding token IDs: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        
    return sim

def get_logits_fea_common(last_token_logits:list, token_yn_ids):
    extracted_logits = []
    vocab_size = np.shape(last_token_logits)[-1]
    for [token_id] in token_yn_ids:
        if 0 <= token_id < vocab_size:
            extracted_logits.append(last_token_logits[-1][token_id])
        else:
            print(f"== Warning: Token ID {token_id} is out of vocab bounds ({vocab_size}).")
            exit()

    if len(extracted_logits) != 2: # Assuming two target tokens like Yes/No
        print(f"== Error: An unexpected error occurred.")
        exit()

    # Calculate relative probability using Softmax over these two logits
    prob_tensor = torch.softmax(torch.tensor(extracted_logits), dim=0)
    sim = prob_tensor[0].item()
    return sim

class cache_desc:
    def __init__(self, model_id_uuid_desc:str, language):
        self.__model_id_uuid = uuid.uuid3(uuid.NAMESPACE_DNS, model_id_uuid_desc)
        self.__language = language

    def __get_des_fn(self, fn:str):
        return os.path.splitext(fn)[0] + "_" + str(self.__model_id_uuid) + "_" + self.__language + ".txt"
    
    def exist_fns(self, imgs:list[str]):
        for id, fn in enumerate(imgs):
            ffn = self.__get_des_fn(fn)
            if not os.path.exists(ffn):
                return False
        return True
        
    def save_des(self, imgs:list[str], dess):
        for id, fn in enumerate(imgs):
            ffn = self.__get_des_fn(fn)
            with open(ffn, 'w') as f:
                f.write(dess[id])
    def load_des(self, imgs:list[str]):
        dess = []
        for id, fn in enumerate(imgs):
            ffn = self.__get_des_fn(fn)
            with open(ffn, 'r') as f:
                dess.append(f.read())
        return dess