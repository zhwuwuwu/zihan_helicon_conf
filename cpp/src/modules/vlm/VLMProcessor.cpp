#include "VLMProcessor.h"
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <windows.h> 

VLMProcessor::VLMProcessor(const std::string& so_path, 
                          const std::string& weights_path, 
                          uint32_t flag, 
                          const std::string& default_prompt)
    : dll_handle_(nullptr)
    , handle_(nullptr)
    , default_prompt_(default_prompt.empty() ? 
        "这是一张在线会议软件的截图。请详细描述并总结这张图片的主要内容。请忽略参会人员，聊天窗口等次要信息。" : 
        default_prompt)
    , gen_count_(800)
    , max_batches_(8)
    , create_model_func_(nullptr)
    , release_model_func_(nullptr)
    , inference_summary_func_(nullptr)
    , inference_text_func_(nullptr) {
    
    if (!LoadLibrary(so_path)) {
        throw std::runtime_error("Failed to load VLM library: " + so_path);
    }
    
    if (!CreateModel(8, weights_path, flag)) {
        UnloadLibrary();
        throw std::runtime_error("Failed to create VLM model");
    }
    
    std::cout << "[VLMProcessor] Initialized successfully" << std::endl;
}

VLMProcessor::~VLMProcessor() {
    ReleaseModel();
    UnloadLibrary();
    std::cout << "===============================[VLMProcessor] ~VLMProcessor successfully" << std::endl;
}

bool VLMProcessor::LoadLibrary(const std::string& so_path) {
    dll_handle_ = ::LoadLibraryA(so_path.c_str());
    if (!dll_handle_) {
        std::cout << "[VLMProcessor] Failed to load library: " << so_path << std::endl;
        return false;
    }
    
    // 获取函数指针
    create_model_func_ = reinterpret_cast<CreateModelFunc>(
        GetProcAddress(dll_handle_, "createModelQwen2vl"));
    release_model_func_ = reinterpret_cast<ReleaseModelFunc>(
        GetProcAddress(dll_handle_, "releaseModelQwen2vl"));
    inference_summary_func_ = reinterpret_cast<InferenceSummaryFunc>(
        GetProcAddress(dll_handle_, "inferenceSummaryQwen2vl"));
    
    // 尝试获取文本推理函数（可能不存在）
    inference_text_func_ = reinterpret_cast<InferenceTextFunc>(
        GetProcAddress(dll_handle_, "inferenceTextQwen2vl"));
    
    if (!create_model_func_ || !release_model_func_ || !inference_summary_func_) {
        std::cout << "[VLMProcessor] Failed to get required function pointers" << std::endl;
        UnloadLibrary();
        return false;
    }
    
    return true;
}

void VLMProcessor::UnloadLibrary() {
    if (dll_handle_) {
        FreeLibrary(dll_handle_);
        dll_handle_ = nullptr;
    }
    
    create_model_func_ = nullptr;
    release_model_func_ = nullptr;
    inference_summary_func_ = nullptr;
    inference_text_func_ = nullptr;
}

bool VLMProcessor::CreateModel(uint32_t batch_size, const std::string& weights_path, uint32_t flag) {
    if (!create_model_func_) {
        return false;
    }
    
    handle_ = create_model_func_(batch_size, weights_path.c_str(), flag);
    return handle_ != nullptr;
}

void VLMProcessor::ReleaseModel() {
    if (handle_ && release_model_func_) {
        release_model_func_(handle_);
        handle_ = nullptr;
    }
}

std::vector<std::string> VLMProcessor::Inference(const std::vector<ImageInput>& input_list, 
                                                const std::string& prompt) {
    if (!handle_ || !inference_summary_func_ || input_list.empty()) {
        return {};
    }
    
    try {
        // 使用提供的prompt或默认prompt
        std::string use_prompt = prompt.empty() ? default_prompt_ : prompt;
        
        // 准备输入文件路径数组
        std::vector<const char*> input_files;
        for (const auto& input : input_list) {
            input_files.push_back(input.path.c_str());
        }
        
        // 准备输出缓冲区
        std::vector<std::vector<char>> output_buffers(input_list.size());
        std::vector<char*> output_ptrs(input_list.size());
        std::vector<uint32_t> output_lengths(input_list.size(), 0);
        
        for (size_t i = 0; i < input_list.size(); ++i) {
            output_buffers[i].resize(4096);
            output_ptrs[i] = output_buffers[i].data();
        }
        
        // 调用推理函数
        inference_summary_func_(
            handle_,
            input_files.data(),
            use_prompt.c_str(),
            output_ptrs.data(),
            output_lengths.data(),
            static_cast<uint32_t>(input_list.size()),
            gen_count_
        );
        
        // 处理结果
        std::vector<std::string> results;
        for (size_t i = 0; i < input_list.size(); ++i) {
            if (output_lengths[i] > 0) {
                std::string result(output_ptrs[i], output_lengths[i]);
                // 移除换行符（如果需要）
                // result.erase(std::remove(result.begin(), result.end(), '\n'), result.end());
                results.push_back(result);
            } else {
                results.push_back("");
            }
        }
        
        return results;
    }
    catch (const std::exception& e) {
        std::cout << "[VLMProcessor] Inference error: " << e.what() << std::endl;
        return {};
    }
}

std::string VLMProcessor::InferenceText(const std::string& input_text) {
    if (!handle_ || !inference_text_func_) {
        // 如果没有专门的文本推理函数，可以尝试使用图片推理函数的变体
        // 或者返回错误信息
        return "Text inference not supported by current model";
    }
    
    try {
        std::vector<char> output_buffer(4096);
        uint32_t output_length = 0;
        
        inference_text_func_(
            handle_,
            input_text.c_str(),
            output_buffer.data(),
            &output_length,
            static_cast<uint32_t>(output_buffer.size())
        );
        
        if (output_length > 0) {
            return std::string(output_buffer.data(), output_length);
        }
        
        return "";
    }
    catch (const std::exception& e) {
        std::cout << "[VLMProcessor] Text inference error: " << e.what() << std::endl;
        return "";
    }
}