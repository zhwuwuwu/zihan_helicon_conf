#ifndef VLM_PROCESSOR_H
#define VLM_PROCESSOR_H

#include <string>
#include <vector>
#include <windows.h>
#include <cstdint>
#ifdef _WIN32
#include <windows.h>
#endif

// 图片输入结构
struct ImageInput {
    std::string path;
};

// VLM处理器类
class VLMProcessor {
public:
    // 构造函数
    VLMProcessor(const std::string& so_path, 
                 const std::string& weights_path, 
                 uint32_t flag, 
                 const std::string& default_prompt = "");
    
    // 析构函数
    ~VLMProcessor();
    
    // 禁止拷贝构造和赋值操作
    VLMProcessor(const VLMProcessor&) = delete;
    VLMProcessor& operator=(const VLMProcessor&) = delete;
    
    // 图片推理
    std::vector<std::string> Inference(const std::vector<ImageInput>& input_list, 
                                      const std::string& prompt = "");
    
    // 纯文本推理
    std::string InferenceText(const std::string& input_text);
    
    // 检查是否已初始化
    bool IsInitialized() const { return handle_ != nullptr; }

private:
    // 底层C库函数指针类型定义
    typedef void* (__cdecl *CreateModelFunc)(uint32_t, const char*, uint32_t);
    typedef void (__cdecl *ReleaseModelFunc)(void*);
    typedef void (__cdecl *InferenceSummaryFunc)(void*, const char**, const char*, char**, uint32_t*, uint32_t, uint32_t);
    typedef void (__cdecl *InferenceTextFunc)(void*, const char*, char*, uint32_t*, uint32_t);
    
    // 成员变量
    HMODULE dll_handle_;                    // DLL句柄
    void* handle_;                          // 模型句柄
    std::string default_prompt_;            // 默认提示词
    uint32_t gen_count_;                    // 生成数量
    uint32_t max_batches_;                  // 最大批次数
    
    // 函数指针
    CreateModelFunc create_model_func_;         // 创建模型函数指针
    ReleaseModelFunc release_model_func_;       // 释放模型函数指针
    InferenceSummaryFunc inference_summary_func_; // 图片推理函数指针
    InferenceTextFunc inference_text_func_;     // 文本推理函数指针
    
    // 私有方法
    bool LoadLibrary(const std::string& so_path);
    void UnloadLibrary();
    bool CreateModel(uint32_t batch_size, const std::string& weights_path, uint32_t flag);
    void ReleaseModel();
};

#endif // VLM_PROCESSOR_H