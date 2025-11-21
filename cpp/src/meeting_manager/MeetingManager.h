#ifndef MEETING_MANAGER_H
#define MEETING_MANAGER_H

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <cstdint>

// 前向声明
class VLMProcessor;
class ChineseCLIPProcessor;
struct ImageInput;

class MeetingManager {
public:
    static MeetingManager& GetInstance();

    // 初始化和清理
    bool Initialize(const std::string& config_path = "");
    void Cleanup();
    bool IsInitialized() const { return initialized_; }

    // VLM模块接口
    std::vector<std::string> VLM_InferenceImages(
        const std::vector<ImageInput>& input_list,
        const std::string& prompt = ""
    );

    std::string VLM_InferenceText(const std::string& input_text);

    // ChineseCLIP模块接口
    std::vector<std::vector<float>> CLIP_EncodeText(const std::vector<std::string>& text_list);
    std::vector<std::vector<float>> CLIP_EncodeImage(const std::vector<std::string>& image_paths);
    float CLIP_CosineSimilarity(const std::vector<float>& text_vec, const std::vector<float>& img_vec);

    // 工具函数
    std::vector<std::string> GetImagePaths(const std::string& directory);

private:
    MeetingManager() = default;
    ~MeetingManager();

    // 禁止拷贝和赋值
    MeetingManager(const MeetingManager&) = delete;
    MeetingManager& operator=(const MeetingManager&) = delete;

    // 模块初始化
    bool InitializeVLMModule();
    void CleanupVLMModule();
    bool InitializeChineseCLIPModule();
    void CleanupChineseCLIPModule();

    // 成员变量
    mutable std::mutex mutex_;
    bool initialized_ = false;

    // 模块实例
    std::unique_ptr<VLMProcessor> vlm_processor_;
    std::unique_ptr<ChineseCLIPProcessor> clip_processor_;

    // VLM配置参数
    std::string vlm_dll_path_ = ".\\models\\qwen2p5vl.dll";
    std::string vlm_weights_path_ = ".\\models\\qwen2p5-3b";
    uint32_t vlm_flag_ = 1;
    std::string vlm_default_prompt_ = "这是一张在线会议软件的截图。请详细描述并总结这张图片的主要内容。请忽略参会人员，聊天窗口等次要信息。";

    // ChineseCLIP配置参数
    std::string clip_model_path_ = "./models/zclip_model";
    std::string clip_device_ = "GPU";
};

#endif // MEETING_MANAGER_H