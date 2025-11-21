#include "MeetingManager.h"
#include "VLMProcessor.h"
#include "ChineseCLIPProcessor.h"
#include <iostream>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

MeetingManager& MeetingManager::GetInstance() {
    static MeetingManager instance;
    return instance;
}

MeetingManager::~MeetingManager() {
    Cleanup();
}

bool MeetingManager::Initialize(const std::string& config_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (initialized_) {
        std::cout << "[MeetingManager] Already initialized" << std::endl;
        return true;
    }
    
    try {
        std::cout << "[MeetingManager] Initializing meeting manager..." << std::endl;
        
        if (!config_path.empty()) {
            std::cout << "[MeetingManager] Config path: " << config_path << std::endl;
            // TODO: 从配置文件加载参数
        }
        
        // 初始化VLM模块
        if (!InitializeVLMModule()) {
            std::cout << "[MeetingManager] Failed to initialize VLM module" << std::endl;
            return false;
        }

        // 初始化ChineseCLIP模块
        if (!InitializeChineseCLIPModule()) {
            std::cout << "[MeetingManager] Failed to initialize ChineseCLIP module" << std::endl;
            // 不返回false，允许只有VLM模块工作
        }
        
        // 未来可以在这里初始化其他模块
        // InitializeASRModule();
        // InitializeNLPModule();
        
        initialized_ = true;
        std::cout << "[MeetingManager] Meeting manager initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "[MeetingManager] Initialization failed: " << e.what() << std::endl;
        return false;
    }
}

void MeetingManager::Cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_) {
        return;
    }
    
    std::cout << "[MeetingManager] Cleaning up meeting manager..." << std::endl;
    
    // 清理各个模块
    CleanupVLMModule();
    CleanupChineseCLIPModule();
    // CleanupASRModule();
    // CleanupNLPModule();
    
    initialized_ = false;
    std::cout << "[MeetingManager] Meeting manager cleanup completed" << std::endl;
}

bool MeetingManager::InitializeVLMModule() {
    try {
        vlm_processor_ = std::make_unique<VLMProcessor>(
            vlm_dll_path_,
            vlm_weights_path_,
            vlm_flag_,
            vlm_default_prompt_
        );
        
        if (!vlm_processor_->IsInitialized()) {
            std::cout << "[MeetingManager] VLM processor initialization failed" << std::endl;
            vlm_processor_.reset();
            return false;
        }
        
        std::cout << "[MeetingManager] VLM module initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "[MeetingManager] VLM module initialization error: " << e.what() << std::endl;
        vlm_processor_.reset();
        return false;
    }
}

void MeetingManager::CleanupVLMModule() {
    if (vlm_processor_) {
        std::cout << "[MeetingManager] Cleaning up VLM module" << std::endl;
        vlm_processor_.reset();
    }
}

std::vector<std::string> MeetingManager::VLM_InferenceImages(
    const std::vector<ImageInput>& input_list,
    const std::string& prompt) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_ || !vlm_processor_) {
        std::cout << "[MeetingManager] VLM module not initialized" << std::endl;
        return {};
    }
    
    try {
        return vlm_processor_->Inference(input_list, prompt);
    } catch (const std::exception& e) {
        std::cout << "[MeetingManager] VLM inference error: " << e.what() << std::endl;
        return {};
    }
}

std::string MeetingManager::VLM_InferenceText(const std::string& input_text) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_ || !vlm_processor_) {
        std::cout << "[MeetingManager] VLM module not initialized" << std::endl;
        return "";
    }
    
    try {
        return vlm_processor_->InferenceText(input_text);
    } catch (const std::exception& e) {
        std::cout << "[MeetingManager] VLM text inference error: " << e.what() << std::endl;
        return "";
    }
}


bool MeetingManager::InitializeChineseCLIPModule() {
    try {
        clip_processor_ = std::make_unique<ChineseCLIPProcessor>(
            clip_model_path_,
            clip_device_
        );

        if (!clip_processor_->IsInitialized()) {
            std::cout << "[MeetingManager] ChineseCLIP processor initialization failed" << std::endl;
            clip_processor_.reset();
            return false;
        }

        std::cout << "[MeetingManager] ChineseCLIP module initialized successfully" << std::endl;
        return true;

    }
    catch (const std::exception& e) {
        std::cout << "[MeetingManager] ChineseCLIP module initialization error: " << e.what() << std::endl;
        clip_processor_.reset();
        return false;
    }
}

void MeetingManager::CleanupChineseCLIPModule() {
    if (clip_processor_) {
        std::cout << "[MeetingManager] Cleaning up ChineseCLIP module" << std::endl;
        clip_processor_.reset();
    }
}

std::vector<std::vector<float>> MeetingManager::CLIP_EncodeText(const std::vector<std::string>& text_list) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_ || !clip_processor_) {
        std::cout << "[MeetingManager] ChineseCLIP module not initialized" << std::endl;
        return {};
    }

    try {
        return clip_processor_->EncodeText(text_list);
    }
    catch (const std::exception& e) {
        std::cout << "[MeetingManager] ChineseCLIP text encoding error: " << e.what() << std::endl;
        return {};
    }
}

std::vector<std::vector<float>> MeetingManager::CLIP_EncodeImage(const std::vector<std::string>& image_paths) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_ || !clip_processor_) {
        std::cout << "[MeetingManager] ChineseCLIP module not initialized" << std::endl;
        return {};
    }

    try {
        return clip_processor_->EncodeImage(image_paths);
    }
    catch (const std::exception& e) {
        std::cout << "[MeetingManager] ChineseCLIP image encoding error: " << e.what() << std::endl;
        return {};
    }
}

float MeetingManager::CLIP_CosineSimilarity(const std::vector<float>& text_vec, const std::vector<float>& img_vec) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_ || !clip_processor_) {
        std::cout << "[MeetingManager] ChineseCLIP module not initialized" << std::endl;
        return 0.0f;
    }

    try {
        return clip_processor_->CosineSimilarity(text_vec, img_vec);
    }
    catch (const std::exception& e) {
        std::cout << "[MeetingManager] ChineseCLIP similarity calculation error: " << e.what() << std::endl;
        return 0.0f;
    }
}


std::vector<std::string> MeetingManager::GetImagePaths(const std::string& directory) {
    std::vector<std::string> extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"};
    std::vector<std::string> found_paths;
    
    try {
        if (!fs::exists(directory) || !fs::is_directory(directory)) {
            std::cout << "[MeetingManager] Directory not found: " << directory << std::endl;
            return {};
        }
        
        for (const auto& entry : fs::directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                
                if (std::find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
                    found_paths.push_back(entry.path().string());
                }
            }
        }
        
    } catch (const std::exception& e) {
        std::cout << "[MeetingManager] Error getting image paths: " << e.what() << std::endl;
    }
    
    return found_paths;
}
