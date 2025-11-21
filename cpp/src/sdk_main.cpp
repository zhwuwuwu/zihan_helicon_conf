#include "HeliconSearchSDK.h"
#include "meeting_manager/MeetingManager.h"
#include "../modules/vlm/VLMProcessor.h"
#include <iostream>
#include <vector>
#include <cstring>

// 实现SDK接口函数
extern "C" {

HELICON_API Helicon_ErrorCode Helicon_Initialize(const char* config_path) {
    try {
        std::cout << "[HeliconSDK] Initializing SDK..." << std::endl;
        
        std::string config = config_path ? config_path : "";
        
        if (MeetingManager::GetInstance().Initialize(config)) {
            std::cout << "[HeliconSDK] SDK initialized successfully" << std::endl;
            return HELICON_SUCCESS;
        } else {
            std::cout << "[HeliconSDK] SDK initialization failed" << std::endl;
            return HELICON_ERROR_INIT_FAILED;
        }
        
    } catch (const std::exception& e) {
        std::cout << "[HeliconSDK] Initialization exception: " << e.what() << std::endl;
        return HELICON_ERROR_UNKNOWN;
    }
}

HELICON_API void Helicon_Cleanup() {
    try {
        std::cout << "[HeliconSDK] Cleaning up SDK..." << std::endl;
        MeetingManager::GetInstance().Cleanup();
        std::cout << "[HeliconSDK] SDK cleanup completed" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "[HeliconSDK] Cleanup exception: " << e.what() << std::endl;
    }
}

HELICON_API const char* Helicon_GetVersion() {
    static const std::string version = std::to_string(HELICON_SDK_VERSION_MAJOR) + "." +
                                      std::to_string(HELICON_SDK_VERSION_MINOR) + "." +
                                      std::to_string(HELICON_SDK_VERSION_PATCH);
    return version.c_str();
}

HELICON_API int Helicon_IsInitialized() {
    return MeetingManager::GetInstance().IsInitialized() ? 1 : 0;
}

// === VLM模块接口 ===
HELICON_API Helicon_ErrorCode Helicon_VLM_InferenceImages(
    const Helicon_ImageInput* input_list,
    uint32_t input_count,
    const char* prompt,
    Helicon_InferenceResult* results,
    uint32_t* result_count) {
    
    if (!input_list || !results || !result_count || input_count == 0) {
        return HELICON_ERROR_INVALID_PARAM;
    }
    
    try {
        // 转换输入格式
        std::vector<ImageInput> cpp_inputs;
        for (uint32_t i = 0; i < input_count; ++i) {
            cpp_inputs.push_back({input_list[i].path});
        }
        
        // 调用会议管理器
        std::string use_prompt = prompt ? prompt : "";
        std::vector<std::string> cpp_results = 
            MeetingManager::GetInstance().VLM_InferenceImages(cpp_inputs, use_prompt);
        
        // 转换输出格式
        *result_count = static_cast<uint32_t>(cpp_results.size());
        for (size_t i = 0; i < cpp_results.size() && i < input_count; ++i) {
            const std::string& result_text = cpp_results[i];
            results[i].length = static_cast<uint32_t>(result_text.length());
            results[i].text = new char[result_text.length() + 1];
            strcpy_s(results[i].text, result_text.length() + 1, result_text.c_str());
        }
        
        return HELICON_SUCCESS;
        
    } catch (const std::exception& e) {
        std::cout << "[HeliconSDK] VLM inference exception: " << e.what() << std::endl;
        return HELICON_ERROR_MODULE_ERROR;
    }
}

HELICON_API Helicon_ErrorCode Helicon_VLM_InferenceText(
    const char* input_text,
    char* output_text,
    uint32_t* output_length,
    uint32_t max_output_length) {
    
    if (!input_text || !output_text || !output_length) {
        return HELICON_ERROR_INVALID_PARAM;
    }
    
    try {
        std::string result = MeetingManager::GetInstance().VLM_InferenceText(input_text);
        
        if (result.length() >= max_output_length) {
            result = result.substr(0, max_output_length - 1);
        }
        
        strcpy_s(output_text, max_output_length, result.c_str());
        *output_length = static_cast<uint32_t>(result.length());
        
        return HELICON_SUCCESS;
        
    } catch (const std::exception& e) {
        std::cout << "[HeliconSDK] VLM text inference exception: " << e.what() << std::endl;
        return HELICON_ERROR_MODULE_ERROR;
    }
}

HELICON_API void Helicon_FreeResults(Helicon_InferenceResult* results, uint32_t count) {
    if (!results) {
        return;
    }
    
    for (uint32_t i = 0; i < count; ++i) {
        delete[] results[i].text;
        results[i].text = nullptr;
        results[i].length = 0;
    }
}

HELICON_API Helicon_ErrorCode Helicon_GetImagePaths(
    const char* directory,
    char*** image_paths,
    uint32_t* path_count) {
    
    if (!directory || !image_paths || !path_count) {
        return HELICON_ERROR_INVALID_PARAM;
    }
    
    try {
        std::vector<std::string> found_paths = 
            MeetingManager::GetInstance().GetImagePaths(directory);
        
        // 分配内存并复制路径
        *path_count = static_cast<uint32_t>(found_paths.size());
        *image_paths = new char*[found_paths.size()];
        
        for (size_t i = 0; i < found_paths.size(); ++i) {
            (*image_paths)[i] = new char[found_paths[i].length() + 1];
            strcpy_s((*image_paths)[i], found_paths[i].length() + 1, found_paths[i].c_str());
        }
        
        return HELICON_SUCCESS;
        
    } catch (const std::exception& e) {
        std::cout << "[HeliconSDK] Get image paths exception: " << e.what() << std::endl;
        return HELICON_ERROR_UNKNOWN;
    }
}

HELICON_API void Helicon_FreeImagePaths(char** image_paths, uint32_t count) {
    if (!image_paths) {
        return;
    }
    
    for (uint32_t i = 0; i < count; ++i) {
        delete[] image_paths[i];
    }
    delete[] image_paths;
}

// === ChineseCLIP模块接口 ===
//HELICON_API Helicon_ErrorCode Helicon_CLIP_EncodeText(
//    const char** text_list,
//    uint32_t text_count,
//    float** embeddings,
//    uint32_t* embedding_count,
//    uint32_t* embedding_dim) {
//
//    if (!text_list || !embeddings || !embedding_count || !embedding_dim || text_count == 0) {
//        return HELICON_ERROR_INVALID_PARAM;
//    }
//
//    try {
//        // 转换输入格式
//        std::vector<std::string> cpp_texts;
//        for (uint32_t i = 0; i < text_count; ++i) {
//            if (text_list[i]) {
//                cpp_texts.push_back(text_list[i]);
//            }
//        }
//
//        if (cpp_texts.empty()) {
//            return HELICON_ERROR_INVALID_PARAM;
//        }
//
//        // 调用会议管理器
//        std::vector<std::vector<float>> cpp_results =
//            MeetingManager::GetInstance().CLIP_EncodeText(cpp_texts);
//
//        if (cpp_results.empty()) {
//            return HELICON_ERROR_MODULE_ERROR;
//        }
//
//        // 转换输出格式
//        *embedding_count = static_cast<uint32_t>(cpp_results.size());
//        *embedding_dim = cpp_results[0].empty() ? 0 : static_cast<uint32_t>(cpp_results[0].size());
//
//        if (*embedding_dim == 0) {
//            return HELICON_ERROR_MODULE_ERROR;
//        }
//
//        embeddings = new float* [*embedding_count];  // 分配指针数组
//        for (uint32_t i = 0; i < *embedding_count; ++i) {
//            (embeddings)[i] = new float[*embedding_dim];  // 为每个指针分配float数组
//            for (uint32_t j = 0; j < *embedding_dim; ++j) {
//                (embeddings)[i][j] = cpp_results[i][j];  // 复制数据
//            }
//        }
//
//        return HELICON_SUCCESS;
//
//    }
//    catch (const std::exception& e) {
//        std::cout << "[HeliconSDK] CLIP text encoding exception: " << e.what() << std::endl;
//        return HELICON_ERROR_MODULE_ERROR;
//    }
//}

HELICON_API Helicon_ErrorCode Helicon_CLIP_EncodeText(
    const char** text_list,
    uint32_t text_count,
    float* embeddings,           // 外部分配的连续内存
    uint32_t embeddings_size,    // 外部分配的内存大小
    uint32_t* embedding_count,   // 返回实际的embedding数量
    uint32_t* embedding_dim) {   // 返回embedding维度

    if (!text_list || !embeddings || !embedding_count || !embedding_dim || text_count == 0) {
        return HELICON_ERROR_INVALID_PARAM;
    }

    try {
        // 转换输入格式
        std::vector<std::string> cpp_texts;
        for (uint32_t i = 0; i < text_count; ++i) {
            if (text_list[i]) {
                cpp_texts.push_back(text_list[i]);
            }
        }

        if (cpp_texts.empty()) {
            return HELICON_ERROR_INVALID_PARAM;
        }

        // 调用推理
        std::vector<std::vector<float>> cpp_results =
            MeetingManager::GetInstance().CLIP_EncodeText(cpp_texts);

        if (cpp_results.empty()) {
            return HELICON_ERROR_MODULE_ERROR;
        }

        // 检查输出参数
        *embedding_count = static_cast<uint32_t>(cpp_results.size());
        *embedding_dim = cpp_results[0].empty() ? 0 : static_cast<uint32_t>(cpp_results[0].size());

        if (*embedding_dim == 0) {
            return HELICON_ERROR_MODULE_ERROR;
        }

        // 检查外部分配的内存是否足够
        uint32_t required_size = (*embedding_count) * (*embedding_dim);
        if (embeddings_size < required_size) {
            return HELICON_ERROR_BUFFER_TOO_SMALL;
        }

        // 复制数据到外部分配的连续内存
        for (uint32_t i = 0; i < *embedding_count; ++i) {
            for (uint32_t j = 0; j < *embedding_dim; ++j) {
                embeddings[i * (*embedding_dim) + j] = cpp_results[i][j];
            }
        }

        return HELICON_SUCCESS;

    }
    catch (const std::exception& e) {
        std::cout << "[HeliconSDK] CLIP text encoding exception: " << e.what() << std::endl;
        return HELICON_ERROR_MODULE_ERROR;
    }
}


HELICON_API Helicon_ErrorCode Helicon_CLIP_EncodeImage(
    const char** image_paths,
    uint32_t image_count,
    float** embeddings,
    uint32_t* embedding_count,
    uint32_t* embedding_dim) {

    if (!image_paths || !embeddings || !embedding_count || !embedding_dim || image_count == 0) {
        return HELICON_ERROR_INVALID_PARAM;
    }

    try {
        // 转换输入格式
        std::vector<std::string> cpp_paths;
        for (uint32_t i = 0; i < image_count; ++i) {
            if (image_paths[i]) {
                cpp_paths.push_back(image_paths[i]);
            }
        }

        if (cpp_paths.empty()) {
            return HELICON_ERROR_INVALID_PARAM;
        }

        // 调用会议管理器
        std::vector<std::vector<float>> cpp_results =
            MeetingManager::GetInstance().CLIP_EncodeImage(cpp_paths);

        if (cpp_results.empty()) {
            return HELICON_ERROR_MODULE_ERROR;
        }

        // 转换输出格式
        *embedding_count = static_cast<uint32_t>(cpp_results.size());
        *embedding_dim = cpp_results[0].empty() ? 0 : static_cast<uint32_t>(cpp_results[0].size());

        if (*embedding_dim == 0) {
            return HELICON_ERROR_MODULE_ERROR;
        }

        // 分配内存
        embeddings = new float* [*embedding_count];
        for (uint32_t i = 0; i < *embedding_count; ++i) {
            (embeddings)[i] = new float[*embedding_dim];
            for (uint32_t j = 0; j < *embedding_dim; ++j) {
                (embeddings)[i][j] = cpp_results[i][j];
            }
        }

        return HELICON_SUCCESS;

    }
    catch (const std::exception& e) {
        std::cout << "[HeliconSDK] CLIP image encoding exception: " << e.what() << std::endl;
        return HELICON_ERROR_MODULE_ERROR;
    }
}

HELICON_API float Helicon_CLIP_CosineSimilarity(
    const float* text_embedding,
    const float* image_embedding,
    uint32_t embedding_dim) {

    if (!text_embedding || !image_embedding || embedding_dim == 0) {
        return 0.0f;
    }

    try {
        // 转换为vector格式
        std::vector<float> text_vec(text_embedding, text_embedding + embedding_dim);
        std::vector<float> image_vec(image_embedding, image_embedding + embedding_dim);

        // 调用会议管理器
        return MeetingManager::GetInstance().CLIP_CosineSimilarity(text_vec, image_vec);

    }
    catch (const std::exception& e) {
        std::cout << "[HeliconSDK] CLIP similarity calculation exception: " << e.what() << std::endl;
        return 0.0f;
    }
}

HELICON_API void Helicon_FreeEmbeddings(float** embeddings, uint32_t count) {
    if (!embeddings) {
        return;
    }

    for (uint32_t i = 0; i < count; ++i) {
        delete[] embeddings[i];
    }
    delete[] embeddings;
}

} // extern "C"