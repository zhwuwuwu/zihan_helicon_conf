#ifndef HELICON_SEARCH_SDK_H
#define HELICON_SEARCH_SDK_H

#include <stdint.h>

#ifdef _WIN32
#ifdef HELICON_SDK_EXPORTS
#define HELICON_API __declspec(dllexport)
#else
#define HELICON_API __declspec(dllimport)
#endif
#else
#define HELICON_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

    // SDK版本信息
#define HELICON_SDK_VERSION_MAJOR 1
#define HELICON_SDK_VERSION_MINOR 0
#define HELICON_SDK_VERSION_PATCH 0

// 错误代码定义
    typedef enum {
        HELICON_SUCCESS = 0,
        HELICON_ERROR_INVALID_PARAM = -1,
        HELICON_ERROR_INIT_FAILED = -2,
        HELICON_ERROR_NOT_INITIALIZED = -3,
        HELICON_ERROR_MODULE_ERROR = -4,
        HELICON_ERROR_BUFFER_TOO_SMALL = -5,
        HELICON_ERROR_UNKNOWN = -999
    } Helicon_ErrorCode;

    // 图片输入结构
    typedef struct {
        const char* path;
    } Helicon_ImageInput;

    // 推理结果结构
    typedef struct {
        char* text;
        uint32_t length;
    } Helicon_InferenceResult;

    // === 基础SDK接口 ===
    HELICON_API Helicon_ErrorCode Helicon_Initialize(const char* config_path);
    HELICON_API void Helicon_Cleanup();
    HELICON_API const char* Helicon_GetVersion();
    HELICON_API int Helicon_IsInitialized();

    // === VLM模块接口 ===
    // 图片推理接口
    HELICON_API Helicon_ErrorCode Helicon_VLM_InferenceImages(
        const Helicon_ImageInput* input_list,
        uint32_t input_count,
        const char* prompt,
        Helicon_InferenceResult* results,
        uint32_t* result_count
    );

    // 纯文本LLM推理接口
    HELICON_API Helicon_ErrorCode Helicon_VLM_InferenceText(
        const char* input_text,
        char* output_text,
        uint32_t* output_length,
        uint32_t max_output_length
    );

    // 释放推理结果内存
    HELICON_API void Helicon_FreeResults(Helicon_InferenceResult* results, uint32_t count);

    // 获取图片文件路径（工具函数）
    HELICON_API Helicon_ErrorCode Helicon_GetImagePaths(
        const char* directory,
        char*** image_paths,
        uint32_t* path_count
    );

    // 释放图片路径内存
    HELICON_API void Helicon_FreeImagePaths(char** image_paths, uint32_t count);

    // === ChineseCLIP模块接口 ===
// 文本编码接口
    /*HELICON_API Helicon_ErrorCode Helicon_CLIP_EncodeText(
        const char** text_list,
        uint32_t text_count,
        float** embeddings,
        uint32_t* embedding_count,
        uint32_t* embedding_dim
    );*/
    HELICON_API Helicon_ErrorCode Helicon_CLIP_EncodeText(
        const char** text_list,
        uint32_t text_count,
        float* embeddings,           // 外部分配的连续内存
        uint32_t embeddings_size,    // 外部分配的内存大小
        uint32_t* embedding_count,   // 返回实际的embedding数量
        uint32_t* embedding_dim);

    // 图片编码接口
    HELICON_API Helicon_ErrorCode Helicon_CLIP_EncodeImage(
        const char** image_paths,
        uint32_t image_count,
        float** embeddings,
        uint32_t* embedding_count,
        uint32_t* embedding_dim
    );

    // 计算余弦相似度
    HELICON_API float Helicon_CLIP_CosineSimilarity(
        const float* text_embedding,
        const float* image_embedding,
        uint32_t embedding_dim
    );

    // 释放嵌入向量内存
    HELICON_API void Helicon_FreeEmbeddings(float** embeddings, uint32_t count);

    // === 未来其他模块接口可以在这里添加 ===
    // HELICON_API Helicon_ErrorCode Helicon_ASR_ProcessAudio(...);
    // HELICON_API Helicon_ErrorCode Helicon_NLP_ProcessText(...);

#ifdef __cplusplus
}
#endif

#endif // HELICON_SEARCH_SDK_H