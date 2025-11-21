#include <iostream>
#include <windows.h>
#include "HeliconSearchSDK.h"


int main() {
    std::cout << "=== Helicon Search SDK Test ===" << std::endl;

    // 测试获取版本信息
    const char* version = Helicon_GetVersion();
    std::cout << "SDK Version: " << version << std::endl;

    // 测试基础SDK功能
    std::cout << "\n=== Testing Basic SDK Functions ===" << std::endl;
    Helicon_ErrorCode result = Helicon_Initialize("./config.ini");

    if (result == HELICON_SUCCESS) {
        std::cout << "+ SDK initialization successful!" << std::endl;

        // 检查初始化状态
        if (Helicon_IsInitialized()) {
            std::cout << "+ SDK is properly initialized" << std::endl;
        }

        // 测试VLM功能
        std::cout << "\n=== Testing VLM Module via SDK Interface ===" << std::endl;

        // 测试文本推理
        char output_text[1024];
        uint32_t output_length = 0;
        result = Helicon_VLM_InferenceText(
            "Please summarize AI development",
            output_text,
            &output_length,
            sizeof(output_text)
        );

        if (result == HELICON_SUCCESS && output_length > 0) {
            std::cout << "+ VLM text inference successful" << std::endl;
            std::cout << "  Output: " << std::string(output_text, min(output_length, 100u)) << "..." << std::endl;
        }
        else {
            std::cout << "- VLM text inference not available (expected without model files)" << std::endl;
        }

        // 测试获取图片路径
        char** image_paths = nullptr;
        uint32_t path_count = 0;
        result = Helicon_GetImagePaths("./", &image_paths, &path_count);

        if (result == HELICON_SUCCESS) {
            std::cout << "+ Found " << path_count << " image files in current directory" << std::endl;
            Helicon_FreeImagePaths(image_paths, path_count);
        }

        // 清理SDK
        std::cout << "\nCleaning up SDK..." << std::endl;
        Helicon_Cleanup();
        std::cout << "+ SDK cleanup completed" << std::endl;

    }
    else {
        std::cout << "- SDK initialization failed with error code: " << result << std::endl;
    }

    std::cout << "\n=== All Tests Completed ===" << std::endl;

    // 等待用户输入，方便查看输出
    std::cout << "Press Enter to exit...";
    std::cin.get();

    return 0;
}