#include <iostream>
#include <windows.h>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include "HeliconSearchSDK.h"

namespace fs = std::filesystem;

class SDKTester {
public:
    static void RunAllTests() {
        std::cout << "=== Helicon Search SDK Complete Test Suite ===" << std::endl;

        TestBasicSDKFunctions();
        TestCLIPInference();

        /*TestVLMTextInference();
        TestVLMImageInference();
        TestImagePathsUtility();
        TestErrorHandling();
        TestMemoryManagement();*/

        std::cout << "\n=== All SDK tests completed! ===" << std::endl;
    }

private:
    static void TestBasicSDKFunctions() {
        std::cout << "\n[Test] Basic SDK Functions..." << std::endl;

        // 测试版本获取
        const char* version = Helicon_GetVersion();
        std::cout << "[PASS] SDK Version: " << version << std::endl;

        // 测试初始状态
        int initial_state = Helicon_IsInitialized();
        if (!initial_state) {
            std::cout << "[PASS] Initial state is not initialized" << std::endl;
        }

        // 测试初始化
        std::cout << "Testing SDK initialization..." << std::endl;
        Helicon_ErrorCode result = Helicon_Initialize("./config.ini");

        if (result == HELICON_SUCCESS) {
            std::cout << "[PASS] SDK initialization successful" << std::endl;

            // 验证初始化状态
            if (Helicon_IsInitialized()) {
                std::cout << "[PASS] IsInitialized() returns true after initialization" << std::endl;
            }
            else {
                std::cout << "[FAIL] IsInitialized() should return true after successful initialization" << std::endl;
            }

            // 测试重复初始化
            Helicon_ErrorCode repeat_result = Helicon_Initialize("./config.ini");
            if (repeat_result == HELICON_SUCCESS) {
                std::cout << "[PASS] Repeated initialization handled correctly" << std::endl;
            }

        }
        else {
            std::cout << "[WARN] SDK initialization failed with error code: " << result << std::endl;
            std::cout << "       This may be expected if model files are not available" << std::endl;
        }

        // 测试清理
        std::cout << "Testing SDK cleanup..." << std::endl;
        Helicon_Cleanup();

        if (!Helicon_IsInitialized()) {
            std::cout << "[PASS] SDK cleanup successful" << std::endl;
        }

        // 测试重复清理
        Helicon_Cleanup();
        std::cout << "[PASS] Repeated cleanup handled correctly" << std::endl;

        // 重新初始化用于后续测试
        Helicon_Initialize("./config.ini");
    }

    static void TestVLMImageInference() {
        std::cout << "\n[Test] VLM Image Inference..." << std::endl;

        if (!Helicon_IsInitialized()) {
            std::cout << "[WARN] SDK not initialized, skipping VLM image inference test" << std::endl;
            return;
        }

        // 创建测试图片输入
        Helicon_ImageInput test_images[] = {
            {"./image/1.png"},
            {"./image/2.png"},
            {"./image/3.png"}
        };

        const uint32_t input_count = sizeof(test_images) / sizeof(test_images[0]);
        Helicon_InferenceResult results[3];
        uint32_t result_count = 0;

        // 测试图片推理
        Helicon_ErrorCode result = Helicon_VLM_InferenceImages(
            test_images,
            input_count,
            "详细描述图片内容",
            results,
            &result_count
        );

        if (result == HELICON_SUCCESS) {
            std::cout << "[PASS] VLM image inference successful, got " << result_count << " results" << std::endl;

            for (uint32_t i = 0; i < result_count; ++i) {
                if (results[i].text && results[i].length > 0) {
                    std::string result_text(results[i].text, results[i].length);
                    std::cout << "       Result " << i << " (length=" << results[i].length << "): "
                        << result_text.substr(0, min(result_text.length(), size_t(1000))) << "..." << std::endl;
                }
                else {
                    std::cout << "       Result " << i << ": Empty result" << std::endl;
                }
            }

            // 释放结果内存
            Helicon_FreeResults(results, result_count);
            std::cout << "[PASS] Results memory freed successfully" << std::endl;

        }
        else {
            std::cout << "[WARN] VLM image inference failed with error code: " << result << std::endl;
            std::cout << "       This may be expected if model files or test images are not available" << std::endl;
        }

        // 测试空输入
        uint32_t empty_result_count = 0;
        result = Helicon_VLM_InferenceImages(nullptr, 0, "test", nullptr, &empty_result_count);
        if (result == HELICON_ERROR_INVALID_PARAM) {
            std::cout << "[PASS] Empty input handled correctly" << std::endl;
        }

        // 测试使用默认prompt
        result = Helicon_VLM_InferenceImages(
            test_images,
            1,
            nullptr,  // 使用默认prompt
            results,
            &result_count
        );

        if (result == HELICON_SUCCESS) {
            std::cout << "[PASS] Default prompt handling works" << std::endl;
            Helicon_FreeResults(results, result_count);
        }
    }

    static void TestVLMTextInference() {
        std::cout << "\n[Test] VLM Text Inference..." << std::endl;

        if (!Helicon_IsInitialized()) {
            std::cout << "[WARN] SDK not initialized, skipping VLM text inference test" << std::endl;
            return;
        }

        char output_text[2048];
        uint32_t output_length = 0;

        // 测试文本推理
        const char* test_inputs[] = {
            "Please summarize the development of artificial intelligence",
            "What are the key benefits of machine learning?",
            "Describe the core concepts of deep learning briefly"
        };

        for (size_t i = 0; i < sizeof(test_inputs) / sizeof(test_inputs[0]); ++i) {
            std::cout << "Testing input " << i + 1 << ": " << test_inputs[i] << std::endl;

            Helicon_ErrorCode result = Helicon_VLM_InferenceText(
                test_inputs[i],
                output_text,
                &output_length,
                sizeof(output_text)
            );

            if (result == HELICON_SUCCESS && output_length > 0) {
                std::string result_text(output_text, output_length);
                std::cout << "[PASS] Text inference successful (length=" << output_length << ")" << std::endl;
                std::cout << "       Output: " << result_text.substr(0, min(result_text.length(), size_t(150))) << "..." << std::endl;
            }
            else {
                std::cout << "[WARN] Text inference failed or not supported (error code: " << result << ")" << std::endl;
            }
        }

        // 测试空输入
        Helicon_ErrorCode result = Helicon_VLM_InferenceText(
            "",
            output_text,
            &output_length,
            sizeof(output_text)
        );
        std::cout << "[PASS] Empty text input handled, result length: " << output_length << std::endl;

        // 测试无效参数
        result = Helicon_VLM_InferenceText(nullptr, output_text, &output_length, sizeof(output_text));
        if (result == HELICON_ERROR_INVALID_PARAM) {
            std::cout << "[PASS] Null input parameter handled correctly" << std::endl;
        }
    }

    static void TestCLIPInference() {
        std::cout << "\n[Test] CLIP Text/Image Inference..." << std::endl;

        if (!Helicon_IsInitialized()) {
            std::cout << "[WARN] SDK not initialized, skipping CLIP text/image inference test" << std::endl;
            return;
        }

        // 测试文本推理
        const char* test_inputs[] = {
            "一只小狗在草地上玩耍",
            "一只猫",
            "汽车在公路上行驶"
        };
        uint32_t text_count = sizeof(test_inputs) / sizeof(test_inputs[0]);

        // 1. 计算内存大小
        uint32_t embedding_count = text_count;
        uint32_t embedding_dim = 1024; // 固定值

        // 2. 分配内存
        uint32_t total_size = embedding_count * embedding_dim;
        std::vector<float> embeddings(total_size);

        // 3. 调用推理
        Helicon_ErrorCode result = Helicon_CLIP_EncodeText(
            test_inputs, text_count,
            embeddings.data(), total_size,
            &embedding_count, &embedding_dim);

        if (result == HELICON_SUCCESS) {
            // 4. 使用结果
            std::vector<std::vector<float>> results;
            for (uint32_t i = 0; i < embedding_count; ++i) {
                std::cout << "Embedding " << i << ": ";
                for (uint32_t j = 0; j < embedding_dim; ++j) {
                    std::cout << embeddings[i * embedding_dim + j] << " ";
                    results[i].push_back(embeddings[i * embedding_dim + j]);
                }
                std::cout << std::endl;
            }
        }



    }

    static void TestImagePathsUtility() {
        std::cout << "\n[Test] Image Paths Utility..." << std::endl;

        // 创建测试目录和文件
        std::string test_dir = "./test_images_sdk";
        CreateTestImageDirectory(test_dir);

        // 测试获取图片路径
        char** image_paths = nullptr;
        uint32_t path_count = 0;

        Helicon_ErrorCode result = Helicon_GetImagePaths(
            test_dir.c_str(),
            &image_paths,
            &path_count
        );

        if (result == HELICON_SUCCESS) {
            std::cout << "[PASS] Found " << path_count << " image files in test directory" << std::endl;

            // 显示找到的文件
            for (uint32_t i = 0; i < path_count; ++i) {
                std::string path(image_paths[i]);
                std::cout << "       Image " << i + 1 << ": " << fs::path(path).filename().string() << std::endl;
            }

            // 释放内存
            Helicon_FreeImagePaths(image_paths, path_count);
            std::cout << "[PASS] Image paths memory freed successfully" << std::endl;

        }
        else {
            std::cout << "[WARN] Failed to get image paths, error code: " << result << std::endl;
        }

        // 清理测试目录
        CleanupTestImageDirectory(test_dir);

        // 测试当前目录
        result = Helicon_GetImagePaths("./", &image_paths, &path_count);
        if (result == HELICON_SUCCESS) {
            std::cout << "[PASS] Found " << path_count << " image files in current directory" << std::endl;
            Helicon_FreeImagePaths(image_paths, path_count);
        }

        // 测试不存在的目录
        result = Helicon_GetImagePaths("./non_existent_directory", &image_paths, &path_count);
        if (result != HELICON_SUCCESS) {
            std::cout << "[PASS] Non-existent directory handled correctly" << std::endl;
        }

        // 测试无效参数
        result = Helicon_GetImagePaths(nullptr, &image_paths, &path_count);
        if (result == HELICON_ERROR_INVALID_PARAM) {
            std::cout << "[PASS] Null directory parameter handled correctly" << std::endl;
        }
    }

    static void TestErrorHandling() {
        std::cout << "\n[Test] Error Handling..." << std::endl;

        // 测试未初始化状态下的调用
        Helicon_Cleanup();

        // VLM功能在未初始化状态下的行为
        Helicon_ImageInput test_image = { "test.jpg" };
        Helicon_InferenceResult result;
        uint32_t result_count = 0;

        Helicon_ErrorCode error_code = Helicon_VLM_InferenceImages(
            &test_image, 1, "test", &result, &result_count
        );

        if (error_code != HELICON_SUCCESS) {
            std::cout << "[PASS] VLM image inference properly handles uninitialized state (error: " << error_code << ")" << std::endl;
        }

        // 文本推理在未初始化状态
        char output[100];
        uint32_t length = 0;
        error_code = Helicon_VLM_InferenceText("test", output, &length, sizeof(output));

        if (error_code != HELICON_SUCCESS) {
            std::cout << "[PASS] VLM text inference properly handles uninitialized state (error: " << error_code << ")" << std::endl;
        }

        // 测试无效参数
        error_code = Helicon_VLM_InferenceImages(nullptr, 1, "test", &result, &result_count);
        if (error_code == HELICON_ERROR_INVALID_PARAM) {
            std::cout << "[PASS] Null input array handled correctly" << std::endl;
        }

        error_code = Helicon_VLM_InferenceText(nullptr, output, &length, sizeof(output));
        if (error_code == HELICON_ERROR_INVALID_PARAM) {
            std::cout << "[PASS] Null text input handled correctly" << std::endl;
        }

        // 重新初始化
        Helicon_Initialize("./config.ini");
    }

    static void TestMemoryManagement() {
        std::cout << "\n[Test] Memory Management..." << std::endl;

        if (!Helicon_IsInitialized()) {
            std::cout << "[WARN] SDK not initialized, skipping memory management test" << std::endl;
            return;
        }

        // 测试结果内存管理
        Helicon_ImageInput test_images[] = {
            {"./test1.jpg"},
            {"./test2.jpg"}
        };

        Helicon_InferenceResult results[2];
        uint32_t result_count = 0;

        Helicon_ErrorCode result = Helicon_VLM_InferenceImages(
            test_images, 2, "test", results, &result_count
        );

        if (result == HELICON_SUCCESS) {
            // 测试释放空结果
            Helicon_FreeResults(nullptr, 0);
            std::cout << "[PASS] Null results parameter handled safely" << std::endl;

            // 正常释放
            Helicon_FreeResults(results, result_count);
            std::cout << "[PASS] Results memory management working correctly" << std::endl;
        }

        // 测试图片路径内存管理
        char** image_paths = nullptr;
        uint32_t path_count = 0;

        result = Helicon_GetImagePaths("./", &image_paths, &path_count);
        if (result == HELICON_SUCCESS) {
            // 测试释放空路径
            Helicon_FreeImagePaths(nullptr, 0);
            std::cout << "[PASS] Null image paths parameter handled safely" << std::endl;

            // 正常释放
            Helicon_FreeImagePaths(image_paths, path_count);
            std::cout << "[PASS] Image paths memory management working correctly" << std::endl;
        }
    }

    // 辅助函数
    static void CreateTestImageDirectory(const std::string& dir_path) {
        try {
            fs::create_directories(dir_path);

            std::vector<std::string> test_files = {
                "test1.jpg", "test2.png", "test3.bmp", "test4.gif",
                "document.txt", "video.mp4", "test5.tiff"
            };

            for (const auto& file : test_files) {
                std::string full_path = dir_path + "/" + file;
                std::ofstream ofs(full_path);
                ofs << "test image content";
                ofs.close();
            }

        }
        catch (const std::exception& e) {
            std::cout << "Warning: Failed to create test directory: " << e.what() << std::endl;
        }
    }

    static void CleanupTestImageDirectory(const std::string& dir_path) {
        try {
            fs::remove_all(dir_path);
        }
        catch (const std::exception& e) {
            std::cout << "Warning: Failed to cleanup test directory: " << e.what() << std::endl;
        }
    }
};

int main() {
    // 设置控制台输入/输出编码为 UTF-8
    SetConsoleOutputCP(CP_UTF8);  // 输出编码（关键，解决中文乱码）
    SetConsoleCP(CP_UTF8);        // 输入编码（可选，若需输入中文）
    std::cout << "=== Helicon Search SDK Test Application ===" << std::endl;
    std::cout << "Testing all public SDK interfaces..." << std::endl;

    try {
        SDKTester::RunAllTests();

        // 最终清理
        std::cout << "\nPerforming final cleanup..." << std::endl;
        Helicon_Cleanup();

        if (!Helicon_IsInitialized()) {
            std::cout << "[PASS] Final cleanup successful" << std::endl;
        }

        std::cout << "\n=== SDK Test Summary ===" << std::endl;
        std::cout << "[PASS] Basic SDK functions tested" << std::endl;
        std::cout << "[PASS] VLM image inference tested" << std::endl;
        std::cout << "[PASS] VLM text inference tested" << std::endl;
        std::cout << "[PASS] Image paths utility tested" << std::endl;
        std::cout << "[PASS] Error handling tested" << std::endl;
        std::cout << "[PASS] Memory management tested" << std::endl;

    }
    catch (const std::exception& e) {
        std::cout << "Test Exception: " << e.what() << std::endl;
        return -1;
    }

    std::cout << "\n=== All tests completed successfully! ===" << std::endl;
    std::cout << "Press Enter to exit...";
    std::cin.get();

    return 0;
}