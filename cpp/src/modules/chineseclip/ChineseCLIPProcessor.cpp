#include "ChineseCLIPProcessor.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <cmath>

// SentencePiece头文件
#include <../third_party/sentencepiece/include/sentencepiece_processor.h>

// OpenVINO头文件
#include <openvino/openvino.hpp>

// 图像处理简化版
#define STB_IMAGE_IMPLEMENTATION
#include "../third_party/stb/stb_image.h"

namespace fs = std::filesystem;

// 静态常量定义（类外初始化）
const std::string ChineseCLIPProcessor::TEXT_INPUT_IDS_NAME = "input_ids";
const std::string ChineseCLIPProcessor::TEXT_ATTENTION_MASK_NAME = "attention_mask";
const std::string ChineseCLIPProcessor::IMAGE_INPUT_NAME = "pixel_values";


ChineseCLIPProcessor::ChineseCLIPProcessor(const std::string& model_path, const std::string& device)
    : model_path_(model_path)
    , device_(device)
    , initialized_(false)
    , tokenizer_(nullptr)
    , core_(nullptr)
    , text_encoder_(nullptr)
    , image_encoder_(nullptr) {
    
    try {
        std::cout << "[ChineseCLIP] Initializing ChineseCLIP processor..." << std::endl;
        std::cout << "[ChineseCLIP] Model path: " << model_path_ << std::endl;
        std::cout << "[ChineseCLIP] Device: " << device_ << std::endl;

        // 设置分词器模型路径
        tokenizer_model_path_ = model_path_ + "/tokenizer.model";

        // 初始化分词器
        if (!InitializeTokenizer()) {
            std::cout << "[ChineseCLIP] Failed to initialize tokenizer" << std::endl;
            return;
        }
        
        // 初始化OpenVINO模型
        if (InitializeModels()) {
            initialized_ = true;
            std::cout << "[ChineseCLIP] ChineseCLIP processor initialized successfully" << std::endl;
        } else {
            std::cout << "[ChineseCLIP] Failed to initialize ChineseCLIP processor" << std::endl;
        }
        
    } catch (const std::exception& e) {
        last_error_ = "Initialization failed: " + std::string(e.what());
        std::cout << "[ChineseCLIP] " << last_error_ << std::endl;
    }
}

ChineseCLIPProcessor::~ChineseCLIPProcessor() {
    std::cout << "[ChineseCLIP] Cleaning up ChineseCLIP processor" << std::endl;
    // 先释放模型资源
    ReleaseModels();
    // 再释放分词器
    ReleaseTokenizer();
}

bool ChineseCLIPProcessor::InitializeTokenizer() {
    try {
        // 检查分词器模型文件是否存在
        if (!fs::exists(tokenizer_model_path_)) {
            last_error_ = "Tokenizer model not found: " + tokenizer_model_path_;
            std::cout << "[ChineseCLIP] " << last_error_ << std::endl;
            return false;
        }

        // 创建SentencePiece处理器
        tokenizer_ = std::make_unique<sentencepiece::SentencePieceProcessor>();

        // 加载模型
        const auto status = tokenizer_->Load(tokenizer_model_path_);
        if (!status.ok()) {
            last_error_ = "Failed to load tokenizer model: " + status.ToString();
            std::cout << "[ChineseCLIP] " << last_error_ << std::endl;
            tokenizer_.reset();
            return false;
        }

        std::cout << "[ChineseCLIP] Tokenizer initialized successfully" << std::endl;
        std::cout << "[ChineseCLIP] Vocabulary size: " << tokenizer_->GetPieceSize() << std::endl;

        return true;

    }
    catch (const std::exception& e) {
        last_error_ = "Tokenizer initialization failed: " + std::string(e.what());
        std::cout << "[ChineseCLIP] " << last_error_ << std::endl;
        return false;
    }
}

bool ChineseCLIPProcessor::ReleaseTokenizer() {
    if (tokenizer_) {
        tokenizer_.reset();
        return true;
    }
    return false;
}

bool ChineseCLIPProcessor::InitializeModels() {
    try {
        // 检查模型文件是否存在
        std::string text_encoder_path = model_path_ + "/text_encoder.xml";
        std::string image_encoder_path = model_path_ + "/image_encoder.xml";
        
        if (!fs::exists(text_encoder_path)) {
            last_error_ = "Text encoder model not found: " + text_encoder_path;
            return false;
        }
        
        if (!fs::exists(image_encoder_path)) {
            last_error_ = "Image encoder model not found: " + image_encoder_path;
            return false;
        }
        
        // 初始化OpenVINO Core
        core_ = std::make_unique<ov::Core>();
        
        // 检查设备是否可用
        auto available_devices = core_->get_available_devices();
        bool device_available = false;
        for (const auto& dev : available_devices) {
            if (dev.find(device_) != std::string::npos) {
                device_available = true;
                break;
            }
        }
        
        if (!device_available) {
            std::cout << "[ChineseCLIP] Warning: Device " << device_ << " not available, falling back to CPU" << std::endl;
            device_ = "CPU";
        }
        
        // 加载并编译文本编码器
        std::cout << "[ChineseCLIP] Loading text encoder..." << std::endl;
        auto text_model = core_->read_model(text_encoder_path);
        text_encoder_ = std::make_unique<ov::CompiledModel>(core_->compile_model(text_model, device_));

        // 验证文本输入端口
        auto text_inputs = text_encoder_->inputs();
        bool has_input_ids = false, has_mask = false;
        for (const auto& input : text_inputs) {
            std::string name = input.get_any_name();
            std::cout << name << std::endl;
            if (name == TEXT_INPUT_IDS_NAME) has_input_ids = true;
            if (name == TEXT_ATTENTION_MASK_NAME) has_mask = true;
        }
        if (!has_input_ids || !has_mask) {
            last_error_ = "Text encoder input mismatch (need input_ids + attention_mask)";
            std::cout << "Text encoder input mismatch (need input_ids + attention_mask)" << std::endl;
            //return false;
        }
        
        // 加载并编译图像编码器
        std::cout << "[ChineseCLIP] Loading image encoder..." << std::endl;
        auto image_model = core_->read_model(image_encoder_path);
        image_encoder_ = std::make_unique<ov::CompiledModel>(core_->compile_model(image_model, device_));

        // 验证图像输入端口
        auto image_inputs = image_encoder_->inputs();
        bool has_image_input = false;
        for (const auto& input : image_inputs) {
            std::cout << input.get_any_name() << std::endl;
            if (input.get_any_name() == IMAGE_INPUT_NAME) has_image_input = true;
        }
        if (!has_image_input) {
            last_error_ = "Image encoder input mismatch (need pixel_values)";
            //return false;
        }
        
        std::cout << "[ChineseCLIP] Models loaded successfully on device: " << device_ << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        last_error_ = "Model initialization failed: " + std::string(e.what());
        return false;
    }
}

bool ChineseCLIPProcessor::ReleaseModels() {
    try {
        // 1. 释放文本编码器（先释放依赖对象）
        if (text_encoder_) {
            text_encoder_.reset();  // 销毁 CompiledModel，释放设备内存/编译资源
            std::cout << "[ChineseCLIP] Text encoder released successfully" << std::endl;
        }

        // 2. 释放图像编码器
        if (image_encoder_) {
            image_encoder_.reset();  // 同上
            std::cout << "[ChineseCLIP] Image encoder released successfully" << std::endl;
        }

        // 3. 释放 OpenVINO Core（最后释放核心对象）
        if (core_) {
            core_.reset();  // 释放 OpenVINO 运行时、插件资源（如 GPU 驱动连接）
            std::cout << "[ChineseCLIP] OpenVINO Core released successfully" << std::endl;
        }

        // 4. 同步状态：标记为未初始化
        initialized_ = false;
        last_error_ = "Models released successfully";

        return true;
    }
    catch (const std::exception& e) {
        last_error_ = "Failed to release models: " + std::string(e.what());
        std::cout << "[ChineseCLIP] " << last_error_ << std::endl;
        return false;
    }
}

bool ChineseCLIPProcessor::TokenizeText(const std::string& text,
    std::vector<int32_t>& input_ids,
    std::vector<int32_t>& attention_mask) {
    if (!tokenizer_) {
        return false;
    }

    try {
        // 使用SentencePiece进行分词
        std::vector<int> piece_ids;
        tokenizer_->Encode(text, &piece_ids);

        // 初始化输出向量
        input_ids.clear();
        attention_mask.clear();
        input_ids.resize(MAX_TEXT_LENGTH, PAD_TOKEN_ID);
        attention_mask.resize(MAX_TEXT_LENGTH, 0);

        // 添加[CLS]标记
        input_ids[0] = CLS_TOKEN_ID;
        attention_mask[0] = 1;

        // 复制分词结果，确保不超过最大长度
        size_t copy_length = std::min(piece_ids.size(), static_cast<size_t>(MAX_TEXT_LENGTH - 2));
        for (size_t i = 0; i < copy_length; ++i) {
            input_ids[i + 1] = static_cast<int32_t>(piece_ids[i]);
            attention_mask[i + 1] = 1;
        }

        // 添加[SEP]标记
        if (copy_length + 1 < MAX_TEXT_LENGTH) {
            input_ids[copy_length + 1] = SEP_TOKEN_ID;
            attention_mask[copy_length + 1] = 1;
        }

        return true;

    }
    catch (const std::exception& e) {
        last_error_ = "Tokenization failed: " + std::string(e.what());
        return false;
    }
}

bool ChineseCLIPProcessor::TokenizeTexts(const std::vector<std::string>& texts,
    std::vector<std::vector<int32_t>>& input_ids_batch,
    std::vector<std::vector<int32_t>>& attention_masks_batch) {
    input_ids_batch.clear();
    attention_masks_batch.clear();

    for (const auto& text : texts) {
        std::vector<int32_t> input_ids;
        std::vector<int32_t> attention_mask;

        if (!TokenizeText(text, input_ids, attention_mask)) {
            return false;
        }

        input_ids_batch.push_back(input_ids);
        attention_masks_batch.push_back(attention_mask);
    }

    return true;
}


std::vector<std::vector<float>> ChineseCLIPProcessor::EncodeText(const std::vector<std::string>& text_list) {
    if (!initialized_ || !text_encoder_ || !tokenizer_) {
        last_error_ = "Text encoder or tokenizer not initialized";
        std::cout << "[ChineseCLIP] " << last_error_ << std::endl;
        return {};
    }

    if (text_list.empty()) {
        return {};
    }

    try {
        // 使用SentencePiece进行分词
        std::vector<std::vector<int32_t>> input_ids_batch;
        std::vector<std::vector<int32_t>> attention_masks_batch;

        if (!TokenizeTexts(text_list, input_ids_batch, attention_masks_batch)) {
            return {};
        }

        std::vector<std::vector<float>> results;

        // 批量处理文本
        for (size_t i = 0; i < text_list.size(); ++i) {
            try {
                // 创建推理请求
                auto infer_request = text_encoder_->create_infer_request();

                //// 设置输入
                //auto input_ids_tensor = ov::Tensor(ov::element::i32, { 1, MAX_TEXT_LENGTH });
                //auto attention_mask_tensor = ov::Tensor(ov::element::i32, { 1, MAX_TEXT_LENGTH });

                //std::copy(input_ids_batch[i].begin(), input_ids_batch[i].end(),
                //    input_ids_tensor.data<int32_t>());
                //std::copy(attention_masks_batch[i].begin(), attention_masks_batch[i].end(),
                //    attention_mask_tensor.data<int32_t>());

                //infer_request.set_input_tensor(0, input_ids_tensor);
                //infer_request.set_input_tensor(1, attention_mask_tensor);


                //test
                // 获取输入端口（按名称绑定，避免索引错误）
                auto input_ids_port = text_encoder_->input(TEXT_INPUT_IDS_NAME);
                auto mask_port = text_encoder_->input(TEXT_ATTENTION_MASK_NAME);

                // 构造输入Tensor
                ov::Tensor input_ids_tensor(input_ids_port.get_element_type(), input_ids_port.get_shape());
                ov::Tensor mask_tensor(mask_port.get_element_type(), mask_port.get_shape());

                // 复制数据（确保长度匹配）
                size_t copy_len = std::min(input_ids_batch[i].size(), static_cast<size_t>(MAX_TEXT_LENGTH));
                std::copy_n(input_ids_batch[i].begin(), copy_len, input_ids_tensor.data<int32_t>());
                std::copy_n(attention_masks_batch[i].begin(), copy_len, mask_tensor.data<int32_t>());

                // 设置输入并执行推理
                infer_request.set_input_tensor(0, input_ids_tensor);
                infer_request.set_input_tensor(1, mask_tensor);
                //end



                // 执行推理
                infer_request.infer();

                // 获取输出
                auto output_tensor = infer_request.get_output_tensor();
                const float* output_data = output_tensor.data<float>();

                // 提取嵌入向量
                std::vector<float> embedding(output_data, output_data + EMBEDDING_DIM);

                // 归一化
                NormalizeVector(embedding);

                results.push_back(embedding);

                std::cout << "[ChineseCLIP] Text " << i << " encoded successfully" << std::endl;

            }
            catch (const std::exception& e) {
                std::cout << "[ChineseCLIP] Text encoding error for item " << i << ": " << e.what() << std::endl;
                results.push_back(std::vector<float>(EMBEDDING_DIM, 0.0f));
            }
        }

        return results;

    }
    catch (const std::exception& e) {
        last_error_ = "Text encoding failed: " + std::string(e.what());
        std::cout << "[ChineseCLIP] " << last_error_ << std::endl;
        return {};
    }
}

std::vector<std::vector<float>> ChineseCLIPProcessor::EncodeImage(const std::vector<std::string>& image_paths) {
    if (!initialized_ || !image_encoder_) {
        last_error_ = "Image encoder not initialized";
        std::cout << "[ChineseCLIP] " << last_error_ << std::endl;
        return {};
    }
    
    if (image_paths.empty()) {
        return {};
    }
    
    try {
        std::vector<std::vector<float>> all_embeddings;
        
        // 处理每张图片
        for (size_t i = 0; i < image_paths.size(); ++i) {
            try {
                // 预处理图片
                std::vector<float> pixel_values;
                if (!PreprocessImage(image_paths[i], pixel_values)) {
                    std::cout << "[ChineseCLIP] Failed to preprocess image: " << image_paths[i] << std::endl;
                    all_embeddings.push_back(std::vector<float>(EMBEDDING_DIM, 0.0f));
                    continue;
                }
                
                // 创建推理请求
                auto infer_request = image_encoder_->create_infer_request();
                
                // 设置输入
                auto input_tensor = ov::Tensor(ov::element::f32, {1, 3, IMAGE_SIZE, IMAGE_SIZE});
                std::copy(pixel_values.begin(), pixel_values.end(), input_tensor.data<float>());
                
                infer_request.set_input_tensor(input_tensor);
                
                // 执行推理
                infer_request.infer();
                
                // 获取输出
                auto output_tensor = infer_request.get_output_tensor();
                const float* output_data = output_tensor.data<float>();
                
                // 提取嵌入向量
                std::vector<float> embedding(output_data, output_data + EMBEDDING_DIM);
                
                // 归一化
                NormalizeVector(embedding);
                
                all_embeddings.push_back(embedding);
                
            } catch (const std::exception& e) {
                std::cout << "[ChineseCLIP] Image encoding error for " << image_paths[i] << ": " << e.what() << std::endl;
                all_embeddings.push_back(std::vector<float>(EMBEDDING_DIM, 0.0f));
            }
        }
        
        // 如果只有一张图片，直接返回
        if (all_embeddings.size() == 1) {
            return all_embeddings;
        }
        
        // 多张图片时计算平均嵌入向量
        std::vector<float> avg_embedding = ComputeAverageEmbedding(all_embeddings);
        return {avg_embedding};
        
    } catch (const std::exception& e) {
        last_error_ = "Image encoding failed: " + std::string(e.what());
        std::cout << "[ChineseCLIP] " << last_error_ << std::endl;
        return {};
    }
}

float ChineseCLIPProcessor::CosineSimilarity(const std::vector<float>& text_vec, const std::vector<float>& img_vec) {
    if (text_vec.size() != img_vec.size() || text_vec.empty()) {
        return 0.0f;
    }
    
    try {
        float dot_product = std::inner_product(text_vec.begin(), text_vec.end(), img_vec.begin(), 0.0f);
        
        float norm_text = std::sqrt(std::inner_product(text_vec.begin(), text_vec.end(), text_vec.begin(), 0.0f));
        float norm_img = std::sqrt(std::inner_product(img_vec.begin(), img_vec.end(), img_vec.begin(), 0.0f));
        
        if (norm_text == 0.0f || norm_img == 0.0f) {
            return 0.0f;
        }
        
        return dot_product / (norm_text * norm_img);
        
    } catch (const std::exception& e) {
        last_error_ = "Similarity calculation failed: " + std::string(e.what());
        return 0.0f;
    }
}

bool ChineseCLIPProcessor::PreprocessText(const std::vector<std::string>& texts,
                                        std::vector<std::vector<int32_t>>& input_ids,
                                        std::vector<std::vector<int32_t>>& attention_masks) {
    // 简化的文本预处理（实际项目中需要使用真正的tokenizer）
    // 这里只是示例，实际需要集成ChineseCLIP的tokenizer
    
    input_ids.clear();
    attention_masks.clear();
    
    for (const auto& text : texts) {
        // 简化的tokenization（实际需要使用ChineseCLIP tokenizer）
        std::vector<int32_t> ids(MAX_TEXT_LENGTH, 0);  // 填充为0
        std::vector<int32_t> mask(MAX_TEXT_LENGTH, 0); // 填充为0
        
        // 这里应该实现真正的tokenization逻辑
        // 暂时使用简化版本
        ids[0] = 101;  // [CLS] token
        mask[0] = 1;
        
        // 简单的字符级tokenization（仅用于演示）
        size_t text_len = std::min(text.length(), static_cast<size_t>(MAX_TEXT_LENGTH - 2));
        for (size_t i = 0; i < text_len; ++i) {
            ids[i + 1] = static_cast<int32_t>(text[i]) % 30000 + 1000;  // 简化映射
            mask[i + 1] = 1;
        }
        
        ids[text_len + 1] = 102;  // [SEP] token
        mask[text_len + 1] = 1;
        
        input_ids.push_back(ids);
        attention_masks.push_back(mask);
    }
    
    return true;
}
//
//bool ChineseCLIPProcessor::PreprocessImage(const std::string& image_path, std::vector<float>& pixel_values) {
//    try {
//        // 使用stb_image加载图片
//        int width, height, channels;
//        unsigned char* img_data = stbi_load(image_path.c_str(), &width, &height, &channels, 3);
//        
//        if (!img_data) {
//            last_error_ = "Failed to load image: " + image_path;
//            return false;
//        }
//        
//        // 调整图片大小到224x224（简化版resize）
//        pixel_values.resize(3 * IMAGE_SIZE * IMAGE_SIZE);
//        
//        // 简化的resize和预处理
//        for (int c = 0; c < 3; ++c) {
//            for (int h = 0; h < IMAGE_SIZE; ++h) {
//                for (int w = 0; w < IMAGE_SIZE; ++w) {
//                    // 简单的最近邻插值
//                    int src_h = (h * height) / IMAGE_SIZE;
//                    int src_w = (w * width) / IMAGE_SIZE;
//                    
//                    src_h = std::min(src_h, height - 1);
//                    src_w = std::min(src_w, width - 1);
//                    
//                    float pixel = static_cast<float>(img_data[src_h * width * 3 + src_w * 3 + c]) / 255.0f;
//                    
//                    // 归一化
//                    if (c == 0) pixel = (pixel - MEAN_R) / STD_R;
//                    else if (c == 1) pixel = (pixel - MEAN_G) / STD_G;
//                    else pixel = (pixel - MEAN_B) / STD_B;
//                    
//                    pixel_values[c * IMAGE_SIZE * IMAGE_SIZE + h * IMAGE_SIZE + w] = pixel;
//                }
//            }
//        }
//        
//        stbi_image_free(img_data);
//        return true;
//        
//    } catch (const std::exception& e) {
//        last_error_ = "Image preprocessing failed: " + std::string(e.what());
//        return false;
//    }
//}

bool ChineseCLIPProcessor::PreprocessImage(const std::string& image_path, std::vector<float>& pixel_values) {
    try {
        // 1. 加载图片（强制转为3通道RGB）
        int src_w, src_h, src_channels;
        unsigned char* img_data = stbi_load(image_path.c_str(), &src_w, &src_h, &src_channels, 3);
        if (!img_data) {
            last_error_ = "Load image failed: " + image_path;
            return false;
        }

        // 2. 初始化输出（3x224x224，CHW格式）
        pixel_values.resize(3 * IMAGE_SIZE * IMAGE_SIZE, 0.0f);

        // 3. 双线性插值缩放
        const float scale_w = static_cast<float>(src_w) / IMAGE_SIZE;
        const float scale_h = static_cast<float>(src_h) / IMAGE_SIZE;

        for (int c = 0; c < 3; ++c) { // RGB通道
            for (int dst_h = 0; dst_h < IMAGE_SIZE; ++dst_h) {
                for (int dst_w = 0; dst_w < IMAGE_SIZE; ++dst_w) {
                    // 计算源图像坐标
                    float src_x = dst_w * scale_w;
                    float src_y = dst_h * scale_h;
                    int x0 = static_cast<int>(std::floor(src_x));
                    int y0 = static_cast<int>(std::floor(src_y));
                    int x1 = std::min(x0 + 1, src_w - 1);
                    int y1 = std::min(y0 + 1, src_h - 1);

                    // 双线性插值权重
                    float wx = src_x - x0;
                    float wy = src_y - y0;

                    // 读取4个邻域像素并归一化到[0,1]
                    float p00 = static_cast<float>(img_data[y0 * src_w * 3 + x0 * 3 + c]) / 255.0f;
                    float p01 = static_cast<float>(img_data[y1 * src_w * 3 + x0 * 3 + c]) / 255.0f;
                    float p10 = static_cast<float>(img_data[y0 * src_w * 3 + x1 * 3 + c]) / 255.0f;
                    float p11 = static_cast<float>(img_data[y1 * src_w * 3 + x1 * 3 + c]) / 255.0f;

                    // 双线性插值计算
                    float pixel = (1 - wx) * (1 - wy) * p00 +
                        (1 - wx) * wy * p01 +
                        wx * (1 - wy) * p10 +
                        wx * wy * p11;

                    // 应用均值和标准差归一化
                    if (c == 0) pixel = (pixel - MEAN_R) / STD_R; // R通道
                    else if (c == 1) pixel = (pixel - MEAN_G) / STD_G; // G通道
                    else pixel = (pixel - MEAN_B) / STD_B; // B通道

                    // 存储到CHW格式中
                    size_t idx = c * IMAGE_SIZE * IMAGE_SIZE + dst_h * IMAGE_SIZE + dst_w;
                    pixel_values[idx] = pixel;
                }
            }
        }

        // 释放资源
        stbi_image_free(img_data);
        return true;
    }
    catch (const std::exception& e) {
        last_error_ = "Image preprocess failed: " + std::string(e.what());
        return false;
    }
}


void ChineseCLIPProcessor::NormalizeVector(std::vector<float>& vec) {
    float norm = std::sqrt(std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0f));
    if (norm > 0.0f) {
        for (float& val : vec) {
            val /= norm;
        }
    }
}

std::vector<float> ChineseCLIPProcessor::ComputeAverageEmbedding(const std::vector<std::vector<float>>& embeddings) {
    if (embeddings.empty()) {
        return std::vector<float>(EMBEDDING_DIM, 0.0f);
    }
    
    std::vector<float> avg_embedding(EMBEDDING_DIM, 0.0f);
    
    for (const auto& embedding : embeddings) {
        for (size_t i = 0; i < EMBEDDING_DIM && i < embedding.size(); ++i) {
            avg_embedding[i] += embedding[i];
        }
    }
    
    for (float& val : avg_embedding) {
        val /= static_cast<float>(embeddings.size());
    }
    
    NormalizeVector(avg_embedding);
    return avg_embedding;
}