#ifndef CHINESECLIP_PROCESSOR_H
#define CHINESECLIP_PROCESSOR_H

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

// SentencePiece前向声明
namespace sentencepiece {
    class SentencePieceProcessor;
}

// OpenVINO前向声明
namespace ov {
    class Core;
    class CompiledModel;
    class InferRequest;
}

// 图片输入结构
struct CLIPImageInput {
    std::string path;
};

// 文本输入结构
struct CLIPTextInput {
    std::string text;
};

// 嵌入向量结果
struct EmbeddingResult {
    std::vector<float> embedding;
    bool success;
    std::string error_message;
};

// ChineseCLIP处理器类
class ChineseCLIPProcessor {
public:
    // 构造函数
    ChineseCLIPProcessor(const std::string& model_path = "./models/zclip_model",
                        const std::string& device = "GPU");
    
    // 析构函数
    ~ChineseCLIPProcessor();
    
    // 禁止拷贝构造和赋值操作
    ChineseCLIPProcessor(const ChineseCLIPProcessor&) = delete;
    ChineseCLIPProcessor& operator=(const ChineseCLIPProcessor&) = delete;
    
    // 文本编码
    std::vector<std::vector<float>> EncodeText(const std::vector<std::string>& text_list);
    
    // 图片编码
    std::vector<std::vector<float>> EncodeImage(const std::vector<std::string>& image_paths);
    
    // 计算余弦相似度
    float CosineSimilarity(const std::vector<float>& text_vec, const std::vector<float>& img_vec);
    
    // 检查是否已初始化
    bool IsInitialized() const { return initialized_; }
    
    // 获取错误信息
    std::string GetLastError() const { return last_error_; }

private:
    // 初始化OpenVINO模型
    bool InitializeModels();

    // 卸载模型
    bool ReleaseModels();

    // 初始化SentencePiece分词器
    bool InitializeTokenizer();

    // 卸载分词器
    bool ReleaseTokenizer();

    // 使用SentencePiece进行文本分词
    bool TokenizeText(const std::string& text,
        std::vector<int32_t>& input_ids,
        std::vector<int32_t>& attention_mask);

    // 批量分词
    bool TokenizeTexts(const std::vector<std::string>& texts,
        std::vector<std::vector<int32_t>>& input_ids_batch,
        std::vector<std::vector<int32_t>>& attention_masks_batch);
    
    // 预处理文本
    bool PreprocessText(const std::vector<std::string>& texts, 
                       std::vector<std::vector<int32_t>>& input_ids,
                       std::vector<std::vector<int32_t>>& attention_masks);
    
    // 预处理图片
    bool PreprocessImage(const std::string& image_path, std::vector<float>& pixel_values);
    
    // 归一化向量
    void NormalizeVector(std::vector<float>& vec);
    
    // 计算平均嵌入向量
    std::vector<float> ComputeAverageEmbedding(const std::vector<std::vector<float>>& embeddings);
    
    // 成员变量
    std::string model_path_;
    std::string device_;
    std::string last_error_;
    bool initialized_;

    // SentencePiece分词器
    std::unique_ptr<sentencepiece::SentencePieceProcessor> tokenizer_;
    std::string tokenizer_model_path_;
    
    // OpenVINO相关
    std::unique_ptr<ov::Core> core_;
    std::unique_ptr<ov::CompiledModel> text_encoder_;
    std::unique_ptr<ov::CompiledModel> image_encoder_;
    
    // 模型参数
    static const int IMAGE_SIZE = 224;
    static const int MAX_TEXT_LENGTH = 77;
    static const int EMBEDDING_DIM = 1024;

    // 特殊token ID
    static const int32_t CLS_TOKEN_ID = 101;
    static const int32_t SEP_TOKEN_ID = 102;
    static const int32_t PAD_TOKEN_ID = 0;
    static const int32_t UNK_TOKEN_ID = 100;
    
    // 图像预处理参数
    static constexpr float MEAN_R = 0.48145466f;
    static constexpr float MEAN_G = 0.4578275f;
    static constexpr float MEAN_B = 0.40821073f;
    static constexpr float STD_R = 0.26862954f;
    static constexpr float STD_G = 0.26130258f;
    static constexpr float STD_B = 0.27577711f;

    // 模型输入名称（与转换后的IR模型一致）
    static const std::string TEXT_INPUT_IDS_NAME;    // 文本输入：input_ids
    static const std::string TEXT_ATTENTION_MASK_NAME; // 文本输入：attention_mask
    static const std::string IMAGE_INPUT_NAME;       // 图像输入：pixel_values
};

#endif // CHINESECLIP_PROCESSOR_H