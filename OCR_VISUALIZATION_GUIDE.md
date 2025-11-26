# OCR 可视化功能使用指南

## 概述

为Python和C++版本的OCR处理器添加了可视化功能，可以在图像上绘制检测框和识别结果。

---

## 🐍 Python 版本

### 功能说明

`visualize_ocr_results()` 方法会在图像上绘制：
- ✅ 绿色的文本检测框（多边形）
- ✅ 识别的文本和置信度（黑色文字，绿色背景）
- ✅ 蓝色的索引编号

### 自动调用

在 `run_paddle_ocr()` 方法中，识别完成后会自动生成可视化图像：

```python
ocr = PaddleOCRWithOpenVINO(models_dir='.\\models\\paddle_ocr', download_models=False)
text = ocr.run_paddle_ocr(image="test.png")
# 自动生成: test_ocr_result.jpg
```

### 手动调用

```python
ocr = PaddleOCRWithOpenVINO(models_dir='.\\models\\paddle_ocr')

# 运行OCR
image_path = "path/to/image.jpg"
# ... 获取 dt_boxes 和 rec_res ...

# 手动调用可视化
ocr.visualize_ocr_results(
    image_path=image_path,
    dt_boxes=dt_boxes,
    rec_res=rec_res,
    output_path="output_result.jpg"  # 可选，None则显示图像
)
```

### 输出示例

```
可视化结果已保存到: test_ocr_result.jpg
```

---

## 🔧 C++ 版本

### 功能说明

`VisualizeOCRResults()` 方法会在图像上绘制：
- ✅ 绿色的文本检测框（多边形）
- ✅ 识别的文本和置信度（黑色文字，绿色背景）
- ✅ 蓝色的索引编号

### 自动调用

在 `ProcessImage()` 方法中，识别完成后会自动生成可视化图像：

```cpp
OCRProcessor ocr("./models/paddle_ocr", "CPU");
std::string result = ocr.ProcessImage("test.png");
// 自动生成: test_ocr_result.jpg
```

### 手动调用

```cpp
OCRProcessor ocr("./models/paddle_ocr", "CPU");

// 运行OCR
std::vector<TextBox> text_boxes = ocr.DetectText("test.png");
std::vector<OCRResult> ocr_results = ocr.RecognizeText("test.png", text_boxes);

// 手动调用可视化
ocr.VisualizeOCRResults(
    "test.png",           // 输入图像路径
    text_boxes,           // 检测框
    ocr_results,          // 识别结果
    "output_result.jpg"   // 输出路径（可选，空字符串则自动生成）
);
```

### 输出示例

```
[VisualizeOCRResults] Visualization saved to: test_ocr_result.jpg
```

---

## 📊 可视化效果

### 图像输出格式

```
原始图像: test.png
输出图像: test_ocr_result.jpg
```

### 绘制内容

1. **检测框**
   - 颜色: 绿色 (0, 255, 0)
   - 线宽: 2px
   - 形状: 多边形（四个顶点）

2. **文本标签**
   - 位置: 检测框上方
   - 格式: `文本内容 (置信度)`
   - 示例: `你好世界 (0.98)`
   - 文字颜色: 黑色 (0, 0, 0)
   - 背景颜色: 绿色 (0, 255, 0)

3. **索引编号**
   - 位置: 检测框左上角向下偏移
   - 颜色: 蓝色 (255, 0, 0)
   - 字体大小: 0.8

---

## 🔍 实际使用示例

### Python 完整示例

```python
from pathlib import Path
from ocr_module import PaddleOCRWithOpenVINO

# 初始化OCR
ocr = PaddleOCRWithOpenVINO(
    models_dir='./models/paddle_ocr',
    download_models=False
)

# 处理图像
image_path = "C:\\test\\document.jpg"
result_text = ocr.run_paddle_ocr(image=image_path)

print("识别结果:", result_text)
print("可视化图像已保存")
# 输出: document_ocr_result.jpg
```

### C++ 完整示例

```cpp
#include "OCRProcessor.h"
#include <iostream>

int main() {
    // 初始化OCR处理器
    OCRProcessor ocr("./models/paddle_ocr", "CPU");
    
    if (!ocr.IsInitialized()) {
        std::cerr << "OCR初始化失败: " << ocr.GetLastError() << std::endl;
        return 1;
    }
    
    // 处理图像
    std::string image_path = "test.jpg";
    std::string result = ocr.ProcessImage(image_path);
    
    std::cout << "识别结果: " << result << std::endl;
    std::cout << "可视化图像已保存" << std::endl;
    // 输出: test_ocr_result.jpg
    
    return 0;
}
```

---

## 📝 控制台输出示例

### Python

```
BOXES: [array([[23, 45], [156, 45], [156, 78], [23, 78]]), ...]
==== DET TEXTUAL BBOX DONE ====
rec_res= [['你好世界', 0.9876], ['OpenVINO', 0.9654]]
==== REC DONE ====
可视化结果已保存到: test_ocr_result.jpg
Extracted Text: 你好世界 OpenVINO
```

### C++

```
[OCRProcessor] Detecting text in image: test.png
BOXES: [
  [[23, 45], [156, 45], [156, 78], [23, 78]],
  [[200, 100], [450, 100], [450, 135], [200, 135]]
]
==== DET TEXTUAL BBOX DONE ====
[OCRProcessor] Detected 2 text boxes
[OCRProcessor] Recognizing text...
rec_res= [['你好世界', 0.9876], ['OpenVINO', 0.9654]]
==== REC DONE ====
[VisualizeOCRResults] Visualization saved to: test_ocr_result.jpg
[OCRProcessor] OCR completed, result length: 18
```

---

## ⚙️ 配置选项

### 修改输出路径

**Python:**
```python
# 自定义输出路径
ocr.visualize_ocr_results(
    image_path, dt_boxes, rec_res,
    output_path="custom_path/result.jpg"
)
```

**C++:**
```cpp
// 自定义输出路径
ocr.VisualizeOCRResults(
    image_path, text_boxes, ocr_results,
    "custom_path/result.jpg"
);
```

### 禁用可视化

如果不需要可视化，可以注释掉相应的调用：

**Python:** 注释 `ocr_module.py` 中第536-541行
**C++:** 注释 `OCRProcessor.cpp` 中第162-165行

---

## 🎨 自定义绘制样式

### 修改颜色和字体大小

**Python (ocr_module.py):**
```python
# 修改检测框颜色（第413行）
cv2.polylines(vis_image, [box], True, (0, 255, 0), 2)  # BGR格式

# 修改字体大小（第416行）
cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2  # 字体、大小、粗细
```

**C++ (OCRProcessor.cpp):**
```cpp
// 修改检测框颜色
cv::polylines(vis_image, contours, true, cv::Scalar(0, 255, 0), 2);  // BGR格式

// 修改字体大小
cv::FONT_HERSHEY_SIMPLEX, 0.6, 2  // 字体、大小、粗细
```

---

## 🐛 故障排除

### 问题1: 可视化图像未生成

**原因:** 检测框或识别结果为空
**解决:** 检查输入图像质量，确保能检测到文本

### 问题2: 中文显示乱码

**Python:** OpenCV默认不支持中文，文本标签中的中文会显示为方框
**解决:** 使用PIL或其他库绘制中文，或仅显示置信度

**C++:** 同样不支持中文显示
**解决:** 可以考虑使用FreeType库或保持英文标签

### 问题3: 输出路径权限错误

**原因:** 没有写入权限
**解决:** 确保输出目录存在且有写入权限

---

## 📚 相关文件

- Python实现: `zihan_helicon_conf/Python/modules/ocr_module.py`
- C++头文件: `zihan_helicon_conf/cpp/src/modules/ocr/OCRProcessor.h`
- C++实现: `zihan_helicon_conf/cpp/src/modules/ocr/OCRProcessor.cpp`

---

**版本:** 1.0  
**更新日期:** 2025-11-26
