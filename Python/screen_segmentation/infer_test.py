from ultralytics import YOLO
import os
import cv2

def infer():
    model = YOLO("shared_screen.pt")

    input_root = "original_images"
    output_root = "outputs"
    
    for subdir in ["summary_ppt0708/images", "teams_0922/images"]:
        input_dir = os.path.join(input_root, subdir)
        output_dir = os.path.join(output_root, subdir)
        os.makedirs(output_dir, exist_ok=True)

        results = model.predict(
            source=input_dir,
            conf=0.8,       # 置信度阈值
            save=False,     # 不用默认保存
            show=False
        )

        for result in results:
            img = result.orig_img.copy()

            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # 画绿色框
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{model.names[cls]} {conf:.2f}", 
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (0, 255, 0), 2)

            # 保存结果图像
            filename = os.path.basename(result.path)
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, img)
            print(f"[{subdir}] 保存结果到 {save_path}")


if __name__ == "__main__":
    #infer()
    model = YOLO("shared_screen.pt")
    results = model.predict(
        source=r'C:\Users\SAS\Downloads\multi-modal-meeting-system\screen_segmentation\original_images\teams_0922\images\00001.jpg',
        conf=0.8,       # 置信度阈值
        save=False,     # 不用默认保存
        show=False
    )

    # 创建输出目录
    os.makedirs('outputs/cropped', exist_ok=True)

    for result in results:
        img = result.orig_img.copy()
        filename = os.path.basename(result.path)
        base_name = os.path.splitext(filename)[0]
        for i, box in enumerate(result.boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # 画绿色框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{model.names[cls]} {conf:.2f}", 
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 255, 0), 2)
            
            # 裁剪检测框区域
            cropped_img = img[y1:y2, x1:x2]
            
            # 保存裁剪的图片
            crop_filename = f"{base_name}_crop_{i}_{model.names[cls]}_{conf:.2f}.jpg"
            crop_path = os.path.join('outputs/cropped', crop_filename)
            cv2.imwrite(crop_path, cropped_img)
            print(f"保存裁剪图片到 {crop_path}")

        # 保存结果图像
        filename = os.path.basename(result.path)
        save_path = os.path.join('outputs', filename)
        cv2.imwrite(save_path, img)
        print(f"保存结果到 {save_path}")













# from ultralytics import YOLO
# import os

# def infer():
#     model = YOLO("shared_screen.pt")  # 训练好的模型

#     input_dir = "test"
#     output_dir = "outputs"
#     os.makedirs(output_dir, exist_ok=True)

#     # 推理并保存结果，只保留置信度 >= 0.9
#     model.predict(
#         source=input_dir,
#         save=True,           # 保存图片
#         project=output_dir,  # 输出目录
#         name="",             # 不生成子文件夹
#         exist_ok=True,
#         show=False,          # 不显示窗口
#         conf=0.9            # ⚡ 置信度阈值
#     )

# if __name__ == "__main__":
#     infer()
