from ultralytics import YOLO
import os
import cv2


class ROIProcessor:
    model = None
    model_path = '.\\models\\shared_screen.pt'
    conf = 0.8
    def __init__(self, model_path='', conf=None):
        if model_path:
            self.model_path = model_path
        
        if conf: # 置信度
           self.conf = conf

        self.model = YOLO(self.model_path)
        if not self.model:
            print(f'*** ROIProcessor: init model failed, model = {self.model_path}***')

    
    def roi_predict(self, keyFrame, subDir=''):
        # 1.keyFrame图上画框
        # 2.保存cropped图片到相同目录下
        keyFrame_crop = ''
        crop_only = ''
        directory, filename = os.path.split(keyFrame)
        base_name, ext = os.path.splitext(filename)
        if subDir:
            if subDir in str(directory):
                subDir = ''
        base_name_crop = f"{base_name}_crop{ext}"
        os.makedirs(os.path.join(directory, subDir), exist_ok=True)
        full_name_crop = os.path.join(directory, subDir, base_name_crop)
        if os.path.exists(full_name_crop):
            return full_name_crop
        results = self.model.predict(
            source=keyFrame,
            conf=0.8,       # 置信度阈值
            save=False,     # 不用默认保存
            show=False
        )
        for result in results:
            img = result.orig_img.copy()
            filename = os.path.basename(result.path)
            base_name, ext = os.path.splitext(filename)
            for i, box in enumerate(result.boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # 画绿色框
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{self.model.names[cls]} {conf:.2f}", 
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (0, 255, 0), 2)
                
                # 裁剪检测框区域
                cropped_img = img[y1:y2, x1:x2]
                
                # 保存裁剪的图片
                if 'ppt_content' in str(self.model.names[cls]):
                    # 确保检测到的区域为ppt
                    crop_only = f"{base_name}_crop{ext}"
                    crop_path = os.path.join(directory, subDir, crop_only)
                    cv2.imwrite(crop_path, cropped_img)
                    crop_only = crop_path
                    #print(f"保存裁剪图片到 {crop_path}")

            # if crop_only:
            #     # 保存画框图像
            #     filename = os.path.basename(result.path)
            #     cv2.imwrite(keyFrame, img)
            #     #print(f"保存画框结果到 {keyFrame}")

        return crop_only if crop_only else keyFrame


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
    roi = ROIProcessor()
    result = roi.roi_predict(r'C:\Users\SAS\Downloads\multi-modal-meeting-system\Teams_20250709_171648\d7b24837-2dfe-4cc8-afc5-ef2d017b5eab\1752053193363_00117_keyFrame.jpg')
    #print(result)

    #infer()
    # model = YOLO("shared_screen.pt")
    # results = model.predict(
    #     source=r'C:\Users\SAS\Downloads\multi-modal-meeting-system\screen_segmentation\original_images\teams_0922\images\00001.jpg',
    #     conf=0.8,       # 置信度阈值
    #     save=False,     # 不用默认保存
    #     show=False
    # )

    # # 创建输出目录
    # os.makedirs('outputs/cropped', exist_ok=True)

    # for result in results:
    #     img = result.orig_img.copy()
    #     filename = os.path.basename(result.path)
    #     base_name = os.path.splitext(filename)[0]
    #     for i, box in enumerate(result.boxes):
    #         cls = int(box.cls[0])
    #         conf = float(box.conf[0])
    #         x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

    #         # 画绿色框
    #         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #         cv2.putText(img, f"{model.names[cls]} {conf:.2f}", 
    #                     (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
    #                     0.6, (0, 255, 0), 2)
            
    #         # 裁剪检测框区域
    #         cropped_img = img[y1:y2, x1:x2]
            
    #         # 保存裁剪的图片
    #         crop_filename = f"{base_name}_crop_{i}_{model.names[cls]}_{conf:.2f}.jpg"
    #         crop_path = os.path.join('outputs/cropped', crop_filename)
    #         cv2.imwrite(crop_path, cropped_img)
    #         print(f"保存裁剪图片到 {crop_path}")

    #     # 保存结果图像
    #     filename = os.path.basename(result.path)
    #     save_path = os.path.join('outputs', filename)
    #     cv2.imwrite(save_path, img)
    #     print(f"保存结果到 {save_path}")

