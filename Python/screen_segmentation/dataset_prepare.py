import os
import shutil
import random

def split_dataset(image_dir, label_dir, dataset_dir, ratios=(0.8, 0.1, 0.1), seed=12):
    random.seed(seed)

    # 获取所有图片文件
    images = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    images.sort()
    random.shuffle(images)

    n_total = len(images)
    n_train = int(ratios[0] * n_total)
    n_val = int(ratios[1] * n_total)
    n_test = n_total - n_train - n_val

    split_dict = {
        "train": images[:n_train],
        "val": images[n_train:n_train+n_val],
        "test": images[n_train+n_val:]
    }

    for split, split_images in split_dict.items():
        img_out = os.path.join(dataset_dir, "images", split)
        lbl_out = os.path.join(dataset_dir, "labels", split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for img_file in split_images:
            # 拷贝图片
            shutil.copy(os.path.join(image_dir, img_file), os.path.join(img_out, img_file))

            # 拷贝对应标签（同名 txt）
            label_file = os.path.splitext(img_file)[0] + ".txt"
            src_label = os.path.join(label_dir, label_file)
            if os.path.exists(src_label):
                shutil.copy(src_label, os.path.join(lbl_out, label_file))

        print(f"{os.path.basename(image_dir)} -> {split}: {len(split_images)} 张图像")


if __name__ == "__main__":
    dataset_dir = "dataset"
    raw_dataset_dir = "original_images"

    # 先清空 dataset 目录
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(os.path.join(dataset_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "labels"), exist_ok=True)

    # 子目录名称
    subdirs = ["summary_ppt0708", "teams_0922"]

    # 生成数据源 (images, labels)
    data_sources = [
        (os.path.join(raw_dataset_dir, subdir, "images"),
        os.path.join(raw_dataset_dir, subdir, "labels"))
        for subdir in subdirs
    ]


    for img_dir, lbl_dir in data_sources:
        split_dataset(img_dir, lbl_dir, dataset_dir, ratios=(0.8, 0.1, 0.1))
