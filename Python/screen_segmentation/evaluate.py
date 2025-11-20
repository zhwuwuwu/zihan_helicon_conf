import os
import cv2
import glob
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

IOU_THRESHOLD = 0.8  # 预测框和GT框的IOU阈值


def compute_iou(box1, box2):
    """计算两个框的IOU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area


def evaluate():
    model = YOLO("shared_screen.pt")
    names = model.names

    image_dir = "dataset/images/test"
    label_dir = "dataset/labels/test"
    output_dir = "outputs_eval"
    os.makedirs(output_dir, exist_ok=True)

    stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    error_log = defaultdict(lambda: {"TP": [], "FP": [], "FN": []})

    image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        label_path = os.path.join(label_dir, filename.replace(".jpg", ".txt"))

        # 读取GT
        gt_boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    cls, x, y, w, h = map(float, line.strip().split())
                    gt_boxes.append((int(cls), x, y, w, h))

        img = cv2.imread(img_path)
        h, w, _ = img.shape

        # 转换GT为xyxy
        gt_boxes_xyxy = []
        for cls, x, y, bw, bh in gt_boxes:
            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)
            x2 = int((x + bw / 2) * w)
            y2 = int((y + bh / 2) * h)
            gt_boxes_xyxy.append((cls, x1, y1, x2, y2))

        # 模型预测
        results = model.predict(source=img_path, conf=0.8, save=False, show=False)
        pred_boxes_xyxy = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                pred_boxes_xyxy.append((cls, x1, y1, x2, y2, conf))

        # 每个类别只保留 1 个最高置信度框
        best_per_class = {}
        for cls, x1, y1, x2, y2, conf in pred_boxes_xyxy:
            if cls not in best_per_class or conf > best_per_class[cls][-1]:
                best_per_class[cls] = (cls, x1, y1, x2, y2, conf)
        pred_boxes_xyxy = list(best_per_class.values())

        # 匹配预测和GT
        matched_gt = set()
        for i, (p_cls, px1, py1, px2, py2, conf) in enumerate(pred_boxes_xyxy):
            pred_box = (px1, py1, px2, py2)
            best_iou, best_gt_idx = 0, -1
            for j, (g_cls, gx1, gy1, gx2, gy2) in enumerate(gt_boxes_xyxy):
                if g_cls != p_cls:
                    continue
                iou = compute_iou(pred_box, (gx1, gy1, gx2, gy2))
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            if best_iou >= IOU_THRESHOLD:
                stats[p_cls]["TP"] += 1
                matched_gt.add(best_gt_idx)
                error_log[p_cls]["TP"].append((filename, pred_box, best_iou))
            else:
                stats[p_cls]["FP"] += 1
                error_log[p_cls]["FP"].append((filename, pred_box, best_iou))

        # 统计FN
        for j, (g_cls, gx1, gy1, gx2, gy2) in enumerate(gt_boxes_xyxy):
            if j not in matched_gt:
                stats[g_cls]["FN"] += 1
                error_log[g_cls]["FN"].append((filename, (gx1, gy1, gx2, gy2), None))

        # 画框：GT红色，预测绿色
        for cls, x1, y1, x2, y2 in gt_boxes_xyxy:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, names[cls], (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        for cls, x1, y1, x2, y2, conf in pred_boxes_xyxy:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{names[cls]} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, img)

    # 输出指标
    print("\n=== Evaluation Results ===")
    total_tp, total_fp, total_fn = 0, 0, 0
    for cls, m in stats.items():
        tp, fp, fn = m["TP"], m["FP"], m["FN"]
        total_tp += tp
        total_fp += fp
        total_fn += fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"Class {cls} ({names[cls]}): TP={tp}, FP={fp}, FN={fn} "
              f"| Precision={precision:.2%}, Recall={recall:.2%}, F1-score={f1:.2%}")

        # 打印错误分析
        if error_log[cls]["FN"]:
            print(f"  FN (漏检): {[f'{fn[0]} {fn[1]}' for fn in error_log[cls]['FN']]}")
        if error_log[cls]["FP"]:
            print(f"  FP (误检): {[f'{fp[0]} {fp[1]}' for fp in error_log[cls]['FP']]}")

    # 总体统计
    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0
    print(f"\nOverall: TP={total_tp}, FP={total_fp}, FN={total_fn} "
          f"| Precision={total_precision:.2%}, Recall={total_recall:.2%}, F1-score={total_f1:.2%}")


if __name__ == "__main__":
    evaluate()
