"""
EDA скрипт для COCO Keypoints датасета (HAR)
Использование: python eda_har_coco.py --ann path/to/annotations.json [--img-dir path/to/images]
"""

import json
import argparse
import os
import numpy as np
from collections import defaultdict, Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─── COCO keypoint names (17 точек, стандарт COCO person) ───────────────────
COCO_KP_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# ─── Загрузка данных ──────────────────────────────────────────────────────────
def load_coco(ann_path):
    with open(ann_path, "r") as f:
        data = json.load(f)
    print(f"[OK] Загружен файл: {ann_path}")
    return data

# ─── Базовая статистика ───────────────────────────────────────────────────────
def basic_stats(data):
    images     = data.get("images", [])
    anns       = data.get("annotations", [])
    categories = data.get("categories", [])

    print(f"  Изображений : {len(images)}")
    print(f"  Аннотаций   : {len(anns)}")
    print(f"  Категорий   : {len(categories)}")
    for cat in categories:
        print(f"    └─ id={cat['id']}: {cat['name']} | keypoints: {len(cat.get('keypoints', []))}")

    no_kp = sum(1 for a in anns if not a.get("keypoints"))
    print(f"  Без keypoints : {no_kp}")

    ann_img_ids  = set(a["image_id"] for a in anns)
    img_ids      = set(img["id"] for img in images)
    unannotated  = img_ids - ann_img_ids
    print(f"  Без аннотаций : {len(unannotated)} изображений")

    return images, anns, categories

# ─── Статистика кейпоинтов ────────────────────────────────────────────────────
def keypoint_stats(anns, n_kp=17):
    """Считаем visibility для каждой точки: 0=нет, 1=occluded, 2=visible"""
    vis_counts       = np.zeros((n_kp, 3), dtype=int)
    persons_per_image = Counter()

    for ann in anns:
        kps    = ann.get("keypoints", [])
        img_id = ann.get("image_id")
        persons_per_image[img_id] += 1

        if len(kps) < n_kp * 3:
            continue
        for i in range(n_kp):
            v = int(kps[i * 3 + 2])
            v = min(v, 2)
            vis_counts[i, v] += 1

    total         = vis_counts.sum(axis=1)
    visible_rate  = np.where(total > 0, vis_counts[:, 2] / total, 0)
    occluded_rate = np.where(total > 0, vis_counts[:, 1] / total, 0)
    invisible_rate= np.where(total > 0, vis_counts[:, 0] / total, 0)

    print("\n══════════════════════════════════")
    print("  VISIBILITY КЕЙПОИНТОВ (%)")
    print("══════════════════════════════════")
    print(f"  {'Keypoint':<16} {'Visible':>8} {'Occluded':>9} {'Missing':>8}")
    print(f"  {'─'*16} {'─'*8} {'─'*9} {'─'*8}")
    for i, name in enumerate(COCO_KP_NAMES[:n_kp]):
        print(f"  {name:<16} {visible_rate[i]*100:>7.1f}% {occluded_rate[i]*100:>8.1f}% {invisible_rate[i]*100:>7.1f}%")

    ppi = list(persons_per_image.values())
    if ppi:
        print(f"\n  Людей на кадр : min={min(ppi)}, "
              f"max={max(ppi)}, "
              f"mean={np.mean(ppi):.1f}")

    return vis_counts, visible_rate, occluded_rate, invisible_rate

# ─── Статистика bbox ──────────────────────────────────────────────────────────
def bbox_stats(anns):
    widths, heights, areas, aspects = [], [], [], []
    for ann in anns:
        bbox = ann.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x, y, w, h = bbox
        if w > 0 and h > 0:
            widths.append(w)
            heights.append(h)
            areas.append(w * h)
            aspects.append(w / h)

    if not widths:
        print("\n  [!] bbox отсутствуют в аннотациях")
        return widths, heights, areas, aspects

    print("\n══════════════════════════════════")
    print("  СТАТИСТИКА BBOX")
    print("══════════════════════════════════")
    print(f"  Ширина  : min={min(widths):.0f}  max={max(widths):.0f}  mean={np.mean(widths):.0f}")
    print(f"  Высота  : min={min(heights):.0f}  max={max(heights):.0f}  mean={np.mean(heights):.0f}")
    print(f"  Площадь : min={min(areas):.0f}  max={max(areas):.0f}  mean={np.mean(areas):.0f}")
    print(f"  Aspect  : min={min(aspects):.2f}  max={max(aspects):.2f}  mean={np.mean(aspects):.2f}")

    tiny = sum(1 for w, h in zip(widths, heights) if w < 32 or h < 32)
    print(f"  Tiny (<32px): {tiny} ({tiny/len(widths)*100:.1f}%)")

    return widths, heights, areas, aspects

# ─── Генерация графиков ───────────────────────────────────────────────────────
def plot_eda(vis_counts, visible_rate, occluded_rate, invisible_rate,
             widths, heights, areas, aspects,
             n_kp, output_dir):

    kp_names = COCO_KP_NAMES[:n_kp]
    os.makedirs(output_dir, exist_ok=True)
    COLORS = ["#4c9be8", "#f4a142", "#5dbf6e", "#e05c5c", "#a57ee8",
              "#4ecdc4", "#ff6b9d", "#c0ca33", "#ff8a65", "#78909c"]

    # ── 1. Keypoint visibility heatmap ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(n_kp)
    w = 0.28
    ax.bar(x - w, visible_rate  * 100, w, label="Visible",  color=COLORS[0], alpha=0.9)
    ax.bar(x,     occluded_rate * 100, w, label="Occluded", color=COLORS[1], alpha=0.9)
    ax.bar(x + w, invisible_rate* 100, w, label="Missing",  color=COLORS[3], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(kp_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Процент (%)")
    ax.set_xlabel("Keypoint")
    ax.set_title("Visibility кейпоинтов по всем аннотациям")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_keypoint_visibility.png"), dpi=150)
    plt.close()
    print(f"  [plot] 01_keypoint_visibility.png")

    # ── 2. BBox размеры ───────────────────────────────────────────────────────
    if widths and heights:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].hist(widths,   bins=40, color=COLORS[2], alpha=0.85, edgecolor="white")
        axes[0].set_title("Ширина bbox (px)")
        axes[0].set_xlabel("px"); axes[0].set_ylabel("Кол-во")
        axes[0].grid(axis="y", alpha=0.3)

        axes[1].hist(heights,  bins=40, color=COLORS[4], alpha=0.85, edgecolor="white")
        axes[1].set_title("Высота bbox (px)")
        axes[1].set_xlabel("px"); axes[1].set_ylabel("Кол-во")
        axes[1].grid(axis="y", alpha=0.3)

        axes[2].hist(aspects,  bins=40, color=COLORS[1], alpha=0.85, edgecolor="white")
        axes[2].set_title("Aspect ratio bbox (w/h)")
        axes[2].set_xlabel("w/h"); axes[2].set_ylabel("Кол-во")
        axes[2].grid(axis="y", alpha=0.3)

        plt.suptitle("Статистика bounding boxes", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "02_bbox_stats.png"), dpi=150)
        plt.close()
        print(f"  [plot] 02_bbox_stats.png")

    # ── 3. Missing keypoints heatmap ─────────────────────────────────────────
    missing_rate = invisible_rate.reshape(1, -1)
    fig, ax = plt.subplots(figsize=(14, 2.2))
    im = ax.imshow(missing_rate * 100, cmap="Reds", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(n_kp))
    ax.set_xticklabels(kp_names, rotation=35, ha="right", fontsize=9)
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.08, pad=0.35,
                 label="% missing (visibility=0)")
    ax.set_title("Тепловая карта: пропущенные кейпоинты (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_missing_heatmap.png"), dpi=150)
    plt.close()
    print(f"  [plot] 03_missing_heatmap.png")

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="EDA для COCO Keypoints / HAR датасета")
    parser.add_argument("--ann",     required=True, help="Путь к JSON-файлу аннотаций COCO")
    parser.add_argument("--img-dir", default=None,  help="Путь к папке с изображениями (опционально)")
    parser.add_argument("--out",     default="eda_output", help="Папка для сохранения графиков")
    args = parser.parse_args()

    data = load_coco(args.ann)
    images, anns, categories = basic_stats(data)

    n_kp = 17
    for cat in categories:
        kps = cat.get("keypoints", [])
        if kps:
            n_kp = len(kps)
            break

    vis_counts, visible_rate, occluded_rate, invisible_rate = keypoint_stats(anns, n_kp)
    widths, heights, areas, aspects = bbox_stats(anns)

    print("\n══════════════════════════════════")
    print("  ГЕНЕРАЦИЯ ГРАФИКОВ")
    print("══════════════════════════════════")
    plot_eda(vis_counts, visible_rate, occluded_rate, invisible_rate,
             widths, heights, areas, aspects,
             n_kp, args.out)

    print(f"\n✅ EDA завершён. Графики сохранены в: {os.path.abspath(args.out)}/")
    print("  Файлы:")
    print("    01_keypoint_visibility.png — visibility по суставам")
    print("    02_bbox_stats.png          — размеры bbox")
    print("    03_missing_heatmap.png     — тепловая карта пропусков")

if __name__ == "__main__":
    main()
