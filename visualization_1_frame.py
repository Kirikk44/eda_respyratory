#!/usr/bin/env python3
"""
Визуализация скелета кейпоинтов на одном кадре из COCO-датасета.

Использование:
  python visualization_1_frame.py --ann annotations.json
  python visualization_1_frame.py --ann annotations.json --image-id 42
  python visualization_1_frame.py --ann annotations.json --random
  python visualization_1_frame.py --ann annotations.json --out frame.png
"""

import json
import argparse
import os
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# ─── Стандартный COCO skeleton (17 kp) ───────────────────────────────────────
COCO_SKELETON = [
    [0, 1], [0, 2], [1, 3], [2, 4],           # голова
    [5, 6],                                     # плечи
    [5, 7], [7, 9],                             # левая рука
    [6, 8], [8, 10],                            # правая рука
    [5, 11], [6, 12], [11, 12],                # торс
    [11, 13], [13, 15],                         # левая нога
    [12, 14], [14, 16],                         # правая нога
]

PERSON_COLORS = [
    "#4fc3f7", "#81c784", "#ffb74d",
    "#f06292", "#ce93d8", "#4dd0e1",
    "#aed581", "#ff8a65",
]

BG_COLOR  = "#12121e"
BOX_COLOR = "#1e1e30"


def load_coco(ann_path):
    with open(ann_path, "r") as f:
        return json.load(f)


def build_skeleton(data):
    """Берём skeleton из самого JSON, иначе COCO-default."""
    for cat in data.get("categories", []):
        sk = cat.get("skeleton", [])
        if sk:
            # COCO хранит 1-based индексы
            return [[s - 1, e - 1] for s, e in sk]
    return COCO_SKELETON


def pick_frame(data, image_id=None, use_random=False):
    """Выбираем кадр: по id / случайный / с максимальным числом кейпоинтов."""
    images = {img["id"]: img for img in data.get("images", [])}
    anns   = data.get("annotations", [])

    if image_id is not None:
        if image_id not in images:
            raise ValueError("image_id {} не найден в датасете.".format(image_id))
        frame_anns = [a for a in anns if a["image_id"] == image_id]
        return images[image_id], frame_anns

    # Группируем аннотации по изображению
    by_img = {}
    for a in anns:
        by_img.setdefault(a["image_id"], []).append(a)

    if use_random:
        iid = random.choice(list(by_img.keys()))
    else:
        # Кадр с максимальной суммой видимых кейпоинтов
        iid = max(by_img, key=lambda i: sum(
            a.get("num_keypoints", 0) for a in by_img[i]
        ))

    return images[iid], by_img[iid]


def draw_frame(img_info, frame_anns, skeleton, out_path):
    W = img_info.get("width",  1280)
    H = img_info.get("height", 720)
    fname  = img_info.get("file_name", "frame")
    n_pers = len(frame_anns)

    # Собираем заголовок
    actions = [a.get("attributes", {}).get("action", "") for a in frame_anns]
    actions = [a for a in actions if a]
    title_action = ", ".join(sorted(set(actions))) if actions else "—"

    fig, ax = plt.subplots(figsize=(max(9, W / 100), max(6, H / 100)))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Фон кадра
    ax.add_patch(patches.Rectangle(
        (0, 0), W, H,
        linewidth=1, edgecolor="#444",
        facecolor=BOX_COLOR, zorder=0
    ))

    ax.set_xlim(-30, W + 30)
    ax.set_ylim(H + 30, -30)
    ax.set_aspect("equal")
    ax.axis("off")

    for p_idx, ann in enumerate(frame_anns):
        c   = PERSON_COLORS[p_idx % len(PERSON_COLORS)]
        kps = ann.get("keypoints", [])
        n   = len(kps) // 3

        # ── Bounding box ──────────────────────────────────────────────────
        if ann.get("bbox") and ann["bbox"][2] > 0 and ann["bbox"][3] > 0:
            bx, by, bw, bh = ann["bbox"]
            ax.add_patch(patches.Rectangle(
                (bx, by), bw, bh,
                linewidth=1.3, edgecolor=c,
                facecolor="none", alpha=0.5, zorder=1
            ))
            label = "#{} {}".format(
                p_idx + 1,
                ann.get("attributes", {}).get("action", "")
            ).strip()
            ax.text(
                bx + 4, by + 16, label,
                color=c, fontsize=8, fontweight="bold",
                fontfamily="monospace",
                bbox=dict(facecolor=BG_COLOR, alpha=0.55, pad=1.5, edgecolor="none"),
                zorder=5
            )

        # ── Keypoints ─────────────────────────────────────────────────────
        pts = {}
        for i in range(n):
            x_k = kps[i * 3]
            y_k = kps[i * 3 + 1]
            v   = int(kps[i * 3 + 2])
            if v > 0:
                pts[i] = (x_k, y_k, v)

        # ── Skeleton lines ────────────────────────────────────────────────
        for s, e in skeleton:
            if s in pts and e in pts:
                ax.plot(
                    [pts[s][0], pts[e][0]],
                    [pts[s][1], pts[e][1]],
                    color=c, linewidth=2.2, alpha=0.8,
                    solid_capstyle="round", zorder=2
                )

        # ── Joint markers ─────────────────────────────────────────────────
        for i, (x_k, y_k, v) in pts.items():
            if v == 2:   # visible
                ax.plot(x_k, y_k, "o",
                        color=c, markersize=6,
                        markeredgecolor=BG_COLOR, markeredgewidth=0.8, zorder=3)
            else:        # occluded
                ax.plot(x_k, y_k, "s",
                        color=c, markersize=5, alpha=0.55,
                        markeredgecolor="white", markeredgewidth=0.6, zorder=3)

    # ── Заголовок ─────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.98,
        "{}   |   действие: {}   |   людей: {}".format(
            os.path.basename(fname), title_action, n_pers),
        ha="center", va="top",
        color="white", fontsize=9, fontfamily="monospace"
    )

    # ── Легенда ───────────────────────────────────────────────────────────
    legend_elems = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="white", markeredgecolor=BG_COLOR,
               markersize=7, label="visible  (v=2)", linestyle="None"),
        Line2D([0], [0], marker="s", color="gray",
               markerfacecolor="gray", markeredgecolor="white",
               markersize=6, label="occluded (v=1)", linestyle="None"),
    ]
    if len(frame_anns) > 1:
        for p_idx, ann in enumerate(frame_anns):
            act = ann.get("attributes", {}).get("action", "")
            lbl = "#{} {}".format(p_idx + 1, act).strip()
            legend_elems.append(Line2D(
                [0], [0], color=PERSON_COLORS[p_idx % len(PERSON_COLORS)],
                linewidth=2, label=lbl
            ))

    ax.legend(
        handles=legend_elems, loc="lower right",
        facecolor="#1e1e30", edgecolor="#555",
        labelcolor="white", fontsize=8, framealpha=0.85
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=BG_COLOR)
    plt.close()
    print("[OK] Сохранено: " + os.path.abspath(out_path))


def main():
    parser = argparse.ArgumentParser(
        description="Нарисовать скелет кейпоинтов на одном кадре COCO-датасета"
    )
    parser.add_argument("--ann",      required=True,
                        help="Путь к JSON аннотаций COCO")
    parser.add_argument("--image-id", type=int, default=None,
                        help="ID конкретного изображения (опционально)")
    parser.add_argument("--random",   action="store_true",
                        help="Выбрать случайный кадр")
    parser.add_argument("--out",      default="frame_skeleton.png",
                        help="Путь для сохранения PNG (default: frame_skeleton.png)")
    args = parser.parse_args()

    data     = load_coco(args.ann)
    skeleton = build_skeleton(data)
    img_info, frame_anns = pick_frame(data, args.image_id, args.random)

    if not frame_anns:
        print("[!] На этом кадре нет аннотаций.")
        return

    print("[INFO] Кадр: " + img_info.get("file_name", "?"))
    print("[INFO] Аннотаций: " + str(len(frame_anns)))
    print("[INFO] Костей в скелете: " + str(len(skeleton)))

    draw_frame(img_info, frame_anns, skeleton, args.out)


if __name__ == "__main__":
    main()
