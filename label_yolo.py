#!/usr/bin/env python3
"""Simple YOLO labeler for body-part detection.

Purpose:
- Manually annotate body-part boxes and save labels in YOLO format.

Dataset expected:
- Local image folder in `DATASET/images`; labels are saved to `DATASET/labels`.

Classes:
0 -> testa
1 -> mano
2 -> busto

Usage:
    python label_yolo.py --dataset-dir DATASET

Place images in DATASET/images.
Labels will be saved to DATASET/labels with the same stem as image names.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ["testa", "mano", "busto"]
WINDOW_NAME = "YOLO Labeler"


@dataclass
class Box:
    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int

    def normalized(self, width: int, height: int) -> Tuple[float, float, float, float]:
        x1, x2 = sorted((self.x1, self.x2))
        y1, y2 = sorted((self.y1, self.y2))

        x_center = ((x1 + x2) / 2.0) / width
        y_center = ((y1 + y2) / 2.0) / height
        box_w = (x2 - x1) / width
        box_h = (y2 - y1) / height
        return x_center, y_center, box_w, box_h


def clamp(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(value, max_value))


def list_images(images_dir: Path) -> List[Path]:
    return sorted(
        [
            p
            for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def label_path_for(image_path: Path, labels_dir: Path) -> Path:
    return labels_dir / f"{image_path.stem}.txt"


def load_labels(label_path: Path, width: int, height: int) -> List[Box]:
    if not label_path.exists():
        return []

    boxes: List[Box] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        box_w = float(parts[3])
        box_h = float(parts[4])

        x1 = int((x_center - box_w / 2.0) * width)
        y1 = int((y_center - box_h / 2.0) * height)
        x2 = int((x_center + box_w / 2.0) * width)
        y2 = int((y_center + box_h / 2.0) * height)

        x1 = clamp(x1, 0, width - 1)
        y1 = clamp(y1, 0, height - 1)
        x2 = clamp(x2, 0, width - 1)
        y2 = clamp(y2, 0, height - 1)

        boxes.append(Box(class_id=class_id, x1=x1, y1=y1, x2=x2, y2=y2))

    return boxes


def save_labels(label_path: Path, boxes: List[Box], width: int, height: int) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for box in boxes:
        x_center, y_center, box_w, box_h = box.normalized(width, height)
        lines.append(
            f"{box.class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
        )

    label_path.write_text("\n".join(lines), encoding="utf-8")


def draw_ui(
    image,
    boxes: List[Box],
    active_class: int,
    is_drawing: bool,
    start_point: Tuple[int, int],
    current_point: Tuple[int, int],
):
    canvas = image.copy()

    colors = {
        0: (0, 255, 255),
        1: (0, 255, 0),
        2: (255, 0, 0),
    }

    for i, box in enumerate(boxes):
        x1, x2 = sorted((box.x1, box.x2))
        y1, y2 = sorted((box.y1, box.y2))
        color = colors.get(box.class_id, (255, 255, 255))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{i + 1}: {CLASS_NAMES[box.class_id]}"
        cv2.putText(
            canvas,
            label,
            (x1, max(y1 - 6, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    if is_drawing:
        x1, y1 = start_point
        x2, y2 = current_point
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colors[active_class], 2)

    info_lines = [
        f"Class: {active_class} ({CLASS_NAMES[active_class]})",
        "Keys: 1/2/3 class | n next | p prev | s save | d delete last | c clear | q quit",
        "Mouse: left click + drag to draw box",
    ]

    y = 24
    for line in info_lines:
        cv2.putText(
            canvas,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 24

    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLO body-part labeler")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("DATASET"),
        help="Dataset root dir containing images/ and labels/",
    )
    args = parser.parse_args()

    dataset_dir: Path = args.dataset_dir
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(images_dir)
    if not images:
        print(f"No images found in: {images_dir}")
        print("Supported formats:", ", ".join(sorted(IMAGE_EXTENSIONS)))
        return 1

    current_index = 0
    active_class = 0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        image_path = images[current_index]
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path.name}")
            current_index = (current_index + 1) % len(images)
            continue

        height, width = image.shape[:2]
        boxes = load_labels(label_path_for(image_path, labels_dir), width, height)

        is_drawing = False
        start_point = (0, 0)
        current_point = (0, 0)

        def mouse_callback(event, x, y, flags, param):
            nonlocal is_drawing, start_point, current_point, boxes

            x = clamp(x, 0, width - 1)
            y = clamp(y, 0, height - 1)

            if event == cv2.EVENT_LBUTTONDOWN:
                is_drawing = True
                start_point = (x, y)
                current_point = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and is_drawing:
                current_point = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and is_drawing:
                is_drawing = False
                current_point = (x, y)
                x1, y1 = start_point
                x2, y2 = current_point
                if abs(x2 - x1) >= 4 and abs(y2 - y1) >= 4:
                    boxes.append(Box(class_id=active_class, x1=x1, y1=y1, x2=x2, y2=y2))

        cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

        while True:
            canvas = draw_ui(
                image,
                boxes,
                active_class,
                is_drawing,
                start_point,
                current_point,
            )

            status = f"{current_index + 1}/{len(images)} - {image_path.name}"
            cv2.setWindowTitle(WINDOW_NAME, status)
            cv2.imshow(WINDOW_NAME, canvas)

            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue

            if key in (ord("1"), ord("2"), ord("3")):
                active_class = key - ord("1")
            elif key == ord("d"):
                if boxes:
                    boxes.pop()
            elif key == ord("c"):
                boxes.clear()
            elif key == ord("s"):
                save_labels(label_path_for(image_path, labels_dir), boxes, width, height)
                print(f"Saved: {image_path.name}")
            elif key == ord("n"):
                save_labels(label_path_for(image_path, labels_dir), boxes, width, height)
                current_index = (current_index + 1) % len(images)
                break
            elif key == ord("p"):
                save_labels(label_path_for(image_path, labels_dir), boxes, width, height)
                current_index = (current_index - 1) % len(images)
                break
            elif key == ord("q"):
                save_labels(label_path_for(image_path, labels_dir), boxes, width, height)
                cv2.destroyAllWindows()
                return 0


if __name__ == "__main__":
    raise SystemExit(main())
