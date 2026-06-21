#!/usr/bin/env python3
"""Convert COCO person keypoints annotations into a YOLO body-parts dataset.

Purpose:
- Extract only body-part targets (testa, mano, busto) from COCO keypoints.

Dataset expected:
- COCO-style images + JSON annotations (person keypoints).

Classes to keep:
0 -> testa (head/face keypoint)
1 -> mano (hand keypoint)
2 -> busto (torso/chest keypoint)

Maps COCO keypoints to body parts:
- Head: nose (0), left_eye (1), right_eye (2), left_ear (3), right_ear (4)
- Hands: left_wrist (9), right_wrist (10), left_hand (21), right_hand (22) [if available]
- Torso: neck (1 custom), left_shoulder (5), right_shoulder (6), left_hip (11), right_hip (12)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2


COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

KEYPOINT_TO_CLASS = {
    # Head
    0: 0,   # nose -> testa
    1: 0,   # left_eye -> testa
    2: 0,   # right_eye -> testa
    3: 0,   # left_ear -> testa
    4: 0,   # right_ear -> testa
    # Hands
    9: 1,   # left_wrist -> mano
    10: 1,  # right_wrist -> mano
    # Torso
    5: 2,   # left_shoulder -> busto
    6: 2,   # right_shoulder -> busto
    11: 2,  # left_hip -> busto
    12: 2,  # right_hip -> busto
}

CLASS_NAMES = ["testa", "mano", "busto"]


def bbox_from_keypoints(keypoints: List[float], img_width: int, img_height: int) -> Optional[Tuple[float, float, float, float]]:
    """
    Convert COCO keypoints to bounding box in YOLO format (normalized).
    Returns (x_center, y_center, width, height) or None if no valid keypoints.
    """
    valid_x = []
    valid_y = []

    for i in range(0, len(keypoints), 3):
        x, y, v = keypoints[i], keypoints[i + 1], keypoints[i + 2]
        if v > 0:  # visibility > 0
            valid_x.append(x)
            valid_y.append(y)

    if not valid_x or not valid_y:
        return None

    x_min, x_max = min(valid_x), max(valid_x)
    y_min, y_max = min(valid_y), max(valid_y)

    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height

    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.01, min(1.0, width))
    height = max(0.01, min(1.0, height))

    return x_center, y_center, width, height


def process_coco_annotation(
    ann_file: Path,
    images_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
) -> int:
    """
    Process COCO annotation file and convert to YOLO format.
    Filters for body part keypoints only.
    Returns number of annotations processed.
    """
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    with open(ann_file, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    image_id_to_info = {img["id"]: img for img in coco_data.get("images", [])}
    annotations = coco_data.get("annotations", [])

    processed = 0
    skipped_no_boxes = 0
    skipped_no_image = 0

    for ann_idx, ann in enumerate(annotations):
        image_id = ann.get("image_id")
        if image_id not in image_id_to_info:
            skipped_no_image += 1
            continue

        img_info = image_id_to_info[image_id]
        img_filename = img_info["file_name"]
        img_path = images_dir / img_filename
        img_width = img_info["width"]
        img_height = img_info["height"]

        if not img_path.exists():
            skipped_no_image += 1
            continue

        # Extract keypoints for body parts
        keypoints = ann.get("keypoints", [])
        class_boxes = {}  # class_id -> list of (x_center, y_center, width, height)

        for kpt_idx, class_id in KEYPOINT_TO_CLASS.items():
            if kpt_idx * 3 + 2 < len(keypoints):
                x = keypoints[kpt_idx * 3]
                y = keypoints[kpt_idx * 3 + 1]
                v = keypoints[kpt_idx * 3 + 2]

                if v > 0 and x > 0 and y > 0:
                    if class_id not in class_boxes:
                        class_boxes[class_id] = []
                    class_boxes[class_id].append((x, y))

        if not class_boxes:
            skipped_no_boxes += 1
            continue

        # Convert to bounding boxes per class
        labels_lines = []
        for class_id in sorted(class_boxes.keys()):
            points = class_boxes[class_id]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            x_center = (x_min + x_max) / 2.0 / img_width
            y_center = (y_min + y_max) / 2.0 / img_height
            box_w = (x_max - x_min) / img_width
            box_h = (y_max - y_min) / img_height

            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            box_w = max(0.01, min(1.0, box_w))
            box_h = max(0.01, min(1.0, box_h))

            labels_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
            )

        if not labels_lines:
            skipped_no_boxes += 1
            continue

        # Write label file
        label_path = output_labels_dir / f"{img_filename.rsplit('.', 1)[0]}.txt"
        label_path.write_text("\n".join(labels_lines), encoding="utf-8")

        # Copy image
        output_img_path = output_images_dir / img_filename
        cv2.imwrite(str(output_img_path), cv2.imread(str(img_path)))

        processed += 1
        if processed % 100 == 0:
            print(f"Processed {processed} annotations...")

    print(f"\n✓ Processed: {processed}")
    print(f"✗ Skipped (no valid boxes): {skipped_no_boxes}")
    print(f"✗ Skipped (image not found): {skipped_no_image}")

    return processed


def create_data_yaml(output_root: Path, split_name: str) -> Path:
    """Create YAML file for YOLO training."""
    data_yaml_content = f"""path: {str(output_root.resolve())}
train: {split_name}/images
val: {split_name}/images

names:
  0: testa
  1: mano
  2: busto
"""
    yaml_path = output_root / f"data_{split_name}.yaml"
    yaml_path.write_text(data_yaml_content, encoding="utf-8")
    return yaml_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Standardize COCO to YOLO format (body parts only)")
    parser.add_argument("--coco-root", type=Path, default=Path("dataset/coco2017"), help="COCO dataset root")
    parser.add_argument("--output-root", type=Path, default=Path("dataset/standardized_datasets/coco2017_body_parts"), help="Output root")
    parser.add_argument("--split", type=str, choices=["train", "val", "both"], default="both", help="Which split to process")
    args = parser.parse_args()

    coco_root: Path = args.coco_root.resolve()
    output_root: Path = args.output_root.resolve()

    if not coco_root.exists():
        print(f"COCO root not found: {coco_root}")
        return 1

    output_root.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val"] if args.split == "both" else [args.split]

    total_processed = 0

    for split in splits:
        split_suffix = "2017"
        ann_file = coco_root / "annotations" / f"person_keypoints_{split}{split_suffix}.json"
        images_dir = coco_root / f"{split}{split_suffix}"
        output_images_dir = output_root / split / "images"
        output_labels_dir = output_root / split / "labels"

        if not ann_file.exists():
            print(f"Annotation file not found: {ann_file}")
            continue

        print(f"\nProcessing {split.upper()} split...")
        processed = process_coco_annotation(ann_file, images_dir, output_images_dir, output_labels_dir)
        total_processed += processed

        create_data_yaml(output_root, split)
        print(f"Data YAML created: {output_root}/data_{split}.yaml")

    print(f"\n{'='*60}")
    print(f"Total annotations processed: {total_processed}")
    print(f"Output directory: {output_root}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
