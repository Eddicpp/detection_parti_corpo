#!/usr/bin/env python3
"""Unify COCO + MPII human pose datasets into two YOLO stages.

Purpose:
- Build a hierarchical training dataset for person -> body parts detection.

Source datasets:
- COCO 2017 Person Keypoints (train/val JSON annotations).
- MPII Human Pose (JSON converted format or original `.mat`, when parsable).

Stage 1: person detection (single class: person)
Stage 2: body-part detection on person crops (6 classes)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm


PART_NAMES = ["head", "torso", "left_arm", "right_arm", "left_hand", "right_hand"]

COCO_KP_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

MPII_TO_NAME = {
    0: "right_ankle",
    1: "right_knee",
    2: "right_hip",
    3: "left_hip",
    4: "left_knee",
    5: "left_ankle",
    6: "pelvis",
    7: "thorax",
    8: "upper_neck",
    9: "head_top",
    10: "right_wrist",
    11: "right_elbow",
    12: "right_shoulder",
    13: "left_shoulder",
    14: "left_elbow",
    15: "left_wrist",
}


@dataclass
class PersonInstance:
    source: str
    image_path: Path
    image_w: int
    image_h: int
    person_bbox_xyxy: Tuple[float, float, float, float]
    keypoints: Dict[str, Tuple[float, float, int]]
    head_rect_xyxy: Optional[Tuple[float, float, float, float]] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standardize COCO+MPII into hierarchical YOLO datasets (person -> body parts)"
    )
    parser.add_argument("--coco-dir", type=Path, required=True, help="COCO root directory")
    parser.add_argument("--mpii-dir", type=Path, required=True, help="MPII root directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output root directory")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--margin-head", type=float, default=0.4, help="Head margin")
    parser.add_argument("--margin-torso", type=float, default=0.1, help="Torso margin")
    parser.add_argument("--margin-arm", type=float, default=0.25, help="Arm thickness factor")
    parser.add_argument(
        "--person-margin",
        type=float,
        default=0.1,
        help="Margin used to crop person for stage 2",
    )
    parser.add_argument(
        "--symlink-images",
        action="store_true",
        help="Use symlink instead of file copy for stage 1 images",
    )
    return parser.parse_args()


def clamp_bbox_xyxy(
    bbox: Tuple[float, float, float, float], img_w: int, img_h: int
) -> Optional[Tuple[float, float, float, float]]:
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(img_w - 1), x1))
    y1 = max(0.0, min(float(img_h - 1), y1))
    x2 = max(0.0, min(float(img_w - 1), x2))
    y2 = max(0.0, min(float(img_h - 1), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def expand_bbox(
    bbox: Tuple[float, float, float, float],
    margin: float,
    img_w: int,
    img_h: int,
    extra_top: float = 0.0,
) -> Optional[Tuple[float, float, float, float]]:
    x1, y1, x2, y2 = bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    x_pad = w * margin
    y_pad = h * margin
    y_top_extra = h * extra_top
    out = (x1 - x_pad, y1 - y_pad - y_top_extra, x2 + x_pad, y2 + y_pad)
    return clamp_bbox_xyxy(out, img_w, img_h)


def yolo_from_xyxy(
    bbox: Tuple[float, float, float, float], img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    xc = ((x1 + x2) * 0.5) / img_w
    yc = ((y1 + y2) * 0.5) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return (xc, yc, w, h)


def save_yolo_label(label_path: Path, class_id: int, xyxy: Tuple[float, float, float, float], w: int, h: int) -> None:
    xc, yc, bw, bh = yolo_from_xyxy(xyxy, w, h)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8") as f:
        f.write(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


def save_yolo_labels_multi(
    label_path: Path,
    labels: Iterable[Tuple[int, Tuple[float, float, float, float]]],
    w: int,
    h: int,
) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8") as f:
        for class_id, xyxy in labels:
            xc, yc, bw, bh = yolo_from_xyxy(xyxy, w, h)
            f.write(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


def keypoints_to_bbox(
    points: List[Tuple[float, float, int]],
    img_w: int,
    img_h: int,
    min_points: int = 2,
    margin: float = 0.1,
    extra_top: float = 0.0,
) -> Optional[Tuple[float, float, float, float]]:
    valid = [(x, y) for x, y, v in points if v > 0]
    if len(valid) < min_points:
        return None
    xs = [p[0] for p in valid]
    ys = [p[1] for p in valid]
    base = (min(xs), min(ys), max(xs), max(ys))
    return expand_bbox(base, margin=margin, img_w=img_w, img_h=img_h, extra_top=extra_top)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _coco_image_map(coco_json: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {img["id"]: img for img in coco_json.get("images", [])}


def _extract_coco_person_category_id(coco_json: Dict[str, Any]) -> Optional[int]:
    for cat in coco_json.get("categories", []):
        if cat.get("name") == "person":
            return int(cat["id"])
    return None


def _parse_coco_keypoints(flat_kps: List[float]) -> Dict[str, Tuple[float, float, int]]:
    out: Dict[str, Tuple[float, float, int]] = {}
    if len(flat_kps) < 17 * 3:
        return out
    for i, name in enumerate(COCO_KP_NAMES):
        x = float(flat_kps[i * 3 + 0])
        y = float(flat_kps[i * 3 + 1])
        v = int(flat_kps[i * 3 + 2])
        out[name] = (x, y, v)
    return out


def convert_coco(coco_dir: Path) -> List[PersonInstance]:
    ann_dir = coco_dir / "annotations"
    ann_files = [
        ann_dir / "person_keypoints_train2017.json",
        ann_dir / "person_keypoints_val2017.json",
    ]

    instances: List[PersonInstance] = []

    for ann_path in ann_files:
        if not ann_path.exists():
            continue

        data = _read_json(ann_path)
        image_map = _coco_image_map(data)
        person_cat_id = _extract_coco_person_category_id(data)

        split_name = "train2017" if "train" in ann_path.name else "val2017"
        image_root = coco_dir / split_name

        anns = data.get("annotations", [])
        for ann in tqdm(anns, desc=f"COCO {ann_path.name}", unit="ann"):
            if person_cat_id is not None and int(ann.get("category_id", -1)) != person_cat_id:
                continue

            image_id = int(ann.get("image_id", -1))
            img_info = image_map.get(image_id)
            if img_info is None:
                continue

            w = int(img_info.get("width", 0))
            h = int(img_info.get("height", 0))
            if w <= 0 or h <= 0:
                continue

            file_name = img_info.get("file_name")
            if not file_name:
                continue
            image_path = image_root / str(file_name)
            if not image_path.exists():
                continue

            bbox_xywh = ann.get("bbox", None)
            if not bbox_xywh or len(bbox_xywh) != 4:
                continue
            x, y, bw, bh = [float(v) for v in bbox_xywh]
            person_bbox = clamp_bbox_xyxy((x, y, x + bw, y + bh), w, h)
            if person_bbox is None:
                continue

            kp_map = _parse_coco_keypoints(ann.get("keypoints", []))

            instances.append(
                PersonInstance(
                    source="coco",
                    image_path=image_path,
                    image_w=w,
                    image_h=h,
                    person_bbox_xyxy=person_bbox,
                    keypoints=kp_map,
                    head_rect_xyxy=None,
                )
            )

    return instances


def _find_mpii_images_root(mpii_dir: Path) -> Path:
    candidates = [
        mpii_dir / "images",
        mpii_dir / "mpii_human_pose_v1" / "images",
        mpii_dir,
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    return mpii_dir


def _parse_mpii_json_record(
    rec: Dict[str, Any],
    images_root: Path,
) -> Optional[Tuple[Path, Dict[str, Tuple[float, float, int]], Optional[Tuple[float, float, float, float]], Optional[Tuple[float, float]], Optional[float]]]:
    image_name = (
        rec.get("image")
        or rec.get("img_path")
        or rec.get("image_path")
        or rec.get("imgname")
        or rec.get("filename")
    )
    if not image_name:
        return None

    image_path = images_root / str(image_name)
    if not image_path.exists():
        return None

    joints_raw = rec.get("joints") or rec.get("annopoints") or rec.get("points") or []
    vis_raw = rec.get("joints_vis") or rec.get("visible") or rec.get("vis")

    keypoints: Dict[str, Tuple[float, float, int]] = {}

    if isinstance(joints_raw, dict):
        for key, val in joints_raw.items():
            idx = None
            if str(key).isdigit():
                idx = int(key)
            elif key in MPII_TO_NAME.values():
                idx = None
            if idx is not None and idx in MPII_TO_NAME and isinstance(val, (list, tuple)) and len(val) >= 2:
                x = float(val[0])
                y = float(val[1])
                v = 2
                keypoints[MPII_TO_NAME[idx]] = (x, y, v)
    elif isinstance(joints_raw, list):
        for idx, item in enumerate(joints_raw):
            if idx not in MPII_TO_NAME:
                continue
            x = y = None
            v = 0
            if isinstance(item, dict):
                x = item.get("x")
                y = item.get("y")
                if "is_visible" in item:
                    vis_value = item["is_visible"]
                    v = 2 if int(vis_value) > 0 else 0
                elif "v" in item:
                    v = int(item["v"])
                else:
                    v = 2
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                x = item[0]
                y = item[1]
                if len(item) >= 3:
                    v = int(item[2])
                else:
                    v = 2

            if x is None or y is None:
                continue

            if isinstance(vis_raw, list) and idx < len(vis_raw):
                v = 2 if int(vis_raw[idx]) > 0 else 0

            keypoints[MPII_TO_NAME[idx]] = (float(x), float(y), int(v))

    head_rect = rec.get("head_rect") or rec.get("headRect")
    head_rect_xyxy: Optional[Tuple[float, float, float, float]] = None
    if isinstance(head_rect, (list, tuple)) and len(head_rect) >= 4:
        x1, y1, x2, y2 = [float(v) for v in head_rect[:4]]
        head_rect_xyxy = (x1, y1, x2, y2)

    objpos = rec.get("objpos")
    obj_center: Optional[Tuple[float, float]] = None
    if isinstance(objpos, dict) and "x" in objpos and "y" in objpos:
        obj_center = (float(objpos["x"]), float(objpos["y"]))
    elif isinstance(objpos, (list, tuple)) and len(objpos) >= 2:
        obj_center = (float(objpos[0]), float(objpos[1]))

    scale = rec.get("scale")
    scale_value: Optional[float] = float(scale) if scale is not None else None

    return image_path, keypoints, head_rect_xyxy, obj_center, scale_value


def _load_mpii_json_records(mpii_dir: Path) -> List[Dict[str, Any]]:
    json_candidates = [
        mpii_dir / "annotations.json",
        mpii_dir / "mpii_annotations.json",
        mpii_dir / "mpii.json",
    ]

    for path in json_candidates:
        if path.exists():
            data = _read_json(path)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                for key in ["annotations", "records", "data", "items"]:
                    if isinstance(data.get(key), list):
                        return data[key]

    # fallback: first json with a list-like payload
    for path in mpii_dir.glob("*.json"):
        data = _read_json(path)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ["annotations", "records", "data", "items"]:
                if isinstance(data.get(key), list):
                    return data[key]

    return []


def _load_mpii_mat_records(mpii_dir: Path) -> List[Dict[str, Any]]:
    mat_candidates = list(mpii_dir.glob("*.mat"))
    if not mat_candidates:
        return []

    try:
        from scipy.io import loadmat  # type: ignore
    except Exception:
        print("[WARN] MPII .mat trovato ma scipy non disponibile; usa una versione JSON convertita.")
        return []

    # This parser targets common converted structures and may not cover all .mat variants.
    out: List[Dict[str, Any]] = []
    for mat_path in mat_candidates:
        try:
            mat = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
        except Exception as exc:
            print(f"[WARN] Impossibile leggere {mat_path.name}: {exc}")
            continue

        release = mat.get("RELEASE", None)
        if release is None:
            continue

        annolist = getattr(release, "annolist", None)
        if annolist is None:
            continue

        try:
            iterable = annolist if isinstance(annolist, (list, tuple, np.ndarray)) else [annolist]
            for ann in iterable:
                image_obj = getattr(ann, "image", None)
                if image_obj is None:
                    continue
                image_name = getattr(image_obj, "name", None)
                annorect = getattr(ann, "annorect", None)
                if image_name is None or annorect is None:
                    continue

                rects = annorect if isinstance(annorect, (list, tuple, np.ndarray)) else [annorect]
                for rect in rects:
                    rec: Dict[str, Any] = {"image": str(image_name)}
                    if hasattr(rect, "scale"):
                        rec["scale"] = float(getattr(rect, "scale"))
                    if hasattr(rect, "objpos"):
                        obj = getattr(rect, "objpos")
                        if hasattr(obj, "x") and hasattr(obj, "y"):
                            rec["objpos"] = {"x": float(obj.x), "y": float(obj.y)}
                    if hasattr(rect, "x1") and hasattr(rect, "y1") and hasattr(rect, "x2") and hasattr(rect, "y2"):
                        rec["head_rect"] = [
                            float(getattr(rect, "x1")),
                            float(getattr(rect, "y1")),
                            float(getattr(rect, "x2")),
                            float(getattr(rect, "y2")),
                        ]

                    joints: List[List[float]] = [[0.0, 0.0, 0] for _ in range(16)]
                    if hasattr(rect, "annopoints") and rect.annopoints is not None:
                        pts_obj = getattr(rect.annopoints, "point", None)
                        if pts_obj is not None:
                            pts = pts_obj if isinstance(pts_obj, (list, tuple, np.ndarray)) else [pts_obj]
                            for p in pts:
                                pid = getattr(p, "id", None)
                                px = getattr(p, "x", None)
                                py = getattr(p, "y", None)
                                if pid is None or px is None or py is None:
                                    continue
                                idx = int(pid)
                                if 0 <= idx < 16:
                                    vis = 1
                                    if hasattr(p, "is_visible"):
                                        vis = int(getattr(p, "is_visible"))
                                    joints[idx] = [float(px), float(py), 2 if vis > 0 else 0]
                    rec["joints"] = joints
                    out.append(rec)
        except Exception as exc:
            print(f"[WARN] Parsing .mat parziale fallito per {mat_path.name}: {exc}")

    return out


def convert_mpii(mpii_dir: Path) -> List[PersonInstance]:
    images_root = _find_mpii_images_root(mpii_dir)

    records = _load_mpii_json_records(mpii_dir)
    if not records:
        records = _load_mpii_mat_records(mpii_dir)

    if not records:
        print("[WARN] Nessuna annotazione MPII trovata (json/mat).")
        return []

    instances: List[PersonInstance] = []
    for rec in tqdm(records, desc="MPII records", unit="rec"):
        parsed = _parse_mpii_json_record(rec, images_root)
        if parsed is None:
            continue
        image_path, kp_map, head_rect, obj_center, scale = parsed

        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]

        visible_points = [p for p in kp_map.values() if p[2] > 0]
        person_bbox: Optional[Tuple[float, float, float, float]] = None
        if len(visible_points) >= 2:
            base = keypoints_to_bbox(visible_points, w, h, min_points=2, margin=0.15)
            if base is not None:
                person_bbox = base

        if person_bbox is None and obj_center is not None and scale is not None:
            size = max(10.0, float(scale) * 200.0 * 1.3)
            cx, cy = obj_center
            half = size * 0.5
            person_bbox = clamp_bbox_xyxy((cx - half, cy - half, cx + half, cy + half), w, h)

        if person_bbox is None:
            continue

        if head_rect is not None:
            head_rect = clamp_bbox_xyxy(head_rect, w, h)

        instances.append(
            PersonInstance(
                source="mpii",
                image_path=image_path,
                image_w=w,
                image_h=h,
                person_bbox_xyxy=person_bbox,
                keypoints=kp_map,
                head_rect_xyxy=head_rect,
            )
        )

    return instances


def _split_instances(
    instances: List[PersonInstance], val_fraction: float, seed: int
) -> Dict[str, List[PersonInstance]]:
    rng = random.Random(seed)
    by_source: Dict[str, List[PersonInstance]] = defaultdict(list)
    for inst in instances:
        by_source[inst.source].append(inst)

    split = {"train": [], "val": []}
    for source, src_instances in by_source.items():
        rng.shuffle(src_instances)
        n_val = int(len(src_instances) * val_fraction)
        split["val"].extend(src_instances[:n_val])
        split["train"].extend(src_instances[n_val:])
        print(f"[INFO] Split {source}: train={len(src_instances)-n_val}, val={n_val}")

    rng.shuffle(split["train"])
    rng.shuffle(split["val"])
    return split


def _copy_or_symlink(src: Path, dst: Path, use_symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if use_symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def _write_stage_yaml(stage_root: Path, names: List[str]) -> None:
    data = {
        "path": str(stage_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": names,
        "nc": len(names),
    }
    with (stage_root / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=False, sort_keys=False)


def _part_bbox_from_instance(
    inst: PersonInstance,
    part_name: str,
    margin_head: float,
    margin_torso: float,
    margin_arm: float,
) -> Optional[Tuple[float, float, float, float]]:
    w, h = inst.image_w, inst.image_h
    kp = inst.keypoints

    def get(name: str) -> Tuple[float, float, int]:
        return kp.get(name, (0.0, 0.0, 0))

    if part_name == "head":
        if inst.source == "mpii" and inst.head_rect_xyxy is not None:
            return clamp_bbox_xyxy(inst.head_rect_xyxy, w, h)
        points = [get("nose"), get("left_eye"), get("right_eye"), get("left_ear"), get("right_ear")]
        return keypoints_to_bbox(points, w, h, min_points=2, margin=margin_head, extra_top=margin_head)

    if part_name == "torso":
        points = [get("left_shoulder"), get("right_shoulder"), get("left_hip"), get("right_hip")]
        return keypoints_to_bbox(points, w, h, min_points=2, margin=margin_torso)

    if part_name in ("left_arm", "right_arm"):
        side = "left" if part_name.startswith("left") else "right"
        points = [get(f"{side}_shoulder"), get(f"{side}_elbow"), get(f"{side}_wrist")]
        valid = [(x, y) for x, y, v in points if v > 0]
        if len(valid) < 2:
            return None

        pts = np.array(valid, dtype=np.float32)
        xs = pts[:, 0]
        ys = pts[:, 1]
        x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

        max_len = 1.0
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                d = float(np.hypot(valid[i][0] - valid[j][0], valid[i][1] - valid[j][1]))
                max_len = max(max_len, d)

        pad = max_len * (margin_arm * 0.5)
        return clamp_bbox_xyxy((x1 - pad, y1 - pad, x2 + pad, y2 + pad), w, h)

    if part_name in ("left_hand", "right_hand"):
        side = "left" if part_name.startswith("left") else "right"
        wx, wy, wv = get(f"{side}_wrist")
        ex, ey, ev = get(f"{side}_elbow")
        if wv <= 0 or ev <= 0:
            return None
        seg = float(np.hypot(wx - ex, wy - ey))
        side_len = max(8.0, seg * 0.35)
        half = side_len * 0.5
        return clamp_bbox_xyxy((wx - half, wy - half, wx + half, wy + half), w, h)

    return None


def _letterbox_square(image: np.ndarray, target_size: int = 640) -> Tuple[np.ndarray, float, int, int]:
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Invalid image size for letterbox")

    scale = min(target_size / float(w), target_size / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def _transform_bbox_to_crop_and_letterbox(
    bbox_xyxy: Tuple[float, float, float, float],
    crop_xyxy: Tuple[int, int, int, int],
    scale: float,
    pad_x: int,
    pad_y: int,
    target_size: int,
) -> Optional[Tuple[float, float, float, float]]:
    bx1, by1, bx2, by2 = bbox_xyxy
    cx1, cy1, cx2, cy2 = crop_xyxy

    # to crop coordinates
    x1 = (bx1 - cx1) * scale + pad_x
    y1 = (by1 - cy1) * scale + pad_y
    x2 = (bx2 - cx1) * scale + pad_x
    y2 = (by2 - cy1) * scale + pad_y

    x1 = max(0.0, min(float(target_size - 1), x1))
    y1 = max(0.0, min(float(target_size - 1), y1))
    x2 = max(0.0, min(float(target_size - 1), x2))
    y2 = max(0.0, min(float(target_size - 1), y2))

    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def crop_and_relabel(
    inst: PersonInstance,
    out_image_path: Path,
    out_label_path: Path,
    margin_head: float,
    margin_torso: float,
    margin_arm: float,
    person_margin: float,
    target_size: int = 640,
) -> Tuple[int, Dict[str, int]]:
    image = cv2.imread(str(inst.image_path))
    if image is None:
        return 0, {name: 0 for name in PART_NAMES}

    bbox = expand_bbox(inst.person_bbox_xyxy, margin=person_margin, img_w=inst.image_w, img_h=inst.image_h)
    if bbox is None:
        return 0, {name: 0 for name in PART_NAMES}

    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, min(inst.image_w - 1, x1))
    y1 = max(0, min(inst.image_h - 1, y1))
    x2 = max(x1 + 1, min(inst.image_w, x2))
    y2 = max(y1 + 1, min(inst.image_h, y2))

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return 0, {name: 0 for name in PART_NAMES}

    letterboxed, scale, pad_x, pad_y = _letterbox_square(crop, target_size=target_size)

    labels: List[Tuple[int, Tuple[float, float, float, float]]] = []
    part_counts = {name: 0 for name in PART_NAMES}

    for class_id, part_name in enumerate(PART_NAMES):
        part_bbox = _part_bbox_from_instance(
            inst,
            part_name=part_name,
            margin_head=margin_head,
            margin_torso=margin_torso,
            margin_arm=margin_arm,
        )
        if part_bbox is None:
            continue

        transformed = _transform_bbox_to_crop_and_letterbox(
            bbox_xyxy=part_bbox,
            crop_xyxy=(x1, y1, x2, y2),
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            target_size=target_size,
        )
        if transformed is None:
            continue

        labels.append((class_id, transformed))
        part_counts[part_name] += 1

    out_image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_image_path), letterboxed)
    save_yolo_labels_multi(out_label_path, labels, w=target_size, h=target_size)
    return len(labels), part_counts


def _safe_name(path: Path) -> str:
    return path.stem.replace(" ", "_")


def _write_report_csv(report_path: Path, rows: List[Dict[str, Any]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["stage", "source", "class", "count"]
    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = args.output_dir
    person_stage = output_dir / "person_stage"
    parts_stage = output_dir / "parts_stage"

    for root in [person_stage, parts_stage]:
        (root / "images" / "train").mkdir(parents=True, exist_ok=True)
        (root / "images" / "val").mkdir(parents=True, exist_ok=True)
        (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    print("[INFO] Converting COCO...")
    coco_instances = convert_coco(args.coco_dir)
    print(f"[INFO] COCO instances: {len(coco_instances)}")

    print("[INFO] Converting MPII...")
    mpii_instances = convert_mpii(args.mpii_dir)
    print(f"[INFO] MPII instances: {len(mpii_instances)}")

    all_instances = coco_instances + mpii_instances
    if not all_instances:
        print("[ERROR] Nessuna istanza trovata. Verifica i path input.")
        return 1

    split = _split_instances(all_instances, val_fraction=args.val_fraction, seed=args.seed)

    stage1_counts = defaultdict(int)
    stage2_counts = defaultdict(int)
    skipped_parts = defaultdict(int)

    # Stage 1 + Stage 2 generation in one pass.
    for split_name in ["train", "val"]:
        instances = split[split_name]
        for idx, inst in enumerate(tqdm(instances, desc=f"Generate {split_name}", unit="person")):
            stem = f"{inst.source}_{_safe_name(inst.image_path)}_{idx:08d}"
            image_suffix = inst.image_path.suffix.lower() if inst.image_path.suffix else ".jpg"

            # Stage 1
            s1_img = person_stage / "images" / split_name / f"{stem}{image_suffix}"
            s1_lbl = person_stage / "labels" / split_name / f"{stem}.txt"
            _copy_or_symlink(inst.image_path, s1_img, use_symlink=args.symlink_images)
            save_yolo_label(s1_lbl, 0, inst.person_bbox_xyxy, inst.image_w, inst.image_h)
            stage1_counts[(inst.source, "person")] += 1

            # Stage 2
            s2_img = parts_stage / "images" / split_name / f"{stem}.jpg"
            s2_lbl = parts_stage / "labels" / split_name / f"{stem}.txt"
            n_parts, part_count_map = crop_and_relabel(
                inst,
                out_image_path=s2_img,
                out_label_path=s2_lbl,
                margin_head=args.margin_head,
                margin_torso=args.margin_torso,
                margin_arm=args.margin_arm,
                person_margin=args.person_margin,
                target_size=640,
            )

            if n_parts == 0:
                skipped_parts[inst.source] += 1

            for part_name, c in part_count_map.items():
                if c > 0:
                    stage2_counts[(inst.source, part_name)] += c

    _write_stage_yaml(person_stage, ["person"])
    _write_stage_yaml(parts_stage, PART_NAMES)

    print("[INFO] Istanze scartate stage2 (nessuna parte valida):")
    for source in sorted({k for k in skipped_parts.keys()} | {"coco", "mpii"}):
        print(f"  - {source}: {skipped_parts[source]}")

    report_rows: List[Dict[str, Any]] = []
    for (source, cls_name), count in sorted(stage1_counts.items()):
        report_rows.append({"stage": "person_stage", "source": source, "class": cls_name, "count": count})
    for (source, cls_name), count in sorted(stage2_counts.items()):
        report_rows.append({"stage": "parts_stage", "source": source, "class": cls_name, "count": count})

    report_path = output_dir / "final_report.csv"
    _write_report_csv(report_path, report_rows)
    print(f"[INFO] Report scritto in: {report_path}")
    print(f"[INFO] Person stage: {person_stage}")
    print(f"[INFO] Parts stage:  {parts_stage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
