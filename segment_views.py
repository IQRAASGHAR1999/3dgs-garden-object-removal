import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                          SamModel, SamProcessor)

from gs_utils import (camera_center_and_axis, camera_matrices, closest_point_to_axes,
                      load_cameras, project)

GDINO_ID = "IDEA-Research/grounding-dino-tiny"
SAM_ID = "facebook/sam-vit-base"


def detect(processor, model, image, prompt, box_thr, text_thr, device):
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    kwargs = dict(text_threshold=text_thr, target_sizes=[image.size[::-1]])
    try:
        res = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=box_thr, **kwargs)
    except TypeError:
        res = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=box_thr, **kwargs)
    res = res[0]
    return res["boxes"].cpu().numpy(), res["scores"].cpu().numpy()


def choose_box(boxes, scores, anchor, margin):
    ax, ay = anchor
    best = None
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        mx, my = margin * (x1 - x0), margin * (y1 - y0)
        if x0 - mx <= ax <= x1 + mx and y0 - my <= ay <= y1 + my:
            if best is None or scores[i] > scores[best]:
                best = i
    return best


def segment(processor, model, image, box, device):
    inputs = processor(image, input_boxes=[[box.tolist()]], return_tensors="pt").to(device)
    inputs["input_boxes"] = inputs["input_boxes"].float()
    with torch.no_grad():
        out = model(**inputs)
    masks = processor.image_processor.post_process_masks(
        out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu())[0][0]
    scores = out.iou_scores[0, 0].cpu().numpy()
    return masks.numpy().astype(bool), scores


def save_overlay(path, image, mask, box, anchor):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    tint = img.copy()
    tint[mask] = (0, 0, 255)
    out = cv2.addWeighted(img, 0.55, tint, 0.45, 0)
    if box is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.circle(out, (int(round(anchor[0])), int(round(anchor[1]))), 6, (0, 255, 255), -1)
    cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, 85])


def contact_sheet(overlay_dir, out_path, cols=10, thumb_w=240):
    files = sorted(overlay_dir.glob("*.jpg"))
    if not files:
        return
    thumbs = []
    for f in files:
        im = cv2.imread(str(f))
        h = int(round(im.shape[0] * thumb_w / im.shape[1]))
        im = cv2.resize(im, (thumb_w, h))
        cv2.putText(im, f.stem, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        thumbs.append(im)
    h = thumbs[0].shape[0]
    rows = math.ceil(len(thumbs) / cols)
    sheet = np.zeros((rows * h, cols * thumb_w, 3), np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet[r * h:r * h + t.shape[0], c * thumb_w:(c + 1) * thumb_w] = t[:h]
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


def scene_focus(cams):
    centers, axes = [], []
    for cam in cams.values():
        w2c, _ = camera_matrices(cam, 1, 1)
        c, a = camera_center_and_axis(w2c)
        centers.append(c)
        axes.append(a)
    return closest_point_to_axes(np.array(centers), np.array(axes))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--prompt", default="a potted plant.|a vase.")
    p.add_argument("--box_thr", type=float, default=0.2)
    p.add_argument("--text_thr", type=float, default=0.25)
    p.add_argument("--anchor_margin", type=float, default=0.15)
    p.add_argument("--dilate", type=int, default=7)
    p.add_argument("--choice", choices=["score", "largest"], default="score")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    mask_dir, ov_dir = out / "masks", out / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    ov_dir.mkdir(parents=True, exist_ok=True)

    cams = {Path(c.image_name).stem: c for c in load_cameras(args.source, Path(args.images).name)}
    focus = scene_focus(cams)
    print(f"scene focus {focus.round(3)}")

    gd_proc = AutoProcessor.from_pretrained(GDINO_ID)
    gd = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_ID).to(device).eval()
    sam_proc = SamProcessor.from_pretrained(SAM_ID)
    sam = SamModel.from_pretrained(SAM_ID).to(device).eval()

    paths = sorted(f for f in Path(args.images).iterdir()
                   if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if args.limit:
        paths = paths[:args.limit]

    kernel = None
    if args.dilate > 0:
        k = 2 * args.dilate + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    prompts = [s.strip() for s in args.prompt.split("|") if s.strip()]
    log = {}
    for path in paths:
        image = Image.open(path).convert("RGB")
        w2c, K = camera_matrices(cams[path.stem], image.width, image.height)
        u, v, _ = project(focus[None], w2c, K)
        anchor = (float(u[0]), float(v[0]))

        mask = np.zeros((image.height, image.width), bool)
        parts, box = [], None
        for prompt in prompts:
            boxes, scores = detect(gd_proc, gd, image, prompt, args.box_thr, args.text_thr, device)
            i = choose_box(boxes, scores, anchor, args.anchor_margin) if len(scores) else None
            if i is None:
                parts.append({"prompt": prompt, "box": None})
                continue
            masks, sam_scores = segment(sam_proc, sam, image, boxes[i], device)
            if args.choice == "largest":
                k = int(masks.reshape(len(masks), -1).sum(1).argmax())
            else:
                k = int(sam_scores.argmax())
            mask |= masks[k]
            b = boxes[i]
            box = b.copy() if box is None else np.array(
                [min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3])])
            parts.append({"prompt": prompt, "box": [float(x) for x in b],
                          "det_score": float(scores[i]), "sam_score": float(sam_scores[k])})

        m = mask.astype(np.uint8) * 255
        if kernel is not None and m.any():
            m = cv2.dilate(m, kernel)
        entry = {"anchor": list(anchor), "box": None if box is None else [float(x) for x in box],
                 "area": int((m > 0).sum()), "parts": parts}
        cv2.imwrite(str(mask_dir / f"{path.stem}.png"), m)
        save_overlay(ov_dir / f"{path.stem}.jpg", image, m > 0, box, anchor)
        log[path.stem] = entry
        found = sum(pp["box"] is not None for pp in parts)
        print(f"{path.stem}: {found}/{len(prompts)} prompts found, area={entry['area']}")

    with open(out / "segmentation_log.json", "w") as f:
        json.dump({"prompts": prompts, "box_thr": args.box_thr, "text_thr": args.text_thr,
                   "anchor_margin": args.anchor_margin, "dilate": args.dilate,
                   "choice": args.choice, "focus": focus.tolist(), "views": log}, f, indent=2)
    contact_sheet(ov_dir, out / "overlay_sheet.jpg")
    missed = [k for k, v in log.items() if v["box"] is None]
    print(f"done, {len(log)} views, empty mask in {len(missed)}: {missed}")


if __name__ == "__main__":
    main()