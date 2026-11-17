import argparse
from pathlib import Path

import cv2
import numpy as np


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (len(text) * 11 + 10, 26), (0, 0, 0), -1)
    cv2.putText(out, text, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--columns", nargs="+", required=True, help="name=folder pairs")
    p.add_argument("--index", nargs="+", type=int, default=[0, 40, 80, 120, 160])
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cols = [c.split("=", 1) for c in args.columns]
    rows = []
    for i in args.index:
        tiles = []
        for name, folder in cols:
            img = cv2.imread(str(Path(folder) / f"{i:05d}.png"))
            if img is None:
                raise SystemExit(f"missing {folder}/{i:05d}.png")
            h = int(round(img.shape[0] * args.width / img.shape[1]))
            tiles.append(label(cv2.resize(img, (args.width, h)), f"{name} {i:05d}"))
        rows.append(np.hstack(tiles))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, np.vstack(rows))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()