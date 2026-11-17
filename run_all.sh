#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GS="$ROOT/gaussian-splatting"
DATA="$ROOT/data/garden"
MODELS="$ROOT/models"
OUT="$ROOT/outputs/final"
SOL="$ROOT/solution"
PLY="$MODELS/garden/point_cloud/iteration_30000/point_cloud.ply"

cd "$SOL"
python segment_views.py --source "$DATA" --images "$DATA/images_4" --out "$OUT/seg"

python select_gaussians.py --source "$DATA" --ply "$PLY" \
  --masks "$OUT/seg/masks" --out "$OUT/select" \
  --model_in "$MODELS/garden" --model_out "$MODELS/garden_removed"

python refine_removal.py --source "$DATA" --ply "$PLY" \
  --removed "$OUT/select/removed_idx.npy" --out "$OUT/refine" \
  --model_in "$MODELS/garden" --model_out "$MODELS/garden_removed2"

python replace_plate.py --source "$DATA" --masks "$OUT/seg/masks" \
  --model_in "$MODELS/garden_removed2" --model_out "$MODELS/garden_final" \
  --plane "$OUT/refine/plane.json" --out "$OUT/plate" \
  --mask_dilate 20 --loss l2

cd "$GS"
for m in garden garden_final; do
  python render.py -m "$MODELS/$m" -s "$DATA" -r 2 --skip_test
done

cd "$SOL"
python compare_views.py \
  --columns gt="$MODELS/garden/train/ours_30000/gt" \
            before="$MODELS/garden/train/ours_30000/renders" \
            after="$MODELS/garden_final/train/ours_30000/renders" \
  --out "$OUT/compare/final.jpg"

mkdir -p "$OUT/video"
ffmpeg -y -framerate 12 -i "$MODELS/garden/train/ours_30000/renders/%05d.png" \
       -framerate 12 -i "$MODELS/garden_final/train/ours_30000/renders/%05d.png" \
       -filter_complex "[0:v][1:v]hstack=inputs=2,scale=trunc(iw/2)*2:trunc(ih/2)*2" \
       -c:v libx264 -pix_fmt yuv420p "$OUT/video/before_after.mp4"
echo "done"