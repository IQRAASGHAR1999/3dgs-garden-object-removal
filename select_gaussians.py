import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from gs_utils import (camera_center_and_axis, camera_matrices, closest_point_to_axes,
                      load_cameras, project)


def vote(xyz, cams, mask_dir, near, chunk, area_tol):
    masks = {}
    for cam in cams:
        name = Path(cam.image_name).stem
        m = cv2.imread(str(mask_dir / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            print(f"no mask for {name}, skipped")
            continue
        masks[name] = m > 127

    areas = np.array([m.sum() for m in masks.values() if m.any()])
    med = float(np.median(areas))
    lo, hi = med / area_tol, med * area_tol
    print(f"median mask area {med:.0f}, keeping masks between {lo:.0f} and {hi:.0f} pixels")

    seen = np.zeros(len(xyz), np.uint16)
    hit = np.zeros(len(xyz), np.uint16)
    centers, axes, used, rejected = [], [], 0, []
    for cam in cams:
        name = Path(cam.image_name).stem
        fg = masks.get(name)
        if fg is None:
            continue
        h, w = fg.shape
        w2c, K = camera_matrices(cam, w, h)
        c, a = camera_center_and_axis(w2c)
        centers.append(c)
        axes.append(a)
        area = fg.sum()
        if area == 0 or area < lo or area > hi:
            rejected.append(name)
            continue
        used += 1
        for s in range(0, len(xyz), chunk):
            u, v, z = project(xyz[s:s + chunk], w2c, K)
            ui = np.rint(u).astype(np.int64)
            vi = np.rint(v).astype(np.int64)
            ok = (z > near) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
            idx = np.flatnonzero(ok)
            seen[s + idx] += 1
            hit[s + idx] += fg[vi[idx], ui[idx]].astype(np.uint16)
    print(f"rejected {len(rejected)} masks: {rejected}")
    return seen, hit, np.array(centers), np.array(axes), used


def pick_cluster(points, focus, eps, min_samples, min_cluster):
    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(points)
    stats = []
    for lab in np.unique(labels):
        if lab < 0:
            continue
        member = labels == lab
        n = int(member.sum())
        if n < min_cluster:
            continue
        d = float(np.linalg.norm(points[member].mean(0) - focus))
        stats.append((lab, n, d))
    stats.sort(key=lambda t: -t[1])
    for lab, n, d in stats[:10]:
        print(f"  cluster {lab}: {n} points, {d:.3f} from scene centre")
    if not stats:
        return None
    best = min(stats, key=lambda t: t[2])[0]
    return labels == best


def write_ply(vertices, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--images", default="images_4")
    p.add_argument("--ply", required=True)
    p.add_argument("--masks", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model_in", default="")
    p.add_argument("--model_out", default="")
    p.add_argument("--iteration", type=int, default=30000)
    p.add_argument("--ratio", type=float, default=0.7)
    p.add_argument("--ratio_grow", type=float, default=0.5)
    p.add_argument("--min_views", type=int, default=10)
    p.add_argument("--eps_frac", type=float, default=0.01)
    p.add_argument("--min_samples", type=int, default=10)
    p.add_argument("--min_cluster", type=int, default=500)
    p.add_argument("--near", type=float, default=0.2)
    p.add_argument("--chunk", type=int, default=2_000_000)
    p.add_argument("--area_tol", type=float, default=2.5)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    vert = PlyData.read(args.ply)["vertex"].data
    xyz = np.stack([vert["x"], vert["y"], vert["z"]], 1).astype(np.float64)
    print(f"{len(xyz)} gaussians")

    cams = load_cameras(args.source, args.images)
    seen, hit, centers, axes, used = vote(xyz, cams, Path(args.masks), args.near,
                                          args.chunk, args.area_tol)
    print(f"{len(cams)} cameras, {used} masks used in the vote")

    ratio = hit / np.maximum(seen, 1)
    enough = seen >= args.min_views
    cand = np.flatnonzero(enough & (ratio >= args.ratio))
    print(f"{len(cand)} candidates at ratio >= {args.ratio}")
    if len(cand) == 0:
        raise SystemExit("no candidates, lower --ratio or --min_views or check the masks")

    focus = closest_point_to_axes(centers, axes)
    radius = float(np.median(np.linalg.norm(centers - focus, axis=1)))
    eps = args.eps_frac * radius
    print(f"scene centre {focus.round(3)}, camera radius {radius:.3f}, eps {eps:.4f}")

    member = pick_cluster(xyz[cand], focus, eps, args.min_samples, args.min_cluster)
    if member is None:
        raise SystemExit("no cluster found, try a larger --eps_frac or smaller --min_cluster")
    core = cand[member]

    # nearby gaussians the stricter vote missed, mostly thin leaves and edges
    loose = np.flatnonzero(enough & (ratio >= args.ratio_grow))
    dist, _ = cKDTree(xyz[core]).query(xyz[loose], k=1, distance_upper_bound=eps)
    grown = loose[np.isfinite(dist)]
    removed = np.union1d(core, grown)
    print(f"core {len(core)}, after growing {len(removed)}")

    keep = np.ones(len(xyz), bool)
    keep[removed] = False
    np.save(out / "removed_idx.npy", removed)
    write_ply(vert[removed], out / "removed_only.ply")

    with open(out / "selection_stats.json", "w") as f:
        json.dump({"gaussians": int(len(xyz)), "cameras": len(cams), "views_used": used,
                   "candidates": int(len(cand)), "core": int(len(core)),
                   "removed": int(len(removed)), "scene_centre": focus.tolist(),
                   "camera_radius": radius, "eps": eps, "params": vars(args)}, f, indent=2)

    if args.model_out:
        src, dst = Path(args.model_in), Path(args.model_out)
        dst.mkdir(parents=True, exist_ok=True)
        for name in ("cfg_args", "cameras.json"):
            shutil.copy(src / name, dst / name)
        write_ply(vert[keep], dst / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply")
        print(f"edited model written to {dst}")


if __name__ == "__main__":
    main()
    