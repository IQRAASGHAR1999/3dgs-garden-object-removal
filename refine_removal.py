import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from gs_utils import camera_matrices, load_cameras


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def camera_up(cams):
    ups = []
    for cam in cams:
        w2c, _ = camera_matrices(cam, 1, 1)
        # colmap cameras look along +z with y pointing down, so -y is roughly world up
        ups.append(-np.linalg.inv(w2c)[:3, 1])
    up = np.mean(ups, 0)
    return up / np.linalg.norm(up)


def fit_plane(points, iters, thresh, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        nrm = np.cross(b - a, c - a)
        length = np.linalg.norm(nrm)
        if length < 1e-12:
            continue
        nrm /= length
        inl = np.abs((points - a) @ nrm) < thresh
        if best is None or inl.sum() > best.sum():
            best = inl
    pts = points[best]
    centre = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
    return vt[-1], centre, best


def radial(points, origin, axis):
    rel = points - origin
    h = rel @ axis
    r = np.linalg.norm(rel - np.outer(h, axis), axis=1)
    return h, r


def write_ply(vertices, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--images", default="images_4")
    p.add_argument("--ply", required=True)
    p.add_argument("--removed", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model_in", default="")
    p.add_argument("--model_out", default="")
    p.add_argument("--iteration", type=int, default=30000)
    p.add_argument("--ring_in", type=float, default=1.5)
    p.add_argument("--ring_out", type=float, default=4.0)
    p.add_argument("--band", type=float, default=0.5)
    p.add_argument("--opacity_min", type=float, default=0.5)
    p.add_argument("--ransac_iters", type=int, default=2000)
    p.add_argument("--ransac_thresh", type=float, default=0.05)
    p.add_argument("--max_fit_points", type=int, default=200000)
    p.add_argument("--col_margin", type=float, default=1.2)
    p.add_argument("--h_sigma", type=float, default=3.0)
    p.add_argument("--top_margin", type=float, default=1.1)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    vert = PlyData.read(args.ply)["vertex"].data
    xyz = np.stack([vert["x"], vert["y"], vert["z"]], 1).astype(np.float64)
    opacity = sigmoid(vert["opacity"].astype(np.float64))
    removed = np.load(args.removed)
    keep = np.ones(len(xyz), bool)
    keep[removed] = False

    up = camera_up(load_cameras(args.source, args.images))
    obj = xyz[removed]
    h_obj = obj @ up
    lo, hi = np.percentile(h_obj, [1, 99])
    base = obj[h_obj < lo + 0.3 * (hi - lo)]
    axis_pt = base.mean(0)
    _, r_base = radial(base, axis_pt, up)
    r_foot = float(np.percentile(r_base, 95))
    h_lo = float(np.percentile(radial(obj, axis_pt, up)[0], 1))
    print(f"object base at {axis_pt.round(3)}, footprint radius {r_foot:.4f}")

    # table points in a ring around the object, at roughly the height of its base
    kept = np.flatnonzero(keep)
    h, r = radial(xyz[kept], axis_pt, up)
    ring = kept[(r > args.ring_in * r_foot) & (r < args.ring_out * r_foot)
                & (np.abs(h - h_lo) < args.band * r_foot)
                & (opacity[kept] > args.opacity_min)]
    print(f"{len(ring)} ring points for the plane fit")
    if len(ring) < 100:
        raise SystemExit("too few ring points, widen --ring_out or --band")
    fit_pts = xyz[ring]
    if len(fit_pts) > args.max_fit_points:
        pick = np.random.default_rng(0).choice(len(fit_pts), args.max_fit_points, replace=False)
        fit_pts = fit_pts[pick]

    normal, centre, inl = fit_plane(fit_pts, args.ransac_iters, args.ransac_thresh * r_foot)
    if normal @ up < 0:
        normal = -normal
    angle = float(np.degrees(np.arccos(np.clip(normal @ up, -1.0, 1.0))))
    sigma = float(((fit_pts[inl] - centre) @ normal).std())
    h_eps = args.h_sigma * sigma
    print(f"plane: {inl.sum()} inliers of {len(fit_pts)}, {angle:.1f} deg from camera up, "
          f"residual std {sigma:.5f}")

    foot = axis_pt - ((axis_pt - centre) @ normal) * normal
    r_col = args.col_margin * r_foot
    top = args.top_margin * float(np.percentile((obj - foot) @ normal, 99))
    hp, rr = radial(xyz[kept], foot, normal)
    col = kept[(rr < r_col) & (hp > h_eps) & (hp < top)]
    removed_all = np.union1d(removed, col)
    print(f"column removal: {len(col)} more gaussians, {len(removed_all)} removed in total")

    keep_all = np.ones(len(xyz), bool)
    keep_all[removed_all] = False
    np.save(out / "removed_idx.npy", removed_all)
    write_ply(vert[col], out / "column_only.ply")
    with open(out / "plane.json", "w") as f:
        json.dump({"normal": normal.tolist(), "centre": centre.tolist(), "foot": foot.tolist(),
                   "up": up.tolist(), "angle_deg": angle, "sigma": sigma, "h_eps": h_eps,
                   "r_foot": r_foot, "r_col": r_col, "top": top,
                   "column_removed": int(len(col)), "removed_total": int(len(removed_all)),
                   "params": vars(args)}, f, indent=2)

    if args.model_out:
        src, dst = Path(args.model_in), Path(args.model_out)
        dst.mkdir(parents=True, exist_ok=True)
        for name in ("cfg_args", "cameras.json"):
            shutil.copy(src / name, dst / name)
        write_ply(vert[keep_all], dst / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply")
        print(f"edited model written to {dst}")


if __name__ == "__main__":
    main()