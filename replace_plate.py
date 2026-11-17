import argparse
import json
import random
import shutil
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from plyfile import PlyData, PlyElement

from gs_utils import load_cameras
from gaussian_renderer import render
from scene.cameras import MiniCam
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import getProjectionMatrix, getWorld2View2

C0 = 0.28209479177387814


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def radial(points, origin, axis):
    rel = points - origin
    h = rel @ axis
    r = np.linalg.norm(rel - np.outer(h, axis), axis=1)
    return h, r


def plane_basis(n):
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    a = np.cross(n, helper)
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    return a, b


def rot_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def write_ply(vertices, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(path))


def make_camera(info, width, height):
    w2c = getWorld2View2(info.R, info.T).astype(np.float64)
    wv = torch.tensor(w2c, dtype=torch.float32).transpose(0, 1).cuda()
    proj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=info.FovX, fovY=info.FovY).transpose(0, 1).cuda()
    full = wv.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    cam = MiniCam(width, height, info.FovY, info.FovX, 0.01, 100.0, wv, full)
    fx = width / (2.0 * np.tan(info.FovX / 2.0))
    fy = height / (2.0 * np.tan(info.FovY / 2.0))
    K = np.array([[fx, 0.0, (width - 1) / 2.0], [0.0, fy, (height - 1) / 2.0], [0.0, 0.0, 1.0]])
    return cam, w2c, K


def disk_in_view(w2c, K, width, height, centre, n, radius):
    c2w = np.linalg.inv(w2c)
    ys, xs = np.mgrid[0:height, 0:width]
    d = np.stack([(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1],
                  np.ones_like(xs, float)], -1).reshape(-1, 3) @ c2w[:3, :3].T
    o = c2w[:3, 3]
    denom = d @ n
    t = ((centre - o) @ n) / np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    rel = o + d * t[:, None] - centre
    rel_in = rel - np.outer(rel @ n, n)
    inside = (t > 0) & (np.linalg.norm(rel_in, axis=1) < radius)
    return inside.reshape(height, width)


def load_gaussians(vertices, tmp_path):
    write_ply(vertices, tmp_path)
    gs = GaussianModel(3)
    gs.load_ply(str(tmp_path))
    for t in (gs._xyz, gs._features_dc, gs._features_rest, gs._scaling, gs._rotation, gs._opacity):
        t.requires_grad_(False)
    return gs


def to_uint8(img):
    return (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def find_plate_radius(sat, rr, start, stop, step, sat_thr, frac_thr):
    profile = []
    for lo in np.arange(start, stop, step):
        sel = (rr >= lo) & (rr < lo + step)
        frac = float((sat[sel] < sat_thr).mean()) if sel.any() else 0.0
        profile.append((lo, int(sel.sum()), frac))
    for lo, cnt, frac in profile:
        if cnt > 20 and frac < frac_thr:
            return lo, profile
    return stop, profile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--images", default="images_4")
    p.add_argument("--masks", required=True)
    p.add_argument("--model_in", required=True)
    p.add_argument("--model_out", required=True)
    p.add_argument("--plane", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--iteration", type=int, default=30000)
    p.add_argument("--plate_radius", type=float, default=0.0)
    p.add_argument("--search_start", type=float, default=1.2)
    p.add_argument("--search_stop", type=float, default=6.0)
    p.add_argument("--search_step", type=float, default=0.1)
    p.add_argument("--sat_thr", type=float, default=0.22)
    p.add_argument("--frac_thr", type=float, default=0.5)
    p.add_argument("--band", type=float, default=0.5)
    p.add_argument("--grid", type=int, default=80)
    p.add_argument("--overlap", type=float, default=0.9)
    p.add_argument("--thin", type=float, default=0.05)
    p.add_argument("--opacity", type=float, default=0.99)
    p.add_argument("--downscale", type=int, default=2)
    p.add_argument("--edge", type=float, default=0.97)
    p.add_argument("--mask_dilate", type=int, default=6)
    p.add_argument("--min_pixels", type=int, default=200)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.0025)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--loss", choices=["l1", "l2"], default="l2")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    plane = json.load(open(args.plane))
    n = np.array(plane["normal"])
    foot = np.array(plane["foot"])
    r_foot, top = plane["r_foot"], plane["top"]

    src = Path(args.model_in)
    vert = PlyData.read(str(src / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"))["vertex"].data
    xyz = np.stack([vert["x"], vert["y"], vert["z"]], 1).astype(np.float64)
    hp, rr = radial(xyz, foot, n)
    rgb = np.clip(0.5 + C0 * np.stack([vert[f"f_dc_{i}"] for i in range(3)], 1), 0, 1)
    sat = (rgb.max(1) - rgb.min(1)) / np.maximum(rgb.max(1), 1e-6)
    solid = (np.abs(hp) < args.band * r_foot) & (sigmoid(vert["opacity"].astype(np.float64)) > 0.5)

    # the plate is grey, the wood around it is brown, so its edge shows up in the colour profile
    if args.plate_radius > 0:
        R = args.plate_radius * r_foot
        profile = []
    else:
        R, profile = find_plate_radius(sat[solid], rr[solid], args.search_start * r_foot,
                                       args.search_stop * r_foot, args.search_step * r_foot,
                                       args.sat_thr, args.frac_thr)
    for lo, cnt, frac in profile:
        print(f"  r {lo / r_foot:4.1f} r_foot: {cnt:6d} gaussians, {frac:.2f} grey")
    print(f"plate radius {R:.4f} ({R / r_foot:.2f} r_foot)")

    plate_pts = solid & (rr > max(1.2 * r_foot, 0.4 * R)) & (rr < 0.95 * R) & (sat < args.sat_thr)
    h_off = float(np.median(hp[plate_pts]))
    sh_fields = [f for f in vert.dtype.names if f.startswith("f_dc_") or f.startswith("f_rest_")]
    init_sh = np.median(np.stack([vert[plate_pts][f] for f in sh_fields], 1), 0)
    print(f"{plate_pts.sum()} plate gaussians, plate height {h_off / r_foot:.3f} r_foot above the plane")

    clear = (rr < R) & (hp > -args.band * r_foot) & (hp < top)
    kept = vert[~clear]
    print(f"cleared {clear.sum()} gaussians on the plate")

    a, b = plane_basis(n)
    spacing = R / args.grid
    g = spacing * np.arange(-args.grid, args.grid + 1)
    gu, gv = np.meshgrid(g, g)
    inside = gu ** 2 + gv ** 2 <= R ** 2
    gu, gv = gu[inside], gv[inside]
    pos = foot + n * h_off + np.outer(gu, a) + np.outer(gv, b)
    n_new = len(pos)

    new = np.zeros(n_new, dtype=vert.dtype)
    new["x"], new["y"], new["z"] = pos[:, 0], pos[:, 1], pos[:, 2]
    for j, fname in enumerate(sh_fields):
        new[fname] = init_sh[j]
    new["opacity"] = np.log(args.opacity / (1.0 - args.opacity))
    s_in, s_n = np.log(args.overlap * spacing), np.log(args.thin * spacing)
    new["scale_0"], new["scale_1"], new["scale_2"] = s_in, s_in, s_n
    q = rot_to_quat(np.stack([a, b, n], 1))
    for i in range(4):
        new[f"rot_{i}"] = q[i]
    combined = np.concatenate([kept, new])
    print(f"added {n_new} gaussians covering the plate")

    cxyz = np.stack([combined["x"], combined["y"], combined["z"]], 1).astype(np.float64)
    chp, crr = radial(cxyz, foot, n)
    crop = (crr < 1.3 * R) & (np.abs(chp) < args.band * r_foot)
    crop[-n_new:] = True
    crop_idx = np.flatnonzero(crop)
    n_old = len(crop_idx) - n_new

    infos = load_cameras(args.source, args.images)
    img_dir = Path(args.source, args.images)
    first = next(img_dir.iterdir())
    h0, w0 = cv2.imread(str(first)).shape[:2]
    width, height = round(w0 / args.downscale), round(h0 / args.downscale)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * args.mask_dilate + 1,) * 2)
    centre = foot + n * h_off

    # supervision: real photo pixels of the plate, with the pot masked out
    views = []
    for info in infos:
        cam, w2c, K = make_camera(info, width, height)
        plate_px = disk_in_view(w2c, K, width, height, centre, n, args.edge * R)
        obj = cv2.imread(str(Path(args.masks) / f"{Path(info.image_name).stem}.png"), cv2.IMREAD_GRAYSCALE)
        if obj is not None:
            obj = cv2.resize(obj, (width, height), interpolation=cv2.INTER_NEAREST)
            plate_px &= cv2.dilate(obj, kernel) == 0
        if plate_px.sum() < args.min_pixels:
            continue
        photo = cv2.imread(str(img_dir / Path(info.image_name).name))
        photo = cv2.cvtColor(cv2.resize(photo, (width, height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        views.append((cam, torch.tensor(photo, dtype=torch.float32).permute(2, 0, 1) / 255.0,
                      torch.tensor(plate_px, dtype=torch.float32)[None]))
    print(f"{len(views)} views see the plate")

    gs = load_gaussians(combined[crop_idx], out / "crop.ply")
    dc_all, rest_all = gs._features_dc.detach(), gs._features_rest.detach()
    shared_dc = torch.nn.Parameter(dc_all[-1:].clone())
    shared_rest = torch.nn.Parameter(rest_all[-1:].clone())
    opt = torch.optim.Adam([{"params": [shared_dc], "lr": args.lr},
                            {"params": [shared_rest], "lr": args.lr / 20.0}])
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False,
                           debug=False, antialiasing=False)
    bg = torch.zeros(3, device="cuda")

    def apply():
        gs._features_dc = torch.cat([dc_all[:n_old], shared_dc.expand(n_new, -1, -1)], 0)
        gs._features_rest = torch.cat([rest_all[:n_old], shared_rest.expand(n_new, -1, -1)], 0)

    for it in range(args.iters):
        cam, photo, mask = random.choice(views)
        photo, mask = photo.cuda(), mask.cuda()
        apply()
        img = render(cam, gs, pipe, bg)["render"]
        
        diff = (img - photo) * mask
        if args.loss == "l2":
            loss = (diff ** 2).sum() / (mask.sum() * 3.0)
        else:
            loss = diff.abs().sum() / (mask.sum() * 3.0)    
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 100 == 0 or it == args.iters - 1:
            print(f"iter {it}: plate L1 {loss.item():.4f}")

    with torch.no_grad():
        apply()
        for k, (cam, photo, mask) in enumerate(views[::max(1, len(views) // 4)][:4]):
            img = to_uint8(render(cam, gs, pipe, bg)["render"])
            cv2.imwrite(str(out / f"check_{k}.png"),
                        cv2.cvtColor(np.hstack([to_uint8(photo), img]), cv2.COLOR_RGB2BGR))

    final = combined.copy()
    dc = shared_dc.detach()[0, 0].cpu().numpy()
    rest = shared_rest.detach()[0].transpose(0, 1).reshape(-1).cpu().numpy()
    for i in range(3):
        final[f"f_dc_{i}"][-n_new:] = dc[i]
    for j in range(len(rest)):
        final[f"f_rest_{j}"][-n_new:] = rest[j]

    dst = Path(args.model_out)
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("cfg_args", "cameras.json"):
        shutil.copy(src / name, dst / name)
    write_ply(final, dst / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply")
    (out / "crop.ply").unlink(missing_ok=True)
    with open(out / "plate.json", "w") as f:
        json.dump({"plate_radius": R, "plate_radius_rfoot": R / r_foot, "height_offset": h_off,
                   "plate_gaussians": int(plate_pts.sum()), "cleared": int(clear.sum()),
                   "added": n_new, "views": len(views), "params": vars(args)}, f, indent=2)
    print(f"model written to {dst}")


if __name__ == "__main__":
    main()