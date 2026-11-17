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


def radial(points, origin, axis):
    rel = points - origin
    h = rel @ axis
    r = np.linalg.norm(rel - np.outer(h, axis), axis=1)
    return h, r


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--images", default="images_4")
    p.add_argument("--model_in", required=True)
    p.add_argument("--model_out", required=True)
    p.add_argument("--plane", required=True)
    p.add_argument("--fill", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--iteration", type=int, default=30000)
    p.add_argument("--downscale", type=int, default=2)
    p.add_argument("--crop", type=float, default=4.0)
    p.add_argument("--crop_height", type=float, default=0.5)
    p.add_argument("--target", choices=["flat", "inpaint"], default="flat")
    p.add_argument("--ring_in", type=float, default=1.1)
    p.add_argument("--ring_out", type=float, default=1.35)
    p.add_argument("--target_dilate", type=int, default=5)
    p.add_argument("--inpaint_radius", type=int, default=9)
    p.add_argument("--min_pixels", type=int, default=200)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--lr", type=float, default=0.0025)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    plane = json.load(open(args.plane))
    fill = json.load(open(args.fill))
    n = np.array(plane["normal"])
    foot = np.array(plane["foot"])
    r_foot, r_fill, n_new = plane["r_foot"], fill["r_fill"], fill["added"]
    centre = foot + n * fill.get("height_offset", 0.0)

    src = Path(args.model_in)
    vert = PlyData.read(str(src / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"))["vertex"].data
    xyz = np.stack([vert["x"], vert["y"], vert["z"]], 1).astype(np.float64)
    hp, rr = radial(xyz, foot, n)
    crop = (rr < args.crop * r_fill) & (np.abs(hp) < args.crop_height * r_foot)
    crop[-n_new:] = True
    crop_idx = np.flatnonzero(crop)
    n_old = len(crop_idx) - n_new
    print(f"crop holds {len(crop_idx)} gaussians, {n_new} of them are the fill")

    infos = load_cameras(args.source, args.images)
    first = next(Path(args.source, args.images).iterdir())
    h0, w0 = cv2.imread(str(first)).shape[:2]
    width, height = round(w0 / args.downscale), round(h0 / args.downscale)

    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False,
                           debug=False, antialiasing=False)
    bg = torch.zeros(3, device="cuda")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * args.target_dilate + 1,) * 2)

    # per view target: the scene without the fill, with the hole set to the plate's own look in that view
    base = load_gaussians(vert[crop_idx[:n_old]], out / "crop_base.ply")
    views = []
    for info in infos:
        cam, w2c, K = make_camera(info, width, height)
        mask = disk_in_view(w2c, K, width, height, centre, n, r_fill)
        if mask.sum() < args.min_pixels:
            continue
        with torch.no_grad():
            img = to_uint8(render(cam, base, pipe, bg)["render"])
        if args.target == "flat":
            ring = (disk_in_view(w2c, K, width, height, centre, n, args.ring_out * r_fill)
                    & ~disk_in_view(w2c, K, width, height, centre, n, args.ring_in * r_fill))
            if ring.sum() < args.min_pixels:
                continue
            target = img.copy()
            target[mask] = np.median(img[ring], 0).astype(np.uint8)
        else:
            m = cv2.dilate(mask.astype(np.uint8) * 255, kernel)
            target = cv2.inpaint(img, m, args.inpaint_radius, cv2.INPAINT_TELEA)
        views.append((cam, torch.tensor(target, dtype=torch.float32).permute(2, 0, 1) / 255.0,
                      torch.tensor(mask, dtype=torch.float32)[None]))
    del base
    torch.cuda.empty_cache()
    print(f"{len(views)} views see the filled area")

    gs = load_gaussians(vert[crop_idx], out / "crop_full.ply")
    dc_all, rest_all = gs._features_dc.detach(), gs._features_rest.detach()
    new_dc = torch.nn.Parameter(dc_all[n_old:].clone())
    new_rest = torch.nn.Parameter(rest_all[n_old:].clone())
    opt = torch.optim.Adam([{"params": [new_dc], "lr": args.lr},
                            {"params": [new_rest], "lr": args.lr / 20.0}])

    for it in range(args.iters):
        cam, target, mask = random.choice(views)
        target, mask = target.cuda(), mask.cuda()
        gs._features_dc = torch.cat([dc_all[:n_old], new_dc], 0)
        gs._features_rest = torch.cat([rest_all[:n_old], new_rest], 0)
        img = render(cam, gs, pipe, bg)["render"]
        loss = (torch.abs(img - target) * mask).sum() / (mask.sum() * 3.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 200 == 0 or it == args.iters - 1:
            print(f"iter {it}: masked L1 {loss.item():.4f}")

    gs._features_dc = torch.cat([dc_all[:n_old], new_dc.detach()], 0)
    gs._features_rest = torch.cat([rest_all[:n_old], new_rest.detach()], 0)
    for k, (cam, target, mask) in enumerate(views[::max(1, len(views) // 4)][:4]):
        with torch.no_grad():
            img = to_uint8(render(cam, gs, pipe, bg)["render"])
        pair = np.hstack([to_uint8(target), img])
        cv2.imwrite(str(out / f"check_{k}.png"), cv2.cvtColor(pair, cv2.COLOR_RGB2BGR))

    final = vert.copy()
    dc = new_dc.detach()[:, 0, :].cpu().numpy()
    rest = new_rest.detach().transpose(1, 2).reshape(n_new, -1).cpu().numpy()
    for i in range(3):
        final[f"f_dc_{i}"][-n_new:] = dc[:, i]
    for j in range(rest.shape[1]):
        final[f"f_rest_{j}"][-n_new:] = rest[:, j]

    dst = Path(args.model_out)
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("cfg_args", "cameras.json"):
        shutil.copy(src / name, dst / name)
    write_ply(final, dst / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply")
    for tmp in ("crop_base.ply", "crop_full.ply"):
        (out / tmp).unlink(missing_ok=True)
    print(f"refined model written to {dst}")


if __name__ == "__main__":
    main()