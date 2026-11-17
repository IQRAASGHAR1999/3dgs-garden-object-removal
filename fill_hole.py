import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

import gs_utils  # puts the 3DGS repo on the import path
from gaussian_renderer import render
from scene.cameras import MiniCam
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import getProjectionMatrix

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


def topdown_camera(centre, n, a, b, half, res, dist):
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2] = a, -b, -n
    c2w[:3, 3] = centre + n * dist
    w2c = np.linalg.inv(c2w)
    fov = 2.0 * np.arctan(half / dist)
    znear, zfar = 0.01, dist * 4.0
    wv = torch.tensor(w2c, dtype=torch.float32).transpose(0, 1).cuda()
    proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fov, fovY=fov).transpose(0, 1).cuda()
    full = wv.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    cam = MiniCam(res, res, fov, fov, znear, zfar, wv, full)
    f = res / (2.0 * np.tan(fov / 2.0))
    K = np.array([[f, 0.0, (res - 1) / 2.0], [0.0, f, (res - 1) / 2.0], [0.0, 0.0, 1.0]])
    return cam, c2w, w2c, K


def write_ply(vertices, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(path))


def render_vertices(vertices, cam, tmp_path):
    write_ply(vertices, tmp_path)
    gs = GaussianModel(3)
    gs.load_ply(str(tmp_path))
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False,
                           debug=False, antialiasing=False)
    bg = torch.zeros(3, device="cuda")
    with torch.no_grad():
        img = render(cam, gs, pipe, bg)["render"]
    out = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    del gs
    torch.cuda.empty_cache()
    return out


def disk_mask(res, c2w, K, foot, n, a, b, radius):
    ys, xs = np.mgrid[0:res, 0:res]
    d_cam = np.stack([(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1], np.ones_like(xs, float)], -1)
    d_world = d_cam.reshape(-1, 3) @ c2w[:3, :3].T
    origin = c2w[:3, 3]
    t = ((foot - origin) @ n) / (d_world @ n)
    rel = origin + d_world * t[:, None] - foot
    dist = np.sqrt((rel @ a) ** 2 + (rel @ b) ** 2)
    return (dist < radius).reshape(res, res)


def inpaint(img_rgb, mask, method):
    if method == "lama":
        from PIL import Image
        from simple_lama_inpainting import SimpleLama
        lama = SimpleLama()
        out = np.array(lama(Image.fromarray(img_rgb), Image.fromarray(mask)).convert("RGB"))
        return out[:img_rgb.shape[0], :img_rgb.shape[1]]
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(cv2.inpaint(bgr, mask, 7, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB)


def sample_bilinear(img, u, v):
    h, w = img.shape[:2]
    u = np.clip(u, 0, w - 1.001)
    v = np.clip(v, 0, h - 1.001)
    x0, y0 = np.floor(u).astype(int), np.floor(v).astype(int)
    dx, dy = (u - x0)[:, None], (v - y0)[:, None]
    im = img.astype(np.float64) / 255.0
    return (im[y0, x0] * (1 - dx) * (1 - dy) + im[y0, x0 + 1] * dx * (1 - dy)
            + im[y0 + 1, x0] * (1 - dx) * dy + im[y0 + 1, x0 + 1] * dx * dy)


def harmonic_fill(values, known, unknown):
    # discrete Laplace equation on the grid, known cells act as boundary values
    ys, xs = np.nonzero(unknown)
    n = len(ys)
    idx = -np.ones(unknown.shape, np.int64)
    idx[ys, xs] = np.arange(n)
    rows, cols, data = [], [], []
    rhs = np.zeros((n, values.shape[2]))
    diag = np.zeros(n)
    i_all = np.arange(n)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ny, nx = ys + dy, xs + dx
        nu = unknown[ny, nx]
        nk = known[ny, nx]
        diag += nu | nk
        rows.append(i_all[nu])
        cols.append(idx[ny[nu], nx[nu]])
        data.append(-np.ones(nu.sum()))
        rhs[nk] += values[ny[nk], nx[nk]]
    rows.append(i_all)
    cols.append(i_all)
    data.append(diag)
    A = coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(n, n)).tocsc()
    return splu(A).solve(rhs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_in", required=True)
    p.add_argument("--model_out", required=True)
    p.add_argument("--plane", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--iteration", type=int, default=30000)
    p.add_argument("--appearance", choices=["harmonic", "inpaint"], default="harmonic")
    p.add_argument("--inpaint", choices=["lama", "opencv"], default="lama")
    p.add_argument("--fill_scale", type=float, default=1.4)
    p.add_argument("--clear_band", type=float, default=0.15)
    p.add_argument("--donor_in", type=float, default=1.05)
    p.add_argument("--donor_out", type=float, default=1.6)
    p.add_argument("--donor_k", type=int, default=8)
    p.add_argument("--donor_opacity", type=float, default=0.5)
    p.add_argument("--no_match_height", action="store_true")
    p.add_argument("--pad", type=int, default=3)
    p.add_argument("--context", type=float, default=2.5)
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--dist", type=float, default=10.0)
    p.add_argument("--slab_below", type=float, default=0.5)
    p.add_argument("--slab_above", type=float, default=0.5)
    p.add_argument("--mask_dilate", type=int, default=3)
    p.add_argument("--spacing_px", type=float, default=1.5)
    p.add_argument("--overlap", type=float, default=0.7)
    p.add_argument("--thin", type=float, default=0.05)
    p.add_argument("--opacity", type=float, default=0.99)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(args.model_in)
    vert = PlyData.read(str(src / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"))["vertex"].data
    xyz = np.stack([vert["x"], vert["y"], vert["z"]], 1).astype(np.float64)

    plane = json.load(open(args.plane))
    n = np.array(plane["normal"])
    foot = np.array(plane["foot"])
    r_foot, top = plane["r_foot"], plane["top"]
    r_fill = args.fill_scale * r_foot

    hp, rr = radial(xyz, foot, n)
    clear = (rr < r_fill) & (hp > -args.clear_band * r_foot) & (hp < top)
    kept = vert[~clear]
    kxyz = xyz[~clear]
    khp, krr = radial(kxyz, foot, n)
    print(f"cleared {clear.sum()} gaussians inside the fill disk (radius {r_fill:.4f})")

    a, b = plane_basis(n)
    rel = kxyz - foot
    pu, pv = rel @ a, rel @ b
    donors = ((krr > args.donor_in * r_fill) & (krr < args.donor_out * r_fill)
              & (np.abs(khp) < args.clear_band * r_foot)
              & (sigmoid(kept["opacity"].astype(np.float64)) > args.donor_opacity))
    print(f"{donors.sum()} donor gaussians in the ring")
    if donors.sum() < args.donor_k:
        raise SystemExit("too few donors, widen --donor_out or --clear_band")

    # the plate can sit slightly above the wood the plane was fitted to
    h_off = 0.0 if args.no_match_height else float(np.median(khp[donors]))
    print(f"surface height around the hole: {h_off:.5f} ({h_off / r_foot:.3f} r_foot) above the plane")

    half = args.context * r_fill
    cam, c2w, w2c, K = topdown_camera(foot, n, a, b, half, args.res, args.dist * half)

    slab = (krr < 1.5 * half) & (khp > -args.slab_below * r_foot) & (khp < args.slab_above * r_foot)
    before = render_vertices(kept[slab], cam, out / "topdown_subset.ply")
    mask = disk_mask(args.res, c2w, K, foot, n, a, b, r_fill).astype(np.uint8) * 255
    if args.mask_dilate > 0:
        k = 2 * args.mask_dilate + 1
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    cv2.imwrite(str(out / "topdown_before.png"), cv2.cvtColor(before, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "topdown_mask.png"), mask)

    pixel = 2.0 * half / args.res
    spacing = args.spacing_px * pixel
    m = int(np.ceil(r_fill * 1.02 / spacing)) + args.pad
    g = spacing * np.arange(-m, m + 1)
    GU, GV = np.meshgrid(g, g)
    rad = np.sqrt(GU ** 2 + GV ** 2)
    interior = rad <= r_fill * 1.02
    band = (~interior) & (rad <= r_fill * 1.02 + args.pad * spacing)
    gu, gv = GU[interior], GV[interior]
    pos = foot + n * h_off + np.outer(gu, a) + np.outer(gv, b)

    new = np.zeros(len(pos), dtype=vert.dtype)
    sh_fields = [f for f in vert.dtype.names if f.startswith("f_dc_") or f.startswith("f_rest_")]

    if args.appearance == "harmonic":
        tree = cKDTree(np.stack([pu[donors], pv[donors]], 1))
        _, nn = tree.query(np.stack([GU[band], GV[band]], 1), k=args.donor_k)
        donor_sh = np.stack([kept[donors][f] for f in sh_fields], 1).astype(np.float64)
        values = np.zeros(GU.shape + (len(sh_fields),))
        values[band] = donor_sh[nn].mean(1)
        sol = harmonic_fill(values, band, interior)
        for j, fname in enumerate(sh_fields):
            new[fname] = sol[:, j]
    else:
        filled = inpaint(before, mask, args.inpaint)
        cv2.imwrite(str(out / "topdown_inpainted.png"), cv2.cvtColor(filled, cv2.COLOR_RGB2BGR))
        u, v, _ = gs_utils.project(pos, w2c, K)
        rgb = sample_bilinear(filled, u, v)
        for i in range(3):
            new[f"f_dc_{i}"] = (rgb[:, i] - 0.5) / C0

    new["x"], new["y"], new["z"] = pos[:, 0], pos[:, 1], pos[:, 2]
    new["opacity"] = np.log(args.opacity / (1.0 - args.opacity))
    s_in, s_n = np.log(args.overlap * spacing), np.log(args.thin * spacing)
    new["scale_0"], new["scale_1"], new["scale_2"] = s_in, s_in, s_n
    q = rot_to_quat(np.stack([a, b, n], 1))
    for i in range(4):
        new[f"rot_{i}"] = q[i]
    print(f"added {len(new)} gaussians, spacing {spacing:.5f}, appearance {args.appearance}")

    final = np.concatenate([kept, new])
    dst = Path(args.model_out)
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("cfg_args", "cameras.json"):
        shutil.copy(src / name, dst / name)
    write_ply(final, dst / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply")

    fxyz = np.stack([final["x"], final["y"], final["z"]], 1).astype(np.float64)
    fhp, frr = radial(fxyz, foot, n)
    fslab = (frr < 1.5 * half) & (fhp > -args.slab_below * r_foot) & (fhp < args.slab_above * r_foot)
    after = render_vertices(final[fslab], cam, out / "topdown_subset.ply")
    cv2.imwrite(str(out / "topdown_after.png"), cv2.cvtColor(after, cv2.COLOR_RGB2BGR))
    (out / "topdown_subset.ply").unlink(missing_ok=True)

    with open(out / "fill.json", "w") as f:
        json.dump({"cleared": int(clear.sum()), "added": int(len(new)), "r_fill": r_fill,
                   "height_offset": h_off, "spacing": spacing, "half_size": half,
                   "params": vars(args)}, f, indent=2)
    print(f"filled model written to {dst}")


if __name__ == "__main__":
    main()