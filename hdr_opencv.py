"""OpenCV/numpy implementation of the HDR process in HDRAlgDemo.exe.

The demo (built on Halcon) computes, for process type "hdr":
  1. median (circle, r=1) denoise
  2. normalize to [0, 65535] using (percent-cut max - full min) (uint2 input)
  3. log-domain processing:
       L = ln(I); N = L / max(L)
       base = guided_filter(N, N, window=border, eps=0.001) * max(L)
       detail = median_sep3x3(L - base)
       k = ln(intensity) / (max(base) - min(base))
       out = exp(k*(base - max(base)) + detail*detail_param)
  4. normalize result to [0, 1], scale by 65535 -> uint2
  5. GaussFilter(sigma) + GaussFilter(11) * (gauss//11)
All operators were reverse-engineered from the demo binary and verified
against halconcpp.dll directly (see research/).
"""

import numpy as np
import cv2

# ---------------------------------------------------------------- kernels ---
# Exact Halcon gauss_filter kernels (size x size), extracted from halconcpp.dll
# by filtering a real impulse image (see research/probe_ops.py). Normalized.
_GAUSS_KERNELS = {
    3: np.array([
        [0.040936775505542755, 0.1204548329114914, 0.040936775505542755],
        [0.1204548329114914, 0.3544335663318634, 0.1204548329114914],
        [0.040936775505542755, 0.1204548329114914, 0.040936775505542755],
    ], dtype=np.float32),
    5: np.array([
        [0.006634972058236599, 0.019506007432937622, 0.02917337603867054, 0.019506007432937622, 0.006634972058236599],
        [0.019506007432937622, 0.05734528973698616, 0.08576617389917374, 0.05734528973698616, 0.019506007432937622],
        [0.02917337603867054, 0.08576617389917374, 0.12827272713184357, 0.08576617389917374, 0.02917337603867054],
        [0.019506007432937622, 0.05734528973698616, 0.08576617389917374, 0.05734528973698616, 0.019506007432937622],
        [0.006634972058236599, 0.019506007432937622, 0.02917337603867054, 0.019506007432937622, 0.006634972058236599],
    ], dtype=np.float32),
    7: np.array([
        [0.0028497197199612856, 0.006043135654181242, 0.011045951396226883, 0.01350515428930521, 0.011045951396226883, 0.006043135654181242, 0.0028497197199612856],
        [0.006043135654181242, 0.012815115042030811, 0.023424120619893074, 0.02863912284374237, 0.023424120619893074, 0.012815115042030811, 0.006043135654181242],
        [0.011045951396226883, 0.023424120619893074, 0.04281580075621605, 0.052348047494888306, 0.04281580075621605, 0.023424120619893074, 0.011045951396226883],
        [0.013505153357982635, 0.02863912284374237, 0.052348047494888306, 0.06400249153375626, 0.052348047494888306, 0.02863912284374237, 0.013505153357982635],
        [0.011045951396226883, 0.023424120619893074, 0.04281580075621605, 0.052348047494888306, 0.04281580075621605, 0.023424120619893074, 0.011045951396226883],
        [0.006043135654181242, 0.012815115042030811, 0.023424120619893074, 0.02863912284374237, 0.023424120619893074, 0.012815115042030811, 0.006043135654181242],
        [0.0028497197199612856, 0.006043135654181242, 0.011045951396226883, 0.01350515428930521, 0.011045951396226883, 0.006043135654181242, 0.0028497197199612856],
    ], dtype=np.float32),
    9: np.array([
        [0.0017605334287509322, 0.002791805425658822, 0.005074052140116692, 0.007261468097567558, 0.008182993158698082, 0.007261468097567558, 0.005074052140116692, 0.002791805425658822, 0.0017605334287509322],
        [0.002791805425658822, 0.0044271680526435375, 0.008046291768550873, 0.011515036225318909, 0.012976366095244884, 0.011515036225318909, 0.008046291768550873, 0.0044271680526435375, 0.002791805425658822],
        [0.005074052140116692, 0.008046291768550873, 0.014623980037868023, 0.020928354933857918, 0.023584291338920593, 0.020928354933857918, 0.014623980037868023, 0.008046291768550873, 0.005074052140116692],
        [0.007261467631906271, 0.011515036225318909, 0.02092835307121277, 0.02995053306221962, 0.03375144302845001, 0.02995053306221962, 0.02092835307121277, 0.011515036225318909, 0.007261467631906271],
        [0.008182992227375507, 0.01297636516392231, 0.023584289476275444, 0.033751439303159714, 0.03803470730781555, 0.033751439303159714, 0.023584289476275444, 0.01297636516392231, 0.008182992227375507],
        [0.007261467631906271, 0.011515036225318909, 0.02092835307121277, 0.02995053306221962, 0.03375144302845001, 0.02995053306221962, 0.02092835307121277, 0.011515036225318909, 0.007261467631906271],
        [0.005074052140116692, 0.008046291768550873, 0.014623980037868023, 0.020928354933857918, 0.023584291338920593, 0.020928354933857918, 0.014623980037868023, 0.008046291768550873, 0.005074052140116692],
        [0.002791805425658822, 0.0044271680526435375, 0.008046291768550873, 0.011515036225318909, 0.012976366095244884, 0.011515036225318909, 0.008046291768550873, 0.0044271680526435375, 0.002791805425658822],
        [0.0017605334287509322, 0.002791805425658822, 0.005074052140116692, 0.007261468097567558, 0.008182993158698082, 0.007261468097567558, 0.005074052140116692, 0.002791805425658822, 0.0017605334287509322],
    ], dtype=np.float32),
    11: np.array([
        [0.0012909878278151155, 0.0016106247203424573, 0.0027989211957901716, 0.004153468180447817, 0.005263330414891243, 0.005695653147995472, 0.005263330414891243, 0.004153468180447817, 0.0027989211957901716, 0.0016106247203424573, 0.0012909878278151155],
        [0.0016106247203424573, 0.0020094006322324276, 0.0034919087775051594, 0.0051818289794027805, 0.006566483527421951, 0.0071058450266718864, 0.006566483527421951, 0.0051818289794027805, 0.0034919087775051594, 0.0020094006322324276, 0.0016106247203424573],
        [0.0027989211957901716, 0.0034919087775051594, 0.006068191025406122, 0.009004911407828331, 0.011411145329475403, 0.012348439544439316, 0.011411145329475403, 0.009004911407828331, 0.006068191025406122, 0.0034919087775051594, 0.0027989211957901716],
        [0.004153468180447817, 0.005181829445064068, 0.009004911407828331, 0.013362865895032883, 0.016933605074882507, 0.018324507400393486, 0.016933605074882507, 0.013362865895032883, 0.009004911407828331, 0.005181829445064068, 0.004153468180447817],
        [0.00526333088055253, 0.006566483993083239, 0.011411145329475403, 0.016933605074882507, 0.021458491683006287, 0.023221060633659363, 0.021458491683006287, 0.016933605074882507, 0.011411145329475403, 0.006566483993083239, 0.00526333088055253],
        [0.005695653147995472, 0.0071058450266718864, 0.012348439544439316, 0.018324505537748337, 0.023221060633659363, 0.025128405541181564, 0.023221060633659363, 0.018324505537748337, 0.012348439544439316, 0.0071058450266718864, 0.005695653147995472],
        [0.00526333088055253, 0.006566483993083239, 0.011411145329475403, 0.016933605074882507, 0.021458491683006287, 0.023221060633659363, 0.021458491683006287, 0.016933605074882507, 0.011411145329475403, 0.006566483993083239, 0.00526333088055253],
        [0.004153468180447817, 0.005181829445064068, 0.009004911407828331, 0.013362865895032883, 0.016933605074882507, 0.018324507400393486, 0.016933605074882507, 0.013362865895032883, 0.009004911407828331, 0.005181829445064068, 0.004153468180447817],
        [0.0027989211957901716, 0.0034919087775051594, 0.006068191025406122, 0.009004911407828331, 0.011411145329475403, 0.012348439544439316, 0.011411145329475403, 0.009004911407828331, 0.006068191025406122, 0.0034919087775051594, 0.0027989211957901716],
        [0.0016106247203424573, 0.0020094006322324276, 0.0034919087775051594, 0.0051818289794027805, 0.006566483527421951, 0.0071058450266718864, 0.006566483527421951, 0.0051818289794027805, 0.0034919087775051594, 0.0020094006322324276, 0.0016106247203424573],
        [0.0012909878278151155, 0.0016106247203424573, 0.0027989211957901716, 0.004153468180447817, 0.005263330414891243, 0.005695653147995472, 0.005263330414891243, 0.004153468180447817, 0.0027989211957901716, 0.0016106247203424573, 0.0012909878278151155],
    ], dtype=np.float32),
}


def _gauss_kernel(size):
    """Return the exact Halcon gauss_filter kernel for odd size (3..11)."""
    return _GAUSS_KERNELS[size]


# ------------------------------------------------------------------- utils ---
def _reflect101(img):
    """Mirror-pad by 2 pixels on all sides (Halcon 'mirrored' margin)."""
    return cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_REFLECT_101)


def median_cross(img):
    """Halcon median_image(Image, 'circle', 1, 'mirrored'):
    median of center + 4-neighborhood (plus/cross mask), mirrored border."""
    img = np.asarray(img)
    p = _reflect101(img)
    h, w = img.shape
    vals = np.empty((5, h, w), dtype=np.float32)
    vals[0] = p[2:2 + h, 2:2 + w]
    vals[1] = p[1:1 + h, 2:2 + w]
    vals[2] = p[3:3 + h, 2:2 + w]
    vals[3] = p[2:2 + h, 1:1 + w]
    vals[4] = p[2:2 + h, 3:3 + w]
    out = np.median(vals, axis=0)
    # 5 个 16 位整数的中值在 float32 中精确，直接转回 uint16
    return out.astype(np.uint16)


def median_sep3(img):
    """Halcon median_separate(img, 3, 3, 'mirrored'): vertical median then
    horizontal median with mirrored border."""
    h, w = img.shape
    # vertical median (columns, 3-tap)
    p = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REFLECT_101)
    v = np.median(np.stack([p[0:-2, 1:-1], p[1:-1, 1:-1], p[2:, 1:-1]], axis=0), axis=0)
    vp = cv2.copyMakeBorder(v, 1, 1, 1, 1, cv2.BORDER_REFLECT_101)
    o = np.median(np.stack([vp[1:-1, 0:-2], vp[1:-1, 1:-1], vp[1:-1, 2:]], axis=0), axis=0)
    return o


def halcon_mean(img, size):
    """Halcon mean_image with mask size: even sizes are rounded up to the
    next odd number; mirrored border; normalized box mean."""
    if size % 2 == 0:
        size += 1
    if size == 1:
        return img.astype(np.float32)
    return cv2.boxFilter(img.astype(np.float32), -1, (size, size),
                         normalize=True, borderType=cv2.BORDER_REFLECT_101)


def halcon_gauss(img, size):
    """Halcon gauss_filter with the exact kernel and mirrored border."""
    k = _gauss_kernel(size)
    return cv2.filter2D(img.astype(np.float32), -1, k,
                        borderType=cv2.BORDER_REFLECT_101)


def halcon_scale(img, mult, add, out_dtype):
    """Halcon scale_image: computes float, stores into out_dtype with
    truncation toward zero and clipping."""
    r = img.astype(np.float64) * mult + add
    if np.issubdtype(out_dtype, np.integer):
        r = np.trunc(r)
        info = np.iinfo(out_dtype)
        r = np.clip(r, info.min, info.max)
    return r.astype(out_dtype)


def guided_filter(a, b, window, eps):
    """Halcon 0xcb20 guided filter:
       ratio = (mean(B*A) - mean(A)*mean(B)) / (mean(A*A) - mean(A)^2 + eps)
       out   = mean(ratio) * A + mean(mean(B) - ratio * mean(A))
    with box means of size window (even -> rounded up)."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mean_a = halcon_mean(a, window)
    mean_b = halcon_mean(b, window)
    aa = a * a
    ba = b * a
    mean_aa = halcon_mean(aa, window)
    mean_ba = halcon_mean(ba, window)
    var_a = mean_aa - mean_a * mean_a
    cov = mean_ba - mean_a * mean_b
    ratio = cov / (var_a + eps)
    mean_ratio = halcon_mean(ratio, window)
    diff = mean_b - ratio * mean_a
    mean_diff = halcon_mean(diff, window)
    return mean_ratio * a + mean_diff


def _log_image(img):
    """Halcon log_image base 'e': log(0) is mapped to 0."""
    img = np.asarray(img, dtype=np.float32)
    out = np.zeros_like(img)
    mask = img > 0
    out[mask] = np.log(img[mask])
    return out


def _min_max_gray(img, percent):
    """Halcon min_max_gray percent semantics: cut `percent`% of the darkest and
    brightest pixels. Min = lower quantile at percent/100, Max = higher
    quantile at 1-percent/100 (matches numpy 'lower'/'higher' methods)."""
    if percent <= 0:
        return float(np.min(img)), float(np.max(img))
    n = img.size
    s = np.sort(img.ravel())
    lo = int(np.floor((percent / 100.0) * (n - 1)))
    hi = int(np.ceil((1.0 - percent / 100.0) * (n - 1)))
    return float(s[lo]), float(s[hi])


def _exp_image(img):
    """Halcon exp_image base 'e'."""
    return np.exp(img.astype(np.float32)).astype(np.float32)


def _a850(img, eps, in_type):
    """0xa850: normalize using range (percent-cut max - full min).
    The demo computes Max1 = min_max_gray(percent=eps*100).Max and
    Min2 = min_max_gray(percent=0).Min, then uses Max1 - Min2 as the range.
    The result can therefore exceed maxval for the brightest pixels (which the
    demo leaves to the uint2 conversion to clip)."""
    img64 = img.astype(np.float64)
    mn_full, _ = _min_max_gray(img64, 0)
    _, mx_pct = _min_max_gray(img64, eps * 100.0)
    span = mx_pct - mn_full
    if span <= 0:
        return np.zeros_like(img64)
    f = 1.0 / span
    if in_type == 'uint2':
        mult = f * 65535.0
    elif in_type == 'byte':
        mult = f * 255.0
    else:
        mult = f
    out = (img64 - mn_full) * mult
    if in_type in ('uint2', 'byte'):
        # Halcon scale_image on integer images rounds (nearest, half up)
        out = np.floor(out + 0.5)
        out = np.clip(out, 0, 65535 if in_type == 'uint2' else 255)
        return out.astype(np.uint16 if in_type == 'uint2' else np.uint8)
    return out.astype(np.float32)


def _d170(img, intensity, detail_param, border, guided_eps=0.001, median_fn=median_sep3):
    """0xd170: log-domain HDR core."""
    real = img.astype(np.float32)
    log_img = _log_image(real)
    mx = log_img.max()
    mn = log_img.min()
    rng = mx - mn
    if rng < 1e-5:
        return real
    norm = log_img * (1.0 / mx)
    guided = guided_filter(norm, norm, border, guided_eps)
    base = guided * mx
    detail = median_fn(log_img - base)
    base_mx = base.max()
    base_mn = base.min()
    k = float(np.log(intensity)) / (base_mx - base_mn) if base_mx > base_mn else 0.0
    out = base * k + detail * detail_param - base_mx * k
    return _exp_image(out)


def hdr_process(src, intensity=20, detail=3, border=2, gauss=20,
                guided_eps=0.001, norm_eps1=0.01, norm_eps2=0.02):
    """Full 'hdr' process type, matching HDRAlgDemo.exe defaults.
    src: uint16 (H, W) image. Returns uint16 (H, W)."""
    if src.dtype != np.uint16:
        raise ValueError('hdr_process expects a uint16 image')
    img = median_cross(src)
    n1 = _a850(img, norm_eps1, 'uint2').astype(np.float32)
    h = _d170(n1, intensity, detail, border, guided_eps=guided_eps)
    n2 = _a850(h, norm_eps2, 'real')
    out = halcon_scale(n2, 65535.0, 0.0, np.uint16)
    sigma = gauss % 11
    if sigma % 2 == 0:
        sigma += 1
    if sigma < 3:
        sigma = 3
    out = halcon_gauss(out, sigma)
    iters = gauss // 11
    for _ in range(iters):
        out = halcon_gauss(out, 11)
    # Halcon gauss_filter on a uint2 image returns uint2 (rounded)
    return np.clip(np.floor(np.asarray(out, dtype=np.float64) + 0.5),
                   0, 65535).astype(np.uint16)


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description='OpenCV implementation of HDRAlgDemo "hdr" process type')
    ap.add_argument('input', help='input image path (uint16 TIFF recommended)')
    ap.add_argument('-o', '--output', default='hdr_out.tiff',
                    help='output image path (default: hdr_out.tiff)')
    ap.add_argument('--intensity', type=int, default=20,
                    help='intensity parameter, default 20 (demo range 2..300)')
    ap.add_argument('--detail', type=int, default=3,
                    help='detail parameter, default 3 (demo range 1..30)')
    ap.add_argument('--border', type=int, default=2,
                    help='border/guided-filter window, default 2 (demo range 1..200)')
    ap.add_argument('--gauss', type=int, default=20,
                    help='gauss parameter, default 20 (demo range 1..30)')
    args = ap.parse_args()

    src = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if src is None:
        raise SystemExit(f'cannot read image: {args.input}')
    print(f'source: {src.shape} {src.dtype}')
    res = hdr_process(src, intensity=args.intensity, detail=args.detail,
                      border=args.border, gauss=args.gauss)
    cv2.imwrite(args.output, res)
    print(f'saved: {args.output}  min={res.min()} max={res.max()} mean={res.mean():.1f}')


if __name__ == '__main__':
    main()
