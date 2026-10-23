"""Vector PSF simulator for V4-PSF, 3-lenslet MLA, and DH-PSF.

Adapted from the Codex local optical reference
(E:\\文章3内容\\code\\v4_decode\\optical_reference.py), which ports
FLFM_calcPSF_RW.m / FLFM_ulensTransmittance.m / prop2Sensor.m.

The scalar reference was validated against the MATLAB native-depth stacks
(201/201 slices; max relative L2 0.8%, max centroid shift 0.005 px).
This version adds:
  - objective-specific configurations (20x/40x/60x),
  - Richards-Wolf vector components A/B/C,
  - V4, three-hexagon MLA, and the Wang six-zone double-helix phase mask,
  - phase scaling to the objective-specific pupil beam radius.

Outputs normalized PSF stacks; CRLB is computed separately.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy import fft
from matplotlib.path import Path as MplPath

N_SOURCE = 8191
LU0 = 200.0  # um
WAVELENGTH = 0.55  # um
PIXEL_PITCH = 4.5  # um
F1 = 200000.0  # um
F2 = 200000.0  # um
FM = 200000.0  # um
LENS_PITCH = 50000.0  # um
DH_FFT_SIZE = 1024  # DH-only physical-pupil FFT; increase to 2048 for convergence testing


@dataclass(frozen=True)
class Config:
    name: str
    M: float
    NA: float
    immersion_n: float
    zmin: float
    zmax: float
    dz: float = 0.5
    prism_alpha_deg: float = 5.0

    @property
    def fobj(self) -> float:
        return F1 / self.M

    @property
    def pupil_radius(self) -> float:
        return self.fobj * self.NA

    @property
    def sensor_size(self) -> int:
        pupilsize = self.NA / self.M * 2.0 * F2 / PIXEL_PITCH
        return int(math.ceil(pupilsize + 500.0))

    @property
    def sensor_res(self) -> float:
        nspacing = LENS_PITCH / PIXEL_PITCH
        # MATLAB rounds the lenslet grid to an even integer; the sensor grid
        # itself is sampled at the camera pitch, which is the relevant scale.
        return PIXEL_PITCH

    @property
    def depths(self) -> np.ndarray:
        n = int(round((self.zmax - self.zmin) / self.dz)) + 1
        return np.linspace(self.zmin, self.zmax, n)

    @property
    def yspace_end(self) -> float:
        return self.sensor_res * math.floor(self.sensor_size / 2.0)

    @property
    def beam_radius_sensor(self) -> float:
        # After the FFT and the fixed crop/resize, the pupil radius maps to
        # the sensor grid as r_pupil / (Mrelay * LU1 / 2) * (N_sensor / 2).
        # Do not use yspace_end here: yspace_end is a sensor-plane coordinate,
        # not the BFP coordinate of the crop edge.
        lu1 = WAVELENGTH * self.fobj / (LU0 / N_SOURCE)
        return self.pupil_radius / ((F2 / F1) * lu1 / 2.0) * (self.sensor_size / 2.0)


CONFIGS = {
    "20x": Config("20x_0.45NA_air", 20.0, 0.45, 1.0, -10.0, 10.0),
    "40x": Config("40x_0.60NA_air", 40.0, 0.60, 1.0, -10.0, 10.0),
    "60x": Config("60x_1.0NA_water", 60.0, 1.0, 1.333, -5.0, 5.0),
}


def cubic_kernel(x):
    ax = np.abs(x)
    return np.where(ax <= 1, 1.5 * ax**3 - 2.5 * ax**2 + 1,
                    np.where(ax < 2, -0.5 * ax**3 + 2.5 * ax**2 - 4 * ax + 2, 0.0))


def resize_cubic_up(array, shape):
    """MATLAB-style half-pixel Keys cubic (a=-0.5), symmetric edges."""
    result = array
    for axis, target in enumerate(shape):
        original = result.shape[axis]
        if target < original:
            raise ValueError("This reference only implements bicubic upsampling.")
        positions = (np.arange(target) + 0.5) * original / target - 0.5
        base = np.floor(positions).astype(int)
        indices = base[:, None] + np.arange(-1, 3)[None]
        weights = cubic_kernel(positions[:, None] - indices)
        reflected = indices % (2 * original)
        reflected = np.where(reflected < original, reflected, 2 * original - 1 - reflected)
        output_shape = list(result.shape)
        output_shape[axis] = target
        output = np.zeros(output_shape, dtype=result.dtype)
        for k in range(4):
            wshape = [1] * result.ndim
            wshape[axis] = target
            output += np.take(result, reflected[:, k], axis=axis) * weights[:, k].reshape(wshape)
        result = output
    return result


def _bfp_u1(z_um: float, cfg: Config, workers: int = 4):
    """Return the complex BFP field U1 plus the cropped BFP coordinates."""
    n = N_SOURCE
    mid = (n + 1) // 2
    z = float(z_um) if z_um else 1e-8
    x = np.linspace(-LU0 / 2.0, LU0 / 2.0, n)
    k = 2.0 * np.pi / WAVELENGTH * cfg.immersion_n

    field = np.empty((n, n), dtype=np.complex128)
    for begin in range(0, n, 64):
        y = x[begin:begin + 64, None]
        r = np.sqrt(x[None, :]**2 + y * y + z * z)
        if z > 0:
            r = -r
        field[begin:begin + len(y)] = np.exp(1j * k * r) / r

    shifted = np.fft.ifftshift(field)
    del field
    transformed = fft.fft2(shifted, overwrite_x=True, workers=workers)
    del shifted
    dx1 = LU0 / n
    u1 = transformed
    del transformed

    lu1 = WAVELENGTH * cfg.fobj / dx1
    mrelay = F2 / F1
    cut = int(np.round(cfg.yspace_end / (mrelay * lu1 / 2.0) * (n + 1) / 2.0))
    # The FFT output has zero frequency at index 0; the pupil crop must wrap
    # around index 0, exactly as in optical_reference.bfp_field.
    indices = np.arange(-cut, cut + 1) % n
    u1_crop = u1[np.ix_(indices, indices)].copy()
    del u1

    xy = np.arange(-cut, cut + 1) * lu1 / (n - 1)
    x_bfp, y_bfp = np.meshgrid(xy, xy, indexing="ij")
    return u1_crop, x_bfp, y_bfp, cut


def bfp_components(z_um: float, cfg: Config, workers: int = 4):
    """Return resized vector BFP components A/B/C and the sensor-grid beam radius."""
    u1, x_bfp, y_bfp, _ = _bfp_u1(z_um, cfg, workers=workers)
    comps = _vector_components(u1, x_bfp, y_bfp, cfg)
    # V4 uses the validated post-resize phase-mask convention.
    out = [resize_cubic_up(c, (cfg.sensor_size, cfg.sensor_size)) for c in comps]
    return out, cfg.beam_radius_sensor


def _vector_components(u1, x_bfp, y_bfp, cfg: Config):
    r_bfp = np.hypot(x_bfp, y_bfp)
    pupil = r_bfp < cfg.pupil_radius
    sin_max = min(cfg.NA / cfg.immersion_n, 1.0 - 1e-12)
    rho = r_bfp / cfg.pupil_radius
    sin_theta = np.minimum(rho * sin_max, 1.0 - 1e-12)
    cos_theta = np.sqrt(1.0 - sin_theta**2)
    phi = np.arctan2(y_bfp, x_bfp)
    cp, sp = np.cos(phi), np.sin(phi)
    apo = np.sqrt(cos_theta)

    # Richards-Wolf / Novotny & Hecht Eq. 3.66, unpolarised emission.
    f_a = apo * (cp**2 * cos_theta + sp**2)
    f_b = apo * (cos_theta - 1.0) * sp * cp
    f_c = apo * (sp**2 * cos_theta + cp**2)
    return [u1 * pupil * f for f in (f_a, f_b, f_c)]


def bfp_components_raw(z_um: float, cfg: Config, workers: int = 4):
    """Vector components on the cropped BFP grid, before sensor resize."""
    u1, x_bfp, y_bfp, _ = _bfp_u1(z_um, cfg, workers=workers)
    comps = _vector_components(u1, x_bfp, y_bfp, cfg)
    return comps, x_bfp, y_bfp, cfg.pupil_radius


def _hexagon_vertices(center, circumradius, toward_origin=True):
    angle = math.atan2(center[1], center[0])
    if toward_origin:
        angle += math.pi
    a = np.arange(6) * (math.pi / 3.0) + angle
    return np.column_stack([center[0] + circumradius * np.cos(a),
                            center[1] + circumradius * np.sin(a)])


def phase_mask(kind: str, cfg: Config, shape, r_beam: float):
    """Build the pupil/relay phase mask on the sensor-grid coordinates."""
    n = shape[0]
    space = cfg.sensor_res * np.linspace(-math.floor(n / 2), math.floor(n / 2), n)
    X, Y = np.meshgrid(space, space, indexing="ij")
    # The prism and relay lens are in air after the objective; MATLAB uses
    # Camera.k = 2*pi*Camera.n/lambda with Config.n = 1.0 (not immersion_n).
    k = 2.0 * np.pi / WAVELENGTH

    if kind == "v4":
        coeff = k * (1.516 - 1.506) * math.tan(math.radians(cfg.prism_alpha_deg))
        prism = np.zeros_like(X)
        tl = (Y < 0) & (X < 0)
        tr = (Y < 0) & (X > 0)
        br = (Y > 0) & (X > 0)
        bl = (Y > 0) & (X < 0)
        prism[tl] = -X[tl]
        prism[tr] = -Y[tr]
        prism[br] = X[br]
        prism[bl] = Y[bl]
        lens_space = cfg.sensor_res * np.arange(-n // 2, n // 2)
        lens = -(k / (2.0 * FM)) * (lens_space[:, None]**2 + lens_space[None, :]**2)
        return np.exp(1j * (coeff * prism + lens))

    if kind == "mla":
        # Three pointy-top hexagons, union inscribed in the pupil beam.
        # Centers at 120 deg; each hexagon has a vertex toward the origin.
        a = r_beam / 2.0
        centers = [(a * math.cos(t), a * math.sin(t))
                   for t in (math.pi / 2, math.pi / 2 + 2 * math.pi / 3,
                             math.pi / 2 + 4 * math.pi / 3)]
        points = np.column_stack([X.ravel(), Y.ravel()])
        phase = np.zeros_like(X, dtype=np.complex128)
        union = np.zeros(X.shape, dtype=bool)
        for cx, cy in centers:
            verts = _hexagon_vertices((cx, cy), a, toward_origin=True)
            mask = MplPath(verts).contains_points(points).reshape(X.shape)
            if not mask.any():
                continue
            union |= mask
            lens = np.exp(-1j * k / (2.0 * FM) * ((X - cx)**2 + (Y - cy)**2))
            phase[mask] = lens[mask]
        phase[~union] = 0.0
        return phase

    if kind == "dh":
        # N=3 (+1) vortices along x; standard simplified multi-vortex DH mask.
        d = 0.5 * r_beam
        zc = X + 1j * Y
        # Product of (z-z_j) gives a phase with +1 charge at each vortex.
        phase = np.angle((zc + d) * zc * (zc - d))
        lens = -(k / (2.0 * FM)) * (X**2 + Y**2)
        return np.exp(1j * (phase + lens))

    raise ValueError(f"Unknown PSF kind: {kind}")


def propagate(field, cfg: Config):
    """Angular-spectrum propagation from the pupil/relay plane to the sensor."""
    n = field.shape[0]
    k = 2.0 * np.pi / WAVELENGTH
    du = 1.0 / (n * cfg.sensor_res)
    indices = np.concatenate([np.arange((n + 1) // 2), np.arange(-(n // 2), 0)])
    u = indices.astype(np.float64) * du
    q = 1.0 - WAVELENGTH**2 * (u[:, None]**2 + u[None, :]**2)
    q = np.maximum(q, 0.0)
    h = np.exp(1j * np.sqrt(q) * FM * k)
    return fft.ifft2(fft.fft2(field) * h)


def bfp_aperture(kind: str, cfg: Config, X, Y, r_beam: float):
    """Binary aperture in the BFP plane (block light outside the MLA union)."""
    if kind == "mla":
        a = r_beam / 2.0
        centers = [(a * math.cos(t), a * math.sin(t))
                   for t in (math.pi / 2, math.pi / 2 + 2 * math.pi / 3,
                             math.pi / 2 + 4 * math.pi / 3)]
        points = np.column_stack([X.ravel(), Y.ravel()])
        union = np.zeros(X.shape, dtype=bool)
        for cx, cy in centers:
            verts = _hexagon_vertices((cx, cy), a, toward_origin=True)
            union |= MplPath(verts).contains_points(points).reshape(X.shape)
        return union
    if kind == "dh":
        return np.hypot(X, Y) <= r_beam
    raise ValueError(f"bfp_aperture does not handle kind={kind}")


def phase_mask_bfp(kind: str, cfg: Config, X, Y, r_beam: float):
    """Phase mask on the cropped BFP physical grid (µm)."""
    k = 2.0 * np.pi / WAVELENGTH

    if kind == "mla":
        # Three pointy-top hexagons, union inscribed in the pupil beam.
        a = r_beam / 2.0
        centers = [(a * math.cos(t), a * math.sin(t))
                   for t in (math.pi / 2, math.pi / 2 + 2 * math.pi / 3,
                             math.pi / 2 + 4 * math.pi / 3)]
        points = np.column_stack([X.ravel(), Y.ravel()])
        phase = np.zeros_like(X, dtype=np.complex128)
        union = np.zeros(X.shape, dtype=bool)
        for cx, cy in centers:
            verts = _hexagon_vertices((cx, cy), a, toward_origin=True)
            mask = MplPath(verts).contains_points(points).reshape(X.shape)
            if not mask.any():
                continue
            union |= mask
            lens = np.exp(-1j * k / (2.0 * FM) * ((X - cx)**2 + (Y - cy)**2))
            phase[mask] = lens[mask]
        phase[~union] = 0.0
        return phase

    if kind == "dh":
        # Spiral-phase-based DH-PSF: Fresnel-zone sampled spiral phase.
        # Wang et al., J. Biomed. Opt. 24, 031005 (2019), Eq. (1):
        # P_N(rho,theta) = (2n-1) theta
        # for sqrt((n-1)/N) < rho <= sqrt(n/N).
        # N = 6 Fresnel zones as in that paper.
        N_zones = 6
        rho = np.hypot(X, Y) / r_beam
        theta = np.arctan2(Y, X)
        phase = np.zeros_like(X, dtype=np.float64)
        for n in range(1, N_zones + 1):
            zone = ((rho > np.sqrt((n - 1) / N_zones))
                    & (rho <= np.sqrt(n / N_zones)))
            phase[zone] = (2 * n - 1) * theta[zone]
        phase[rho > 1.0] = 0.0
        lens = -(k / (2.0 * FM)) * (X**2 + Y**2)
        return np.exp(1j * (phase + lens))

    raise ValueError(f"phase_mask_bfp does not handle kind={kind}")



def dh_psf_ideal_pupil(z_um: float, cfg: Config, grid_size: int = DH_FFT_SIZE):
    """Wang six-equal-area-zone phase mask with physical objective defocus.

    This is an ideal aplanatic-pupil baseline, NOT a claimed pixel-exact
    reproduction of Wang's Eq. (3) nor a ray trace of the relay optics.

    The Fourier grid is scaled so one FFT sample represents the actual
    object-plane length PIXEL_PITCH / M, i.e. a camera pixel at M magnification.
    The zone phase is applied directly, without cubic interpolation of its
    discontinuities.  The same A/B/C polarization-average assumption as
    _vector_components() is used.
    """
    size = int(grid_size)
    if size < 512 or size % 2 != 0:
        raise ValueError("DH_FFT_SIZE must be even and >=512")
    object_pixel_um = PIXEL_PITCH / cfg.M
    # Spatial frequency at the pupil rim is NA / wavelength (cycles/um).
    # Frequency grid increment for image sampling delta_x is 1 / (N delta_x).
    r_pupil_px = size * object_pixel_um * cfg.NA / WAVELENGTH
    axis = np.arange(size, dtype=np.float64) - size // 2
    X, Y = np.meshgrid(axis, axis, indexing="xy")
    rho = np.hypot(X, Y) / r_pupil_px
    inside = rho <= 1.0
    az = np.arctan2(Y, X)
    n_zones = 6
    # ceil(N * rho**2) implements the published sqrt radial boundaries.
    zone_number = np.clip(np.ceil(n_zones * rho * rho), 1, n_zones)
    mask = np.exp(1j * (2.0 * zone_number - 1.0) * az)

    s = cfg.NA / cfg.immersion_n
    if s > 1.0:
        raise ValueError("NA cannot exceed immersion refractive index")
    cos_alpha = np.sqrt(np.clip(1.0 - (s * rho)**2, 0.0, 1.0))
    # k0*n*z*(cos(alpha)-1), where removing piston has no effect on intensity.
    k0 = 2.0 * np.pi / WAVELENGTH
    defocus = np.exp(1j * k0 * cfg.immersion_n * float(z_um) * (cos_alpha - 1.0))
    cp, sp = np.cos(az), np.sin(az)
    apo = np.sqrt(cos_alpha)
    base = (apo * inside) * mask * defocus
    comp_a = base * (cp**2 * cos_alpha + sp**2)
    comp_b = base * ((cos_alpha - 1.0) * sp * cp)
    comp_c = base * (sp**2 * cos_alpha + cp**2)

    def to_image(component):
        return fft.fftshift(fft.fft2(fft.ifftshift(component), workers=1))

    fa = to_image(comp_a)
    fb = to_image(comp_b)
    fc = to_image(comp_c)
    intensity = np.abs(fa)**2 + 2.0 * np.abs(fb)**2 + np.abs(fc)**2
    intensity = np.maximum(intensity, 0.0)
    total = intensity.sum()
    if total > 0.0:
        intensity /= total
    return intensity.astype(np.float32), r_pupil_px


def psf_at_z(z_um: float, cfg: Config, kind: str, workers: int = 4,
             bfp_oversample: int = 1):
    if kind == "v4":
        comps, r_beam = bfp_components(z_um, cfg, workers=workers)
        phase = phase_mask(kind, cfg, comps[0].shape, r_beam)
        fields = [propagate(comp * phase, cfg) for comp in comps]
    else:
        comps, x_bfp, y_bfp, r_beam = bfp_components_raw(z_um, cfg, workers=workers)
        # The 20x raw BFP grid is too coarse for the lenslet quadratic phase:
        # at the pupil edge the phase step per sample can exceed pi, which
        # aliases the lens and produces an apparent +5 um focus offset.
        # Interpolating the BFP field before applying the phase mask removes
        # that sampling artefact while keeping the physical BFP coordinates.
        if bfp_oversample > 1:
            n = x_bfp.shape[0]
            comps = [resize_cubic_up(c, (n * bfp_oversample, n * bfp_oversample))
                     for c in comps]
            xx = np.linspace(float(x_bfp.min()), float(x_bfp.max()),
                             n * bfp_oversample)
            X, Y = np.meshgrid(xx, xx, indexing="ij")
        else:
            X, Y = x_bfp, y_bfp
        phase = phase_mask_bfp(kind, cfg, X, Y, r_beam)
        # Block interpolated phase-mask edge leakage: after the cubic resize,
        # multiply by the same binary BFP aperture on the target grid.
        target_x = np.linspace(float(x_bfp.min()), float(x_bfp.max()), cfg.sensor_size)
        Xa, Ya = np.meshgrid(target_x, target_x, indexing="ij")
        aperture = bfp_aperture(kind, cfg, Xa, Ya, r_beam)
        fields = []
        for comp in comps:
            resized = resize_cubic_up(comp * phase, (cfg.sensor_size, cfg.sensor_size))
            fields.append(propagate(resized * aperture, cfg))
    image = np.abs(fields[0])**2 + 2.0 * np.abs(fields[1])**2 + np.abs(fields[2])**2
    image = np.maximum(image, 0.0)
    total = image.sum()
    if total > 0:
        image /= total
    return image.astype(np.float32), r_beam


def center_crop(image: np.ndarray, size: int, offset: int = 5):
    n = image.shape[0]
    if size >= n:
        return image
    # The MATLAB reference uses a fixed 329-px window at [757:1086] for the
    # 1834-px sensor, i.e. a +5 px offset from the geometric center. Keep the
    # same convention so the Python stack lines up with the native stacks.
    start = (n - size) // 2 + offset
    return image[start:start + size, start:start + size]


def preview_png(image: np.ndarray, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.2, 4.0), dpi=180)
    im = ax.imshow(np.log10(image + 1e-6), cmap="inferno")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=sorted(CONFIGS), required=True)
    ap.add_argument("--kind", choices=["v4", "mla", "dh", "all"], default="v4")
    ap.add_argument("--zmin", type=float, default=None)
    ap.add_argument("--zmax", type=float, default=None)
    ap.add_argument("--dz", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=None, help="V4 prism wedge angle in degrees")
    ap.add_argument("--bfp_oversample", type=int, default=1,
                    help="Upsample the raw BFP grid before MLA/DH phase mask; use 6 for 20x")
    ap.add_argument("--crop", type=int, default=384)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cfg = CONFIGS[args.config]
    if (args.zmin is not None or args.zmax is not None or args.dz is not None
            or args.alpha is not None):
        cfg = Config(cfg.name, cfg.M, cfg.NA, cfg.immersion_n,
                     cfg.zmin if args.zmin is None else args.zmin,
                     cfg.zmax if args.zmax is None else args.zmax,
                     cfg.dz if args.dz is None else args.dz,
                     cfg.prism_alpha_deg if args.alpha is None else args.alpha)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kinds = ["v4", "mla", "dh"] if args.kind == "all" else [args.kind]
    depths = cfg.depths

    for kind in kinds:
        t0 = time.perf_counter()
        stack = []
        for i, z in enumerate(depths):
            image, r_beam = psf_at_z(float(z), cfg, kind, workers=args.workers,
                                     bfp_oversample=args.bfp_oversample)
            image = center_crop(image, args.crop, offset=0 if kind == "dh" else 5)
            stack.append(image)
            print(json.dumps({"kind": kind, "i": i + 1, "n": len(depths),
                              "z_um": float(z), "r_beam_px": float(r_beam),
                              "seconds": round(time.perf_counter() - t0, 1)}), flush=True)
        stack = np.stack(stack)
        np.savez_compressed(out / f"{cfg.name}_{kind}.npz",
                            stack=stack, depths_um=depths,
                            metadata=json.dumps({"config": asdict(cfg), "kind": kind,
                                                 "crop": args.crop,
                                                 "bfp_oversample": args.bfp_oversample,
                                                 "source": "vector Richards-Wolf",
                                                 "wavelength_um": WAVELENGTH,
                                                 "pixel_um": PIXEL_PITCH,
                                                 "f1": F1, "f2": F2, "fm": FM}))
        if args.preview:
            for z in [depths[0], depths[len(depths) // 2], depths[-1]]:
                idx = int(np.argmin(np.abs(depths - z)))
                preview_png(stack[idx], out / f"{cfg.name}_{kind}_z{z:+.1f}.png",
                            f"{cfg.name} {kind} z={z:+.1f} um")
        print(json.dumps({"kind": kind, "saved": str(out / f'{cfg.name}_{kind}.npz'),
                          "shape": list(stack.shape), "seconds": round(time.perf_counter() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
