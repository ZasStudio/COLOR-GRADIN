#!/usr/bin/env python3
"""Transfiere el color grading de un video (o imagen) de referencia a otro video
usando máscaras, como haría un colorista con qualifiers en DaVinci Resolve.

Capa 1 - máscaras de luminancia: sombras, medios, altas luces.
    Cada zona se iguala por separado (luminancia + color) a la misma zona de la
    referencia. Así se copia, por ejemplo, "sombras teal / luces cálidas".
Capa 2 - máscaras de color: piel, cielo/azules, vegetación/verdes.
    Sobre el resultado anterior, cada rango de color se corrige para parecerse
    al mismo rango de la referencia (tono de piel, color del cielo, verdes...).

Todas las máscaras son suaves (con degradado) y dependen sólo del color del
pixel, así que el grade completo se hornea exactamente en un LUT 3D (.cube)
que puedes usar en Resolve / Premiere / Final Cut / CapCut.

Uso:
    python color_match.py referencia.mp4 mi_video.mp4 -o salida/
"""
import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

LUMA_MASKS = ["sombras", "medios", "altas_luces"]
COLOR_MASKS = ["piel", "cielo_azules", "vegetacion_verdes"]

# Límites para que una zona no se deforme en exceso.
STD_RATIO_MIN, STD_RATIO_MAX = 0.4, 2.5
MIN_COVERAGE = 0.005  # una máscara debe cubrir >=0.5% de la imagen en ambos clips


# ----------------------------------------------------------------------------
# Lectura de fotogramas
# ----------------------------------------------------------------------------

def sample_frames(path, n, max_side=480):
    """Devuelve una lista de fotogramas RGB float32 [0,1] repartidos en el clip."""
    if os.path.splitext(path)[1].lower() in IMAGE_EXTS:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            sys.exit(f"No se pudo leer la imagen: {path}")
        return clean_frames([_prep(img, max_side)])

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"No se pudo abrir el video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frames = []
    if total > 0:
        # Evita el primer y último 5% (fundidos, negros).
        idxs = np.linspace(total * 0.05, total * 0.95, n).astype(int)
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if ok:
                frames.append(_prep(f, max_side))
    if not frames:  # contenedor sin número de frames fiable: leer secuencialmente
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        allf = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            allf.append(f)
        for i in np.linspace(0, len(allf) - 1, min(n, len(allf))).astype(int):
            frames.append(_prep(allf[i], max_side))
    cap.release()
    if not frames:
        sys.exit(f"No se pudieron leer fotogramas de: {path}")
    return clean_frames(frames)


def clean_frames(frames):
    """Quita barras negras (letterbox/pillarbox) y fotogramas casi negros
    (fundidos, placas de título) para que no contaminen el análisis."""
    lum = [f.mean(2) for f in frames]
    # Descarta fotogramas prácticamente negros.
    keep = [i for i, l in enumerate(lum) if np.percentile(l, 90) > 0.08]
    if not keep:
        return frames
    frames = [frames[i] for i in keep]
    lum = [lum[i] for i in keep]
    # Una fila/columna es barra si está negra en TODOS los fotogramas.
    row_max = np.max([np.percentile(l, 98, axis=1) for l in lum], axis=0)
    col_max = np.max([np.percentile(l, 98, axis=0) for l in lum], axis=0)
    rows = np.where(row_max > 0.04)[0]
    cols = np.where(col_max > 0.04)[0]
    if len(rows) < 16 or len(cols) < 16:
        return frames
    h, w = frames[0].shape[:2]
    if rows[0] == 0 and rows[-1] == h - 1 and cols[0] == 0 and cols[-1] == w - 1:
        return frames  # sin barras
    # Margen extra para no incluir bordes redondeados o suavizados de las barras.
    pr, pc = max(2, len(rows) // 50), max(2, len(cols) // 50)
    r0, r1 = rows[0] + pr, rows[-1] + 1 - pr
    c0, c1 = cols[0] + pc, cols[-1] + 1 - pc
    return [f[r0:r1, c0:c1] for f in frames]


def _prep(bgr, max_side):
    h, w = bgr.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


# ----------------------------------------------------------------------------
# Espacio de color y máscaras
# ----------------------------------------------------------------------------

def rgb_to_lab(rgb):
    """(N,3) RGB [0,1] -> (N,3) Lab (L 0..100)."""
    return cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.float32), cv2.COLOR_RGB2Lab).reshape(-1, 3)


def lab_to_rgb(lab):
    rgb = cv2.cvtColor(lab.reshape(-1, 1, 3).astype(np.float32), cv2.COLOR_Lab2RGB).reshape(-1, 3)
    return np.clip(rgb, 0.0, 1.0)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def hue_window(h, center, half, feather):
    d = np.abs((h - center + 180.0) % 360.0 - 180.0)
    return 1.0 - smoothstep(half, half + feather, d)


def luma_masks(lab):
    L = lab[:, 0]
    sh = 1.0 - smoothstep(15.0, 45.0, L)
    hi = smoothstep(55.0, 85.0, L)
    mid = np.clip(1.0 - sh - hi, 0.0, 1.0)
    return {"sombras": sh, "medios": mid, "altas_luces": hi}


def color_masks(lab):
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    C = np.sqrt(a * a + b * b)
    h = np.degrees(np.arctan2(b, a)) % 360.0
    chroma_on = smoothstep(5.0, 14.0, C)
    skin = (hue_window(h, 55.0, 18.0, 15.0) * chroma_on * (1.0 - smoothstep(45.0, 60.0, C))
            * smoothstep(20.0, 35.0, L) * (1.0 - smoothstep(88.0, 97.0, L)))
    sky = hue_window(h, 265.0, 35.0, 20.0) * chroma_on * smoothstep(30.0, 50.0, L)
    veg = hue_window(h, 135.0, 35.0, 20.0) * chroma_on * smoothstep(8.0, 20.0, L)
    return {"piel": skin, "cielo_azules": sky, "vegetacion_verdes": veg}


# ----------------------------------------------------------------------------
# Estadísticas y transferencia
# ----------------------------------------------------------------------------

def weighted_stats(lab, w):
    sw = w.sum()
    if sw < 1e-6:
        return None
    mu = (lab * w[:, None]).sum(0) / sw
    var = (((lab - mu) ** 2) * w[:, None]).sum(0) / sw
    return {"mu": mu, "sd": np.sqrt(var) + 1e-3, "cov": sw / len(w)}


def stats_per_mask(lab, masks):
    return {k: weighted_stats(lab, m) for k, m in masks.items()}


def transfer(lab, src, dst, channel_gain=(1.0, 1.0, 1.0)):
    """Mean/std transfer por canal, con límites y ganancia por canal."""
    ratio = np.clip(dst["sd"] / src["sd"], STD_RATIO_MIN, STD_RATIO_MAX)
    out = (lab - src["mu"]) * ratio + dst["mu"]
    g = np.asarray(channel_gain, dtype=np.float32)
    return lab + (out - lab) * g


class MaskGrade:
    """Grade por máscaras aprendido de (referencia, objetivo)."""

    def __init__(self, ref_lab, tgt_lab, use_color_masks=True, verbose=True):
        self.log = print if verbose else (lambda *a, **k: None)

        # Capa 1: zonas de luminancia.
        self.luma_ref = stats_per_mask(ref_lab, luma_masks(ref_lab))
        self.luma_tgt = stats_per_mask(tgt_lab, luma_masks(tgt_lab))
        self.log("Capa 1 - máscaras de luminancia:")
        for k in LUMA_MASKS:
            r, t = self.luma_ref[k], self.luma_tgt[k]
            ok = r and t and r["cov"] >= MIN_COVERAGE and t["cov"] >= MIN_COVERAGE
            if not ok:
                self.luma_ref[k] = self.luma_tgt[k] = None
            self.log(f"  {k:<18} " + (f"ref {r['cov']*100:5.1f}%  video {t['cov']*100:5.1f}%"
                                         if ok else "sin cobertura suficiente -> se usa grade global"))
        self.global_ref = weighted_stats(ref_lab, np.ones(len(ref_lab), np.float32))
        self.global_tgt = weighted_stats(tgt_lab, np.ones(len(tgt_lab), np.float32))

        # Capa 2: máscaras de color, medidas sobre el resultado de la capa 1.
        self.color_ref, self.color_tgt = {}, {}
        if use_color_masks:
            base = self._luma_layer(tgt_lab)
            cr = stats_per_mask(ref_lab, color_masks(ref_lab))
            ct = stats_per_mask(base, color_masks(base))
            self.log("Capa 2 - máscaras de color:")
            for k in COLOR_MASKS:
                r, t = cr[k], ct[k]
                ok = r and t and r["cov"] >= MIN_COVERAGE and t["cov"] >= MIN_COVERAGE
                if ok:
                    self.color_ref[k], self.color_tgt[k] = r, t
                self.log(f"  {k:<18} " + (f"ref {r['cov']*100:5.1f}%  video {t['cov']*100:5.1f}%"
                                             if ok else "no aparece en ambos clips -> se omite"))

    def _luma_layer(self, lab):
        masks = luma_masks(lab)
        out = np.zeros_like(lab)
        wsum = np.zeros(len(lab), np.float32)
        for k in LUMA_MASKS:
            if self.luma_ref[k] is None:
                continue
            w = masks[k]
            out += transfer(lab, self.luma_tgt[k], self.luma_ref[k]) * w[:, None]
            wsum += w
        # Donde ninguna zona válida cubre el pixel, grade global.
        rest = np.clip(1.0 - wsum, 0.0, 1.0)
        out += transfer(lab, self.global_tgt, self.global_ref) * rest[:, None]
        return out

    def _color_layer(self, lab):
        if not self.color_ref:
            return lab
        masks = color_masks(lab)
        delta = np.zeros_like(lab)
        wsum = np.zeros(len(lab), np.float32)
        for k in self.color_ref:
            w = masks[k]
            # Ajuste fuerte en color (a,b) y suave en luminancia para no romper la capa 1.
            d = transfer(lab, self.color_tgt[k], self.color_ref[k], (0.5, 1.0, 1.0)) - lab
            delta += d * w[:, None]
            wsum += w
        norm = np.maximum(wsum, 1.0)  # si las máscaras se solapan, no sumar de más
        return lab + delta / norm[:, None]

    def apply_lab(self, lab, strength=1.0):
        out = self._color_layer(self._luma_layer(lab))
        return lab + (out - lab) * strength

    def apply_rgb(self, rgb, strength=1.0):
        shape = rgb.shape
        lab = rgb_to_lab(rgb.reshape(-1, 3))
        return lab_to_rgb(self.apply_lab(lab, strength)).reshape(shape)


# ----------------------------------------------------------------------------
# Salidas: LUT, video, previsualizaciones
# ----------------------------------------------------------------------------

def write_cube(grade, path, size, strength):
    g = np.linspace(0.0, 1.0, size, dtype=np.float32)
    # Orden .cube: R varía más rápido, luego G, luego B.
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")
    rgb = np.stack([r.ravel(), gg.ravel(), b.ravel()], axis=1)
    out = grade.apply_rgb(rgb, strength)
    with open(path, "w") as f:
        f.write('TITLE "color_match por mascaras"\n')
        f.write(f"LUT_3D_SIZE {size}\n")
        f.write("DOMAIN_MIN 0.0 0.0 0.0\nDOMAIN_MAX 1.0 1.0 1.0\n")
        for v in out:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def render_video(src, lut, dst, crf):
    lut_arg = lut.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    base = [ffmpeg_exe(), "-y", "-loglevel", "error", "-stats", "-i", src,
            "-vf", f"lut3d=file='{lut_arg}':interp=tetrahedral",
            "-c:v", "libx264", "-crf", str(crf), "-preset", "medium", "-pix_fmt", "yuv420p",
            "-map", "0:v:0", "-map", "0:a?"]
    # Intentar copiar el audio tal cual; si el contenedor no lo acepta, recodificar a AAC.
    if subprocess.run(base + ["-c:a", "copy", dst]).returncode != 0:
        subprocess.run(base + ["-c:a", "aac", "-b:a", "192k", dst], check=True)


def _label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _to_bgr8(rgb):
    return cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _mask_view(rgb, m):
    """Muestra la imagen sólo donde actúa la máscara (resto en gris oscuro)."""
    m = m[..., None]
    return rgb * m + 0.12 * (1 - m)


def save_previews(grade, ref_frame, tgt_frame, out_dir, strength):
    h, w = tgt_frame.shape[:2]
    ref_r = cv2.resize(ref_frame, (w, h), interpolation=cv2.INTER_AREA)
    graded = grade.apply_rgb(tgt_frame, strength)

    row = np.hstack([_label(_to_bgr8(tgt_frame), "ORIGINAL"),
                     _label(_to_bgr8(graded), "RESULTADO"),
                     _label(_to_bgr8(ref_r), "REFERENCIA")])
    cv2.imwrite(os.path.join(out_dir, "comparacion.jpg"), row, [cv2.IMWRITE_JPEG_QUALITY, 92])

    lab = rgb_to_lab(tgt_frame.reshape(-1, 3))
    masks = {**luma_masks(lab), **color_masks(lab)}
    tiles = []
    for k in LUMA_MASKS + COLOR_MASKS:
        m = masks[k].reshape(h, w)
        active = (grade.luma_ref.get(k) is not None) if k in LUMA_MASKS else (k in grade.color_ref)
        tiles.append(_label(_to_bgr8(_mask_view(tgt_frame, m)),
                            f"{k.upper()}" + ("" if active else "  (no usada)")))
    grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
    cv2.imwrite(os.path.join(out_dir, "mascaras.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])


# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Copia el color grading de una referencia a un video usando máscaras.")
    p.add_argument("referencia", help="Video o imagen con el look que quieres copiar")
    p.add_argument("video", help="Video al que aplicar el look")
    p.add_argument("-o", "--out", default="salida", help="Carpeta de salida (por defecto: salida/)")
    p.add_argument("-s", "--strength", type=float, default=1.0, help="Intensidad 0..1 (por defecto 1.0)")
    p.add_argument("--frames", type=int, default=24, help="Fotogramas a muestrear de cada clip")
    p.add_argument("--lut-size", type=int, default=65, choices=[17, 33, 65], help="Resolución del LUT")
    p.add_argument("--sin-mascaras-color", action="store_true", help="Usar sólo las máscaras de luminancia")
    p.add_argument("--solo-lut", action="store_true", help="No renderizar el video, sólo LUT y previews")
    p.add_argument("--crf", type=int, default=16, help="Calidad x264 (menor = mejor, por defecto 16)")
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    print("Leyendo fotogramas...")
    ref_frames = sample_frames(a.referencia, a.frames)
    tgt_frames = sample_frames(a.video, a.frames)
    ref_lab = np.concatenate([rgb_to_lab(f.reshape(-1, 3)) for f in ref_frames])
    tgt_lab = np.concatenate([rgb_to_lab(f.reshape(-1, 3)) for f in tgt_frames])

    grade = MaskGrade(ref_lab, tgt_lab, use_color_masks=not a.sin_mascaras_color)

    name = os.path.splitext(os.path.basename(a.video))[0]
    lut_path = os.path.join(a.out, f"{name}_grade.cube")
    write_cube(grade, lut_path, a.lut_size, a.strength)
    print(f"LUT: {lut_path}")

    save_previews(grade, ref_frames[len(ref_frames) // 2], tgt_frames[len(tgt_frames) // 2], a.out, a.strength)
    print(f"Previews: {os.path.join(a.out, 'comparacion.jpg')}, {os.path.join(a.out, 'mascaras.jpg')}")

    if not a.solo_lut:
        if os.path.splitext(a.video)[1].lower() in IMAGE_EXTS:
            out_img = os.path.join(a.out, f"{name}_graded.png")
            img = cv2.cvtColor(cv2.imread(a.video, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            cv2.imwrite(out_img, _to_bgr8(grade.apply_rgb(img, a.strength)))
            print(f"Imagen: {out_img}")
        else:
            out_vid = os.path.join(a.out, f"{name}_graded.mp4")
            print("Renderizando video...")
            render_video(a.video, lut_path, out_vid, a.crf)
            print(f"Video: {out_vid}")


if __name__ == "__main__":
    main()
