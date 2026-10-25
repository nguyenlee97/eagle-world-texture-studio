"""
Category-Specific 2D Surface Converters for Eagle World Texture & Landmark Studio:
- CAT1_BILLBOARDS: Billboard Dual-Sync (Near + LOD) & UV-Aspect Compensated Converter
- CAT2_MURALS_BANNERS_GRAFFITI: Wall-Grain Blend Mural, Alpha DXT5 Graffiti & Tall Banner Converter
- CAT3_NEON_LED_SCREENS_CASINOS: Sub-Region Atlas Patcher & LED Scanline/Glow Converter
- CAT4_POSTERS_SHOPS_TRANSIT_VENDING: Bus-Stop UV-Unstretch, Outer Frame Mask & Glass Sheen Converter
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from core.rw_txd import RwRasterEntry, build_rw_texnative_chunk


CONVERTER_PRESET_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "CAT1_BILLBOARDS": {
        "id": "CAT1_BILLBOARDS",
        "title": "Converter A — Biển Quảng Cáo Lớn (Dual-Sync Near + LOD)",
        "badge": "CAT1 · Billboard",
        "description": (
            "Tự động bù tỷ lệ khung 3D (.DFF) chống méo hình khi raster gốc lưu vuông 1:1, "
            "nén chuẩn DXT1 + Mipmaps (tiết kiệm 70–85% dung lượng so với GROUP_B1), "
            "và tự động tạo ảnh thu nhỏ đồng bộ vào file *_lod.txd cặp đôi."
        ),
        "default_options": {
            "resolution_tier": "recommended",  # recommended (512), hd_hero (1024), original, compact (256)
            "fit_mode": "cover",               # cover, contain, stretch
            "uv_aspect_compensation": True,
            "auto_sync_lod": True,
            "print_weathering": 0.08,
            "border_frame_px": 0,
            "generate_mipmaps": True,
        },
    },
    "CAT2_MURALS_BANNERS_GRAFFITI": {
        "id": "CAT2_MURALS_BANNERS_GRAFFITI",
        "title": "Converter B — Tranh Tường, Graffiti & Banner Cột Điện",
        "badge": "CAT2 · Mural / Tag / Banner",
        "description": (
            "Hỗ trợ 3 chế độ: (1) Hòa trộn vân gạch/bê tông từ tường gốc để tranh anime nhìn như sơn lên tường thật; "
            "(2) Giữ kênh Alpha rỗng nền chuẩn DXT5 cho Graffiti; (3) Khung cuộn dọc 1:2 / 1:4 cho Banner."
        ),
        "default_options": {
            "sub_mode": "auto",                # auto, mural_wall_blend, graffiti_alpha, tall_banner
            "resolution_tier": "recommended",
            "fit_mode": "cover",
            "uv_aspect_compensation": True,
            "wall_blend_strength": 0.18,       # 0.0 .. 0.45
            "fabric_shade_strength": 0.12,
            "auto_key_bg": False,
            "generate_mipmaps": True,
        },
    },
    "CAT3_NEON_LED_SCREENS_CASINOS": {
        "id": "CAT3_NEON_LED_SCREENS_CASINOS",
        "title": "Converter C — Màn Hình LED, Bảng Đèn Neon & Atlas Casino",
        "badge": "CAT3 · Neon / LED / Atlas",
        "description": (
            "Cho phép thay toàn bộ màn hình hoặc chỉ thay đúng 1 ô vùng con (Sub-Region / UV Island) "
            "trên tấm Atlas ghép nhiều bảng hiệu, kèm tùy chọn lưới bóng đèn LED Scanline & tăng độ rực ban đêm."
        ),
        "default_options": {
            "resolution_tier": "recommended",
            "fit_mode": "cover",
            "uv_aspect_compensation": True,
            "use_sub_region": False,
            "sub_region_rect": [0.0, 0.0, 1.0, 1.0],  # [u0, v0, u1, v1]
            "led_scanline_strength": 0.14,
            "vibrance_boost": 1.15,
            "preserve_alpha": True,
            "generate_mipmaps": True,
        },
    },
    "CAT4_POSTERS_SHOPS_TRANSIT_VENDING": {
        "id": "CAT4_POSTERS_SHOPS_TRANSIT_VENDING",
        "title": "Converter D — Poster, Nhà Chờ Xe Buýt, Mặt Tiền & Máy Bán Nước",
        "badge": "CAT4 · Poster / Bus Stop / Vending",
        "description": (
            "Tự động khử độ kéo giãn UV của khung poster nhà chờ xe buýt (như CJ_BS_MENU5 vuông 256×256 kéo dọc), "
            "giữ viền khung kim loại/nút bấm máy bán nước xung quanh, và phủ bóng kính nhẹ."
        ),
        "default_options": {
            "resolution_tier": "recommended",
            "fit_mode": "cover",
            "uv_aspect_compensation": True,
            "preserve_border_pct": 0.0,        # 0.0 .. 0.25 (keep original outer border/buttons)
            "glass_sheen_strength": 0.10,      # 0.0 .. 0.30
            "generate_mipmaps": True,
        },
    },
}


def _nearest_pot(val: int, min_v: int = 32, max_v: int = 1024) -> int:
    val = max(min_v, min(max_v, int(val)))
    p = 1 << int(round(math.log2(val)))
    return max(min_v, min(max_v, p))


def compute_target_pot_dimensions(
    orig_w: int,
    orig_h: int,
    resolution_tier: str = "recommended",
    category: str = "CAT1_BILLBOARDS",
) -> Tuple[int, int]:
    """
    Determine the optimal Power-of-Two (W, H) dimensions that preserve the original
    raster's stored W:H ratio while adhering to the performance budget envelope.
    """
    w = _nearest_pot(orig_w, 32, 1024)
    h = _nearest_pot(orig_h, 32, 1024)
    if resolution_tier == "original":
        return w, h

    max_side_map = {
        "compact": 256,
        "recommended": 512,
        "hd_hero": 1024,
    }
    target_max = max_side_map.get(resolution_tier, 512)

    # For small graffiti tags (128x128) or narrow scroll strips, don't bloat beyond 256/512 unnecessarily
    if category == "CAT2_MURALS_BANNERS_GRAFFITI" and max(orig_w, orig_h) <= 128 and resolution_tier == "recommended":
        target_max = 256

    cur_max = max(w, h)
    if cur_max > 0:
        scale = target_max / cur_max
        w = _nearest_pot(int(round(w * scale)), 32, 1024)
        h = _nearest_pot(int(round(h * scale)), 32, 1024)
    return w, h


def fit_image_with_aspect(
    src_img: Image.Image,
    out_w: int,
    out_h: int,
    display_aspect_wh: Optional[float] = None,
    fit_mode: str = "cover",
    bg_color: Tuple[int, int, int, int] = (18, 22, 30, 255),
) -> Image.Image:
    """
    Fit `src_img` so that when rendered on a 3D surface of aspect ratio `display_aspect_wh`
    (even if stored in a raster of pixel dimensions `out_w × out_h`), it looks un-stretched!
    """
    rgba = src_img.convert("RGBA")
    if fit_mode == "stretch" or not display_aspect_wh or display_aspect_wh <= 0:
        if fit_mode == "cover":
            return ImageOps.fit(rgba, (out_w, out_h), method=Image.Resampling.LANCZOS)
        elif fit_mode == "contain":
            contained = ImageOps.contain(rgba, (out_w, out_h), method=Image.Resampling.LANCZOS)
            canvas = Image.new("RGBA", (out_w, out_h), bg_color)
            canvas.alpha_composite(
                contained,
                ((out_w - contained.width) // 2, (out_h - contained.height) // 2),
            )
            return canvas
        return rgba.resize((out_w, out_h), Image.Resampling.LANCZOS)

    # Build an intermediate composition canvas whose aspect ratio matches the true 3D surface!
    base_px = max(out_w, out_h, 512)
    if display_aspect_wh >= 1.0:
        comp_w = int(round(base_px))
        comp_h = max(32, int(round(base_px / display_aspect_wh)))
    else:
        comp_h = int(round(base_px))
        comp_w = max(32, int(round(base_px * display_aspect_wh)))

    if fit_mode == "cover":
        composed = ImageOps.fit(rgba, (comp_w, comp_h), method=Image.Resampling.LANCZOS)
    elif fit_mode == "contain":
        contained = ImageOps.contain(rgba, (comp_w, comp_h), method=Image.Resampling.LANCZOS)
        composed = Image.new("RGBA", (comp_w, comp_h), bg_color)
        composed.alpha_composite(
            contained,
            ((comp_w - contained.width) // 2, (comp_h - contained.height) // 2),
        )
    else:
        composed = rgba.resize((comp_w, comp_h), Image.Resampling.LANCZOS)

    # Now resample into the stored raster dimensions (out_w, out_h)
    if (comp_w, comp_h) != (out_w, out_h):
        return composed.resize((out_w, out_h), Image.Resampling.LANCZOS)
    return composed


def _apply_wall_grain_blend(
    new_img: Image.Image,
    orig_wall_img: Image.Image,
    strength: float = 0.18,
) -> Image.Image:
    """
    Blend the brick/concrete surface relief of `orig_wall_img` onto `new_img`
    so a Wibu mural looks painted onto the actual building wall.
    """
    strength = float(np.clip(strength, 0.0, 0.65))
    if strength <= 0.01:
        return new_img

    w, h = new_img.size
    wall_resized = orig_wall_img.convert("RGB").resize((w, h), Image.Resampling.BILINEAR)
    wall_gray = np.asarray(wall_resized.convert("L"), dtype=np.float32)

    # Extract high/mid frequency wall grain relative to local mean
    blur_img = wall_resized.convert("L").filter(ImageFilter.GaussianBlur(radius=max(2, min(w, h) // 16)))
    blur_arr = np.asarray(blur_img, dtype=np.float32)
    grain = (wall_gray - blur_arr) / 128.0  # roughly [-1, +1]
    lum_factor = (wall_gray - 128.0) / 255.0

    arr = np.asarray(new_img.convert("RGBA"), dtype=np.float32).copy()
    mod = 1.0 + strength * (grain * 1.35 + lum_factor * 0.45)
    arr[:, :, :3] = np.clip(arr[:, :, :3] * mod[:, :, None], 0.0, 255.0)
    return Image.fromarray(arr.astype(np.uint8), mode="RGBA")


def _apply_fabric_banner_shading(img: Image.Image, strength: float = 0.12) -> Image.Image:
    strength = float(np.clip(strength, 0.0, 0.45))
    if strength <= 0.01:
        return img
    arr = np.asarray(img.convert("RGBA"), dtype=np.float32).copy()
    h, w, _ = arr.shape
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32)
    wave = np.sin(ys[:, None] * math.pi * 6.0 + xs[None, :] * 1.5) * 0.6 + np.cos(xs[None, :] * math.pi * 2.0) * 0.4
    mod = 1.0 + strength * wave[:, :, None]
    arr[:, :, :3] = np.clip(arr[:, :, :3] * mod, 0.0, 255.0)
    return Image.fromarray(arr.astype(np.uint8), mode="RGBA")


def _apply_led_scanline_effect(
    img: Image.Image,
    scanline_strength: float = 0.14,
    vibrance_boost: float = 1.15,
) -> Image.Image:
    rgba = img.convert("RGBA")
    if abs(vibrance_boost - 1.0) > 0.02:
        r, g, b, a = rgba.split()
        rgb = Image.merge("RGB", (r, g, b))
        rgb = ImageEnhance.Color(rgb).enhance(float(np.clip(vibrance_boost, 0.8, 1.8)))
        rgb = ImageEnhance.Contrast(rgb).enhance(1.06)
        nr, ng, nb = rgb.split()
        rgba = Image.merge("RGBA", (nr, ng, nb, a))

    scanline_strength = float(np.clip(scanline_strength, 0.0, 0.50))
    if scanline_strength <= 0.01:
        return rgba

    arr = np.asarray(rgba, dtype=np.float32).copy()
    h, w, _ = arr.shape
    row_mask = np.ones((h, 1, 1), dtype=np.float32)
    row_mask[1::2, 0, 0] = 1.0 - scanline_strength
    col_mask = np.ones((1, w, 1), dtype=np.float32)
    col_mask[0, 3::4, 0] = 1.0 - (scanline_strength * 0.45)
    arr[:, :, :3] = np.clip(arr[:, :, :3] * row_mask * col_mask, 0.0, 255.0)
    return Image.fromarray(arr.astype(np.uint8), mode="RGBA")


def _apply_glass_sheen(img: Image.Image, strength: float = 0.10) -> Image.Image:
    strength = float(np.clip(strength, 0.0, 0.40))
    if strength <= 0.01:
        return img
    arr = np.asarray(img.convert("RGBA"), dtype=np.float32).copy()
    h, w, _ = arr.shape
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, h, dtype=np.float32),
        np.linspace(0.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )
    diag = xx * 0.75 + yy * 0.55
    streak1 = np.exp(-((diag - 0.42) ** 2) / 0.0045)
    streak2 = np.exp(-((diag - 0.54) ** 2) / 0.0012) * 0.65
    sheen = (streak1 + streak2)[:, :, None] * (255.0 * strength)
    arr[:, :, :3] = np.clip(arr[:, :, :3] + sheen, 0.0, 255.0)
    return Image.fromarray(arr.astype(np.uint8), mode="RGBA")


def _auto_key_solid_background(img: Image.Image, tolerance: int = 28) -> Image.Image:
    """If an image has an opaque uniform corner background, key it out into alpha."""
    arr = np.asarray(img.convert("RGBA"), dtype=np.int32).copy()
    # Check if already has meaningful transparency
    if (arr[:, :, 3] < 240).mean() > 0.05:
        return img.convert("RGBA")
    bg_rgb = arr[0, 0, :3]
    dist = np.sqrt(((arr[:, :, :3] - bg_rgb[None, None, :]) ** 2).sum(axis=-1))
    alpha = np.clip((dist - tolerance * 0.6) / max(1.0, tolerance * 0.8) * 255.0, 0.0, 255.0)
    arr[:, :, 3] = np.minimum(arr[:, :, 3], alpha.astype(np.int32))
    return Image.fromarray(arr.astype(np.uint8), mode="RGBA")


def convert_surface_raster(
    category: str,
    user_img: Image.Image,
    orig_entry: RwRasterEntry,
    orig_decoded_img: Image.Image,
    dff_tex_info: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Any]] = None,
    lod_orig_entry: Optional[RwRasterEntry] = None,
) -> Dict[str, Any]:
    """
    Execute the category-specific surface conversion pipeline.
    Returns:
    {
      "near_entry": RwRasterEntry,
      "near_preview_pil": Image.Image,
      "lod_entry": Optional[RwRasterEntry],
      "lod_preview_pil": Optional[Image.Image],
      "report": { ... metrics, dimensions, format, size_kb, uv_compensation_applied ... }
    }
    """
    schema = CONVERTER_PRESET_SCHEMAS.get(category, CONVERTER_PRESET_SCHEMAS["CAT1_BILLBOARDS"])
    opts = dict(schema["default_options"])
    if options:
        opts.update(options)

    res_tier = str(opts.get("resolution_tier", "recommended"))
    fit_mode = str(opts.get("fit_mode", "cover"))
    use_uv_comp = bool(opts.get("uv_aspect_compensation", True))
    gen_mips = bool(opts.get("generate_mipmaps", True))

    out_w, out_h = compute_target_pot_dimensions(
        orig_entry.width, orig_entry.height, resolution_tier=res_tier, category=category
    )
    stored_aspect_wh = out_w / max(1, out_h)

    # Determine effective 3D display aspect ratio from DFF analysis if available
    eff_aspect_wh = stored_aspect_wh
    uv_comp_note = "Giữ nguyên tỷ lệ gốc của Raster"
    if use_uv_comp and dff_tex_info:
        dff_eff = float(dff_tex_info.get("effective_tex_aspect_wh") or 0.0)
        if 0.15 <= dff_eff <= 8.0 and abs(math.log(dff_eff / stored_aspect_wh)) > 0.12:
            eff_aspect_wh = dff_eff
            uv_comp_note = (
                f"Đã bù giãn UV khung 3D: cắt theo khung hiển thị thực "
                f"{dff_tex_info.get('effective_tex_aspect_label', f'{dff_eff:.2f}:1')} "
                f"rồi đóng gói vào raster {out_w}×{out_h}"
            )

    dxt_format = "DXT1"

    # -----------------------------------------------------------------------
    # Category 1: Billboards (Near + Auto-Sync LOD)
    # -----------------------------------------------------------------------
    if category == "CAT1_BILLBOARDS":
        canvas = fit_image_with_aspect(
            user_img, out_w, out_h, display_aspect_wh=eff_aspect_wh, fit_mode=fit_mode
        )
        weathering = float(opts.get("print_weathering", 0.08))
        if weathering > 0.01:
            canvas = _apply_wall_grain_blend(canvas, orig_decoded_img, strength=weathering * 0.6)
        border_px = int(opts.get("border_frame_px", 0))
        if border_px > 0:
            draw = ImageDraw.Draw(canvas)
            for b in range(min(border_px, min(out_w, out_h) // 8)):
                draw.rectangle([b, b, out_w - 1 - b, out_h - 1 - b], outline=(24, 26, 30, 255))
        dxt_format = "DXT5" if orig_entry.has_alpha and bool(opts.get("preserve_alpha", False)) else "DXT1"

    # -----------------------------------------------------------------------
    # Category 2: Murals, Graffiti Tags & Tall Banners
    # -----------------------------------------------------------------------
    elif category == "CAT2_MURALS_BANNERS_GRAFFITI":
        sub_mode = str(opts.get("sub_mode", "auto"))
        if sub_mode == "auto":
            if orig_entry.has_alpha or "tag" in orig_entry.name.lower() or "graff" in orig_entry.name.lower():
                sub_mode = "graffiti_alpha"
            elif out_h >= out_w * 2:
                sub_mode = "tall_banner"
            else:
                sub_mode = "mural_wall_blend"

        if sub_mode == "graffiti_alpha":
            src_rgba = user_img.convert("RGBA")
            if bool(opts.get("auto_key_bg", False)):
                src_rgba = _auto_key_solid_background(src_rgba)
            canvas = fit_image_with_aspect(
                src_rgba,
                out_w,
                out_h,
                display_aspect_wh=eff_aspect_wh,
                fit_mode=fit_mode,
                bg_color=(0, 0, 0, 0),
            )
            dxt_format = "DXT5"
        elif sub_mode == "tall_banner":
            canvas = fit_image_with_aspect(
                user_img, out_w, out_h, display_aspect_wh=eff_aspect_wh, fit_mode=fit_mode
            )
            canvas = _apply_fabric_banner_shading(canvas, strength=float(opts.get("fabric_shade_strength", 0.12)))
            dxt_format = "DXT5" if orig_entry.has_alpha else "DXT1"
        else:
            canvas = fit_image_with_aspect(
                user_img, out_w, out_h, display_aspect_wh=eff_aspect_wh, fit_mode=fit_mode
            )
            canvas = _apply_wall_grain_blend(
                canvas, orig_decoded_img, strength=float(opts.get("wall_blend_strength", 0.18))
            )
            dxt_format = "DXT1"

    # -----------------------------------------------------------------------
    # Category 3: Neon Signs, LED Jumbotrons & Sub-Region Casino Atlases
    # -----------------------------------------------------------------------
    elif category == "CAT3_NEON_LED_SCREENS_CASINOS":
        use_sub = bool(opts.get("use_sub_region", False))
        sub_rect = opts.get("sub_region_rect") or [0.0, 0.0, 1.0, 1.0]
        if use_sub and len(sub_rect) == 4:
            u0, v0, u1, v1 = [float(np.clip(v, 0.0, 1.0)) for v in sub_rect]
            if u1 - u0 > 0.03 and v1 - v0 > 0.03:
                base_atlas = orig_decoded_img.convert("RGBA").resize((out_w, out_h), Image.Resampling.LANCZOS)
                px0, py0 = int(round(u0 * out_w)), int(round(v0 * out_h))
                px1, py1 = max(px0 + 4, int(round(u1 * out_w))), max(py0 + 4, int(round(v1 * out_h)))
                sub_w, sub_h = px1 - px0, py1 - py0
                patch = fit_image_with_aspect(
                    user_img, sub_w, sub_h, display_aspect_wh=(sub_w / max(1, sub_h)), fit_mode=fit_mode
                )
                patch = _apply_led_scanline_effect(
                    patch,
                    scanline_strength=float(opts.get("led_scanline_strength", 0.14)),
                    vibrance_boost=float(opts.get("vibrance_boost", 1.15)),
                )
                base_atlas.paste(patch, (px0, py0))
                canvas = base_atlas
                uv_comp_note = f"Đã thay vùng con Atlas [{u0:.2f}, {v0:.2f} → {u1:.2f}, {v1:.2f}] ({sub_w}×{sub_h}px), giữ nguyên các chữ đèn Neon khác"
            else:
                canvas = fit_image_with_aspect(
                    user_img, out_w, out_h, display_aspect_wh=eff_aspect_wh, fit_mode=fit_mode
                )
        else:
            canvas = fit_image_with_aspect(
                user_img, out_w, out_h, display_aspect_wh=eff_aspect_wh, fit_mode=fit_mode
            )
            canvas = _apply_led_scanline_effect(
                canvas,
                scanline_strength=float(opts.get("led_scanline_strength", 0.14)),
                vibrance_boost=float(opts.get("vibrance_boost", 1.15)),
            )
        dxt_format = "DXT5" if (orig_entry.has_alpha and bool(opts.get("preserve_alpha", True))) else "DXT1"

    # -----------------------------------------------------------------------
    # Category 4: Posters, Bus Stops, Storefronts & Vending Machines
    # -----------------------------------------------------------------------
    else:
        border_pct = float(np.clip(float(opts.get("preserve_border_pct", 0.0)), 0.0, 0.35))
        if border_pct > 0.01:
            base_frame = orig_decoded_img.convert("RGBA").resize((out_w, out_h), Image.Resampling.LANCZOS)
            mx = int(round(out_w * border_pct))
            my = int(round(out_h * border_pct))
            iw = max(8, out_w - 2 * mx)
            ih = max(8, out_h - 2 * my)
            inner_aspect = eff_aspect_wh * (iw / max(1, out_w)) / (ih / max(1, out_h))
            inner_poster = fit_image_with_aspect(
                user_img, iw, ih, display_aspect_wh=inner_aspect, fit_mode=fit_mode
            )
            inner_poster = _apply_glass_sheen(
                inner_poster, strength=float(opts.get("glass_sheen_strength", 0.10))
            )
            base_frame.paste(inner_poster, (mx, my))
            canvas = base_frame
            uv_comp_note += f" | Giữ {int(border_pct * 100)}% khung viền/nút bấm gốc"
        else:
            canvas = fit_image_with_aspect(
                user_img, out_w, out_h, display_aspect_wh=eff_aspect_wh, fit_mode=fit_mode
            )
            canvas = _apply_glass_sheen(
                canvas, strength=float(opts.get("glass_sheen_strength", 0.10))
            )
        dxt_format = "DXT5" if orig_entry.has_alpha else "DXT1"

    near_entry = build_rw_texnative_chunk(
        name=orig_entry.name,
        pil_img=canvas,
        dxt_format=dxt_format,
        generate_mipmaps=gen_mips,
        mask_name=orig_entry.mask_name if dxt_format == "DXT5" else "",
        filter_flags=orig_entry.filter_flags,
        version=orig_entry.version,
    )

    # Auto-Sync LOD raster if requested and available
    lod_entry: Optional[RwRasterEntry] = None
    lod_preview_pil: Optional[Image.Image] = None
    if bool(opts.get("auto_sync_lod", True)) and lod_orig_entry is not None:
        lw = _nearest_pot(lod_orig_entry.width, 16, 128)
        lh = _nearest_pot(lod_orig_entry.height, 16, 128)
        lod_preview_pil = canvas.resize((lw, lh), Image.Resampling.LANCZOS)
        lod_dxt = "DXT5" if lod_orig_entry.has_alpha else "DXT1"
        lod_entry = build_rw_texnative_chunk(
            name=lod_orig_entry.name,
            pil_img=lod_preview_pil,
            dxt_format=lod_dxt,
            generate_mipmaps=True,
            mask_name=lod_orig_entry.mask_name if lod_dxt == "DXT5" else "",
            filter_flags=lod_orig_entry.filter_flags,
            version=lod_orig_entry.version,
        )

    before_kb = round(orig_entry.total_chunk_bytes / 1024.0, 2)
    after_kb = round(near_entry.total_chunk_bytes / 1024.0, 2)
    delta_kb = round(after_kb - before_kb, 2)
    saved_pct = round((1.0 - after_kb / max(0.01, before_kb)) * 100.0, 1)

    return {
        "near_entry": near_entry,
        "near_preview_pil": canvas,
        "lod_entry": lod_entry,
        "lod_preview_pil": lod_preview_pil,
        "report": {
            "raster_name": orig_entry.name,
            "category": category,
            "orig_dimensions": f"{orig_entry.width}×{orig_entry.height}",
            "new_dimensions": f"{near_entry.width}×{near_entry.height}",
            "orig_format": orig_entry.format_label,
            "new_format": near_entry.format_label,
            "orig_mipmaps": orig_entry.mip_levels,
            "new_mipmaps": near_entry.mip_levels,
            "before_kb": before_kb,
            "after_kb": after_kb,
            "delta_kb": delta_kb,
            "saved_pct": saved_pct,
            "uv_compensation_note": uv_comp_note,
            "lod_synced": lod_entry is not None,
            "lod_raster_name": lod_entry.name if lod_entry else None,
            "lod_dimensions": f"{lod_entry.width}×{lod_entry.height}" if lod_entry else None,
            "lod_size_kb": round(lod_entry.total_chunk_bytes / 1024.0, 2) if lod_entry else 0.0,
        },
    }
