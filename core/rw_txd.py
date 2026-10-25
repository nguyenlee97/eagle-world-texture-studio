"""
RenderWare .TXD Binary Parser, Vectorized DXT1/DXT3/DXT5 Decoder & Encoder
for Eagle World Texture & Landmark Studio.

Supports:
- Reading any RenderWare 0x16 TexDictionary (D3D8 platform 8 & D3D9 platform 9).
- Decoding DXT1, DXT2/DXT3, DXT4/DXT5, A8R8G8B8 (21), X8R8G8B8 (22), R5G6B5 (23),
  X1R5G5B5 (24), A1R5G5B5 (25), A4R4G4B4 (26), P8/PAL8 (0x2000), LUM8 to PIL RGBA Image.
- Encoding PIL Images into DXT1 or DXT5 RenderWare 0x15 TextureNative chunks with full
  Power-of-Two mipmap chains (via Microsoft DirectXTex texconv.exe when available, or
  built-in vectorized NumPy DXT1/DXT5 compressor fallback).
- Losslessly replacing one or more named rasters inside an existing .TXD while keeping
  all sibling rasters and extension chunks untouched.
"""

from __future__ import annotations

import io
import math
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

RW_VERSION_SA = 0x1803FFFF
RW_STRUCT = 0x01
RW_EXTENSION = 0x03
RW_TEXNATIVE = 0x15
RW_TEXDICT = 0x16

FOURCC_DXT1 = 0x31545844  # b'DXT1'
FOURCC_DXT2 = 0x32545844  # b'DXT2'
FOURCC_DXT3 = 0x33545844  # b'DXT3'
FOURCC_DXT4 = 0x34545844  # b'DXT4'
FOURCC_DXT5 = 0x35545844  # b'DXT5'

TEXCONV_DEFAULT_PATH = Path(
    r"C:\Users\LENOVO\Desktop\GTA\tools\skin-viewer\vendor\directxtex\texconv.exe"
)


@dataclass
class RwRasterEntry:
    index: int
    name: str
    mask_name: str
    platform_id: int
    filter_flags: int
    raster_format: int
    d3d_format: int
    width: int
    height: int
    depth: int
    mip_levels: int
    raster_type: int
    compression: int
    format_label: str
    has_alpha: bool
    is_pot: bool
    aspect_label: str
    top_level_bytes: int
    total_chunk_bytes: int
    raw_chunk: bytes  # Full 0x15 chunk including 12-byte header
    version: int


def _is_pot(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _aspect_str(w: int, h: int) -> str:
    if w <= 0 or h <= 0:
        return "1:1"
    g = math.gcd(w, h)
    return f"{w // g}:{h // g}"


def classify_raster_format(
    platform_id: int,
    raster_fmt: int,
    d3d_or_alpha: int,
    depth: int,
    comp: int,
) -> Tuple[str, bool]:
    base_fmt = raster_fmt & 0x0F00
    has_pal8 = (raster_fmt & 0x2000) != 0
    has_pal4 = (raster_fmt & 0x4000) != 0
    has_alpha = (base_fmt in (0x0100, 0x0300, 0x0500)) or (
        platform_id == 9 and (comp & 0x01) != 0
    )

    if has_pal8:
        return "PAL8", has_alpha
    if has_pal4:
        return "PAL4", has_alpha

    if d3d_or_alpha == FOURCC_DXT1 or comp == 1 or (platform_id == 9 and (comp & 0x08) and d3d_or_alpha == FOURCC_DXT1):
        return "DXT1", has_alpha
    if d3d_or_alpha in (FOURCC_DXT2, FOURCC_DXT3) or comp == 3:
        return "DXT3", True
    if d3d_or_alpha in (FOURCC_DXT4, FOURCC_DXT5) or comp == 5:
        return "DXT5", True

    if platform_id == 8 and comp in (1, 2, 3, 4, 5):
        if comp == 1:
            return "DXT1", has_alpha
        if comp in (2, 3):
            return "DXT3", True
        return "DXT5", True

    if platform_id == 9 and (comp & 0x08):
        if base_fmt in (0x0200, 0x0600) and not (comp & 0x01):
            return "DXT1", False
        return "DXT3", True

    if d3d_or_alpha == 21 or (depth == 32 and base_fmt == 0x0500):
        return "RGBA32", True
    if d3d_or_alpha == 22 or (depth in (24, 32) and base_fmt == 0x0600):
        return "RGB32", False
    if d3d_or_alpha == 23 or base_fmt == 0x0200:
        return "R5G6B5", False
    if d3d_or_alpha == 25 or base_fmt == 0x0100:
        return "A1R5G5B5", True
    if d3d_or_alpha == 26 or base_fmt == 0x0300:
        return "A4R4G4B4", True
    if base_fmt == 0x0400:
        return "LUM8", False
    return f"RAW_{depth}B", has_alpha


def rw_pack_chunk(chunk_type: int, payload: bytes, version: int = RW_VERSION_SA) -> bytes:
    return struct.pack("<III", chunk_type, len(payload), version) + payload


def parse_txd_bytes(data: bytes) -> Tuple[int, int, List[RwRasterEntry], bytes]:
    """
    Parse a RenderWare .TXD buffer.
    Returns (root_version, device_id, list_of_RwRasterEntry, trailing_extension_bytes).
    """
    if len(data) < 28:
        raise ValueError(f"TXD buffer too small ({len(data)} bytes)")

    root_type, root_size, root_ver = struct.unpack_from("<III", data, 0)
    if root_type != RW_TEXDICT:
        raise ValueError(f"Invalid TXD root chunk 0x{root_type:X}")

    st_type, st_size, st_ver = struct.unpack_from("<III", data, 12)
    if st_type != RW_STRUCT or st_size < 4:
        raise ValueError("Invalid TXD struct header")

    tex_count, device_id = struct.unpack_from("<HH", data, 24)
    pos = 24 + st_size
    end = min(len(data), 12 + root_size)

    entries: List[RwRasterEntry] = []
    idx = 0
    while pos + 12 <= end:
        c_type, c_size, c_ver = struct.unpack_from("<III", data, pos)
        chunk_end = pos + 12 + c_size
        if chunk_end > len(data):
            break
        if c_type == RW_TEXNATIVE and c_size >= 100:
            raw_chunk = data[pos:chunk_end]
            ist_type, ist_size, ist_ver = struct.unpack_from("<III", raw_chunk, 12)
            if ist_type == RW_STRUCT and ist_size >= 88:
                (
                    plat,
                    filt,
                    name_b,
                    mask_b,
                    rfmt,
                    d3dfmt,
                    w,
                    h,
                    depth,
                    mips,
                    rtype,
                    comp,
                ) = struct.unpack_from("<II32s32sIIHHBBBB", raw_chunk, 24)
                name = name_b.split(b"\x00", 1)[0].decode("latin1", errors="ignore").strip()
                mask = mask_b.split(b"\x00", 1)[0].decode("latin1", errors="ignore").strip()
                fmt_label, has_alpha = classify_raster_format(plat, rfmt, d3dfmt, depth, comp)
                if mask:
                    has_alpha = True

                # Offset 112 in raw_chunk is either palette (if PAL8/PAL4) or uint32 top_level_size
                pal_bytes = 1024 if (rfmt & 0x2000) else (64 if (rfmt & 0x4000) else 0)
                sz_off = 112 + pal_bytes
                top_bytes = (
                    struct.unpack_from("<I", raw_chunk, sz_off)[0]
                    if sz_off + 4 <= len(raw_chunk)
                    else 0
                )

                entries.append(
                    RwRasterEntry(
                        index=idx,
                        name=name,
                        mask_name=mask,
                        platform_id=plat,
                        filter_flags=filt,
                        raster_format=rfmt,
                        d3d_format=d3dfmt,
                        width=w,
                        height=h,
                        depth=depth,
                        mip_levels=max(1, mips),
                        raster_type=rtype,
                        compression=comp,
                        format_label=fmt_label,
                        has_alpha=has_alpha,
                        is_pot=_is_pot(w) and _is_pot(h),
                        aspect_label=_aspect_str(w, h),
                        top_level_bytes=top_bytes,
                        total_chunk_bytes=len(raw_chunk),
                        raw_chunk=raw_chunk,
                        version=c_ver,
                    )
                )
                idx += 1
        elif c_type == RW_EXTENSION:
            tail_ext = data[pos:chunk_end]
            return root_ver, device_id, entries, tail_ext

        if c_size == 0 and c_type != RW_EXTENSION:
            break
        pos = chunk_end

    default_ext = rw_pack_chunk(RW_EXTENSION, b"", root_ver)
    return root_ver, device_id, entries, default_ext


def serialize_txd_bytes(
    entries: List[RwRasterEntry],
    root_version: int = RW_VERSION_SA,
    device_id: int = 2,
    tail_ext: Optional[bytes] = None,
) -> bytes:
    """Serialize RwRasterEntry list back into a valid RenderWare 0x16 .TXD file."""
    st_payload = struct.pack("<HH", len(entries), device_id if device_id in (1, 2) else 2)
    st_chunk = rw_pack_chunk(RW_STRUCT, st_payload, root_version)
    body = b"".join(e.raw_chunk for e in entries)
    ext = tail_ext if tail_ext is not None else rw_pack_chunk(RW_EXTENSION, b"", root_version)
    return rw_pack_chunk(RW_TEXDICT, st_chunk + body + ext, root_version)


# ---------------------------------------------------------------------------
# Fast Vectorized NumPy DXT1 / DXT3 / DXT5 Decompression
# ---------------------------------------------------------------------------

def _decode_rgb565_array(c: np.ndarray) -> np.ndarray:
    """Convert uint16 RGB565 array (N,) into uint8 RGB array (N, 3)."""
    r = ((c >> 11) & 0x1F).astype(np.uint32)
    g = ((c >> 5) & 0x3F).astype(np.uint32)
    b = (c & 0x1F).astype(np.uint32)
    r = (r * 255 + 15) // 31
    g = (g * 255 + 31) // 63
    b = (b * 255 + 15) // 31
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def _decompress_dxt_color_blocks(color_blocks_8b: np.ndarray, is_dxt1: bool) -> np.ndarray:
    """
    Decompress N 8-byte DXT color blocks (shape (N, 8) uint8) into (N, 4, 4, 4) RGBA uint8.
    """
    n_blocks = color_blocks_8b.shape[0]
    u16 = color_blocks_8b[:, :4].view(np.uint16).reshape(n_blocks, 2)
    c0_16 = u16[:, 0]
    c1_16 = u16[:, 1]
    bits = color_blocks_8b[:, 4:8].view(np.uint32).reshape(n_blocks)

    rgb0 = _decode_rgb565_array(c0_16).astype(np.int32)  # (N, 3)
    rgb1 = _decode_rgb565_array(c1_16).astype(np.int32)  # (N, 3)

    palette = np.zeros((n_blocks, 4, 4), dtype=np.uint8)
    palette[:, 0, :3] = rgb0
    palette[:, 0, 3] = 255
    palette[:, 1, :3] = rgb1
    palette[:, 1, 3] = 255

    if is_dxt1:
        opaque_mask = c0_16 > c1_16
        # When c0 > c1: 4 colors
        palette[opaque_mask, 2, :3] = (2 * rgb0[opaque_mask] + rgb1[opaque_mask] + 1) // 3
        palette[opaque_mask, 2, 3] = 255
        palette[opaque_mask, 3, :3] = (rgb0[opaque_mask] + 2 * rgb1[opaque_mask] + 1) // 3
        palette[opaque_mask, 3, 3] = 255
        # When c0 <= c1: 3 colors + transparent black
        trans_mask = ~opaque_mask
        palette[trans_mask, 2, :3] = (rgb0[trans_mask] + rgb1[trans_mask]) // 2
        palette[trans_mask, 2, 3] = 255
        palette[trans_mask, 3, :3] = 0
        palette[trans_mask, 3, 3] = 0
    else:
        palette[:, 2, :3] = (2 * rgb0 + rgb1 + 1) // 3
        palette[:, 2, 3] = 255
        palette[:, 3, :3] = (rgb0 + 2 * rgb1 + 1) // 3
        palette[:, 3, 3] = 255

    shifts = np.arange(0, 32, 2, dtype=np.uint32)  # (16,)
    indices = ((bits[:, None] >> shifts[None, :]) & 0x03).astype(np.intp)  # (N, 16)
    block_idx = np.arange(n_blocks, dtype=np.intp)[:, None]
    out_pixels = palette[block_idx, indices]  # (N, 16, 4)
    return out_pixels.reshape(n_blocks, 4, 4, 4)


def decode_dxt1(buf: bytes, width: int, height: int) -> Image.Image:
    bw = max(1, (width + 3) // 4)
    bh = max(1, (height + 3) // 4)
    needed = bw * bh * 8
    raw = np.frombuffer(buf[:needed].ljust(needed, b"\x00"), dtype=np.uint8).reshape(bw * bh, 8)
    blocks = _decompress_dxt_color_blocks(raw, is_dxt1=True)  # (bw*bh, 4, 4, 4)
    img_arr = blocks.reshape(bh, bw, 4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 4)
    return Image.fromarray(img_arr[:height, :width, :], mode="RGBA")


def decode_dxt3(buf: bytes, width: int, height: int) -> Image.Image:
    bw = max(1, (width + 3) // 4)
    bh = max(1, (height + 3) // 4)
    needed = bw * bh * 16
    raw = np.frombuffer(buf[:needed].ljust(needed, b"\x00"), dtype=np.uint8).reshape(bw * bh, 16)
    alpha_raw = raw[:, :8]  # 8 bytes = 16 x 4-bit alpha values
    color_raw = raw[:, 8:16]

    blocks = _decompress_dxt_color_blocks(color_raw, is_dxt1=False)
    low_nibbles = (alpha_raw & 0x0F) * 17
    high_nibbles = ((alpha_raw >> 4) & 0x0F) * 17
    alpha_16 = np.empty((bw * bh, 16), dtype=np.uint8)
    alpha_16[:, 0::2] = low_nibbles
    alpha_16[:, 1::2] = high_nibbles
    blocks[:, :, :, 3] = alpha_16.reshape(bw * bh, 4, 4)

    img_arr = blocks.reshape(bh, bw, 4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 4)
    return Image.fromarray(img_arr[:height, :width, :], mode="RGBA")


def decode_dxt5(buf: bytes, width: int, height: int) -> Image.Image:
    bw = max(1, (width + 3) // 4)
    bh = max(1, (height + 3) // 4)
    n_blocks = bw * bh
    needed = n_blocks * 16
    raw = np.frombuffer(buf[:needed].ljust(needed, b"\x00"), dtype=np.uint8).reshape(n_blocks, 16)

    a0 = raw[:, 0].astype(np.int32)
    a1 = raw[:, 1].astype(np.int32)
    alpha_lut = np.zeros((n_blocks, 8), dtype=np.uint8)
    alpha_lut[:, 0] = a0
    alpha_lut[:, 1] = a1

    m7 = a0 > a1
    if np.any(m7):
        for k in range(1, 7):
            alpha_lut[m7, k + 1] = ((7 - k) * a0[m7] + k * a1[m7] + 3) // 7
    m5 = ~m7
    if np.any(m5):
        for k in range(1, 5):
            alpha_lut[m5, k + 1] = ((5 - k) * a0[m5] + k * a1[m5] + 2) // 5
        alpha_lut[m5, 6] = 0
        alpha_lut[m5, 7] = 255

    # Extract 48-bit alpha index table (bytes 2..8) into uint64
    a_bytes = np.zeros((n_blocks, 8), dtype=np.uint8)
    a_bytes[:, :6] = raw[:, 2:8]
    a_bits = a_bytes.view(np.uint64).reshape(n_blocks)
    shifts = np.arange(0, 48, 3, dtype=np.uint64)
    a_indices = ((a_bits[:, None] >> shifts[None, :]) & 0x07).astype(np.intp)
    block_idx = np.arange(n_blocks, dtype=np.intp)[:, None]
    alpha_pixels = alpha_lut[block_idx, a_indices].reshape(n_blocks, 4, 4)

    blocks = _decompress_dxt_color_blocks(raw[:, 8:16], is_dxt1=False)
    blocks[:, :, :, 3] = alpha_pixels

    img_arr = blocks.reshape(bh, bw, 4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 4)
    return Image.fromarray(img_arr[:height, :width, :], mode="RGBA")


def decode_raster_to_pil(entry: RwRasterEntry) -> Image.Image:
    """Decode the top mip level of an RwRasterEntry into a PIL RGBA Image."""
    raw = entry.raw_chunk
    w, h = max(1, entry.width), max(1, entry.height)
    rfmt = entry.raster_format
    has_pal8 = (rfmt & 0x2000) != 0
    has_pal4 = (rfmt & 0x4000) != 0

    if has_pal8:
        pal_raw = raw[112 : 112 + 1024]
        data_sz = struct.unpack_from("<I", raw, 112 + 1024)[0]
        idx_raw = raw[112 + 1024 + 4 : 112 + 1024 + 4 + data_sz]
        pal = np.frombuffer(pal_raw.ljust(1024, b"\x00"), dtype=np.uint8).reshape(256, 4)
        indices = np.frombuffer(idx_raw.ljust(w * h, b"\x00")[: w * h], dtype=np.uint8)
        rgba = pal[indices].reshape(h, w, 4)
        return Image.fromarray(rgba, mode="RGBA")

    if has_pal4:
        pal_raw = raw[112 : 112 + 64]
        data_sz = struct.unpack_from("<I", raw, 112 + 64)[0]
        idx_raw = raw[112 + 64 + 4 : 112 + 64 + 4 + data_sz]
        pal = np.frombuffer(pal_raw.ljust(64, b"\x00"), dtype=np.uint8).reshape(16, 4)
        indices = np.frombuffer(idx_raw.ljust(w * h, b"\x00")[: w * h], dtype=np.uint8) & 0x0F
        rgba = pal[indices].reshape(h, w, 4)
        return Image.fromarray(rgba, mode="RGBA")

    if len(raw) < 116:
        return Image.new("RGBA", (w, h), (128, 128, 128, 255))

    data_sz = struct.unpack_from("<I", raw, 112)[0]
    pix = raw[116 : 116 + data_sz]
    fmt = entry.format_label

    if fmt == "DXT1":
        return decode_dxt1(pix, w, h)
    if fmt == "DXT3":
        return decode_dxt3(pix, w, h)
    if fmt == "DXT5":
        return decode_dxt5(pix, w, h)

    if fmt in ("RGBA32", "RGB32") or entry.depth == 32:
        needed = w * h * 4
        img = Image.frombytes("RGBA", (w, h), pix.ljust(needed, b"\x00")[:needed], "raw", "BGRA")
        if fmt == "RGB32" or entry.d3d_format == 22:
            r, g, b, _ = img.split()
            img = Image.merge("RGBA", (r, g, b, Image.new("L", (w, h), 255)))
        return img

    if entry.depth == 24:
        needed = w * h * 3
        rgb = Image.frombytes("RGB", (w, h), pix.ljust(needed, b"\x00")[:needed], "raw", "BGR")
        return rgb.convert("RGBA")

    if entry.depth == 16:
        needed = w * h * 2
        u16 = np.frombuffer(pix.ljust(needed, b"\x00")[:needed], dtype=np.uint16).reshape(h, w)
        if fmt == "A4R4G4B4":
            a = ((u16 >> 12) & 0x0F) * 17
            r = ((u16 >> 8) & 0x0F) * 17
            g = ((u16 >> 4) & 0x0F) * 17
            b = (u16 & 0x0F) * 17
        elif fmt == "A1R5G5B5":
            a = ((u16 >> 15) & 0x01) * 255
            r = (((u16 >> 10) & 0x1F) * 255 + 15) // 31
            g = (((u16 >> 5) & 0x1F) * 255 + 15) // 31
            b = ((u16 & 0x1F) * 255 + 15) // 31
        else:
            a = np.full((h, w), 255, dtype=np.uint16)
            r = (((u16 >> 11) & 0x1F) * 255 + 15) // 31
            g = (((u16 >> 5) & 0x3F) * 255 + 31) // 63
            b = ((u16 & 0x1F) * 255 + 15) // 31
        rgba = np.stack([r, g, b, a], axis=-1).astype(np.uint8)
        return Image.fromarray(rgba, mode="RGBA")

    if fmt == "LUM8" or entry.depth == 8:
        needed = w * h
        l_img = Image.frombytes("L", (w, h), pix.ljust(needed, b"\x00")[:needed])
        return l_img.convert("RGBA")

    return Image.new("RGBA", (w, h), (128, 128, 128, 255))


# ---------------------------------------------------------------------------
# Fast Vectorized NumPy DXT1 / DXT5 Encoder + DirectXTex texconv Integration
# ---------------------------------------------------------------------------

def _rgb888_to_rgb565(rgb: np.ndarray) -> np.ndarray:
    """Convert (..., 3) uint8 RGB array to (...) uint16 RGB565."""
    r = (rgb[..., 0].astype(np.uint16) >> 3) & 0x1F
    g = (rgb[..., 1].astype(np.uint16) >> 2) & 0x3F
    b = (rgb[..., 2].astype(np.uint16) >> 3) & 0x1F
    return (r << 11) | (g << 5) | b


def _encode_dxt_color_blocks_numpy(rgba_arr: np.ndarray, force_opaque_mode: bool = True) -> np.ndarray:
    """
    Vectorized bounding-box endpoint DXT color block compressor.
    Input: (H, W, 4) uint8 where H, W are multiples of 4.
    Returns: (n_blocks, 8) uint8.
    """
    h, w, _ = rgba_arr.shape
    bh, bw = h // 4, w // 4
    n_blocks = bh * bw
    blocks = rgba_arr[:, :, :3].reshape(bh, 4, bw, 4, 3).transpose(0, 2, 1, 3, 4).reshape(n_blocks, 16, 3)

    # Inset bounding box by 1/16th to reduce outlier distortion
    c_min = blocks.min(axis=1).astype(np.int32)
    c_max = blocks.max(axis=1).astype(np.int32)
    inset = (c_max - c_min) >> 4
    c_min = np.clip(c_min + inset, 0, 255).astype(np.uint8)
    c_max = np.clip(c_max - inset, 0, 255).astype(np.uint8)

    c0_16 = _rgb888_to_rgb565(c_max)
    c1_16 = _rgb888_to_rgb565(c_min)

    # Ensure c0_16 > c1_16 for 4-color mode
    swap = c0_16 < c1_16
    c0_final = np.where(swap, c1_16, c0_16)
    c1_final = np.where(swap, c0_16, c1_16)

    equal = c0_final == c1_final
    if force_opaque_mode:
        # When equal, indices will all be 0, so c0 == c1 is safe as long as indices=0
        pass

    rgb0 = _decode_rgb565_array(c0_final).astype(np.int32)  # (N, 3)
    rgb1 = _decode_rgb565_array(c1_final).astype(np.int32)  # (N, 3)
    rgb2 = (2 * rgb0 + rgb1 + 1) // 3
    rgb3 = (rgb0 + 2 * rgb1 + 1) // 3

    pal = np.stack([rgb0, rgb1, rgb2, rgb3], axis=1)  # (N, 4, 3)
    diffs = blocks[:, :, None, :].astype(np.int32) - pal[:, None, :, :]  # (N, 16, 4, 3)
    dists = (diffs * diffs).sum(axis=-1)  # (N, 16, 4)
    best_idx = dists.argmin(axis=-1).astype(np.uint32)  # (N, 16)
    best_idx[equal, :] = 0

    shifts = np.arange(0, 32, 2, dtype=np.uint32)
    packed_bits = (best_idx << shifts[None, :]).sum(axis=1, dtype=np.uint32)

    out = np.empty((n_blocks, 8), dtype=np.uint8)
    out[:, 0:2] = c0_final[:, None].view(np.uint8)
    out[:, 2:4] = c1_final[:, None].view(np.uint8)
    out[:, 4:8] = packed_bits[:, None].view(np.uint8)
    return out


def encode_dxt1_numpy(pil_img: Image.Image) -> bytes:
    w = max(4, (pil_img.width + 3) & ~3)
    h = max(4, (pil_img.height + 3) & ~3)
    if (pil_img.width, pil_img.height) != (w, h):
        pil_img = pil_img.resize((w, h), Image.Resampling.LANCZOS)
    arr = np.asarray(pil_img.convert("RGBA"), dtype=np.uint8)
    return _encode_dxt_color_blocks_numpy(arr, force_opaque_mode=True).tobytes()


def encode_dxt5_numpy(pil_img: Image.Image) -> bytes:
    w = max(4, (pil_img.width + 3) & ~3)
    h = max(4, (pil_img.height + 3) & ~3)
    if (pil_img.width, pil_img.height) != (w, h):
        pil_img = pil_img.resize((w, h), Image.Resampling.LANCZOS)
    arr = np.asarray(pil_img.convert("RGBA"), dtype=np.uint8)
    bh, bw = h // 4, w // 4
    n_blocks = bh * bw

    color_blocks = _encode_dxt_color_blocks_numpy(arr, force_opaque_mode=False)
    alpha_blocks = arr[:, :, 3].reshape(bh, 4, bw, 4).transpose(0, 2, 1, 3).reshape(n_blocks, 16)

    a0 = alpha_blocks.max(axis=1).astype(np.int32)
    a1 = alpha_blocks.min(axis=1).astype(np.int32)
    same = a0 == a1
    a0_adj = np.where(same & (a0 < 255), a0 + 1, a0)
    a1_adj = np.where(same & (a0 == 255), 254, a1)

    lut = np.zeros((n_blocks, 8), dtype=np.int32)
    lut[:, 0] = a0_adj
    lut[:, 1] = a1_adj
    for k in range(1, 7):
        lut[:, k + 1] = ((7 - k) * a0_adj + k * a1_adj + 3) // 7

    a_diffs = np.abs(alpha_blocks[:, :, None].astype(np.int32) - lut[:, None, :])
    a_idx = a_diffs.argmin(axis=-1).astype(np.uint64)
    a_idx[same, :] = 0

    shifts = np.arange(0, 48, 3, dtype=np.uint64)
    packed_a = (a_idx << shifts[None, :]).sum(axis=1, dtype=np.uint64)
    packed_a_bytes = packed_a[:, None].view(np.uint8)[:, :6]

    out = np.empty((n_blocks, 16), dtype=np.uint8)
    out[:, 0] = a0_adj.astype(np.uint8)
    out[:, 1] = a1_adj.astype(np.uint8)
    out[:, 2:8] = packed_a_bytes
    out[:, 8:16] = color_blocks
    return out.tobytes()


def _make_rgba_dds_bytes(pil_img: Image.Image) -> bytes:
    rgba = pil_img.convert("RGBA")
    w, h = rgba.size
    header = bytearray(128)
    header[0:4] = b"DDS "
    struct.pack_into("<I", header, 4, 124)
    struct.pack_into("<I", header, 8, 0x100F)
    struct.pack_into("<I", header, 12, h)
    struct.pack_into("<I", header, 16, w)
    struct.pack_into("<I", header, 20, w * 4)
    struct.pack_into("<I", header, 28, 1)
    struct.pack_into("<I", header, 76, 32)
    struct.pack_into("<I", header, 80, 0x41)
    struct.pack_into("<I", header, 88, 32)
    struct.pack_into("<I", header, 92, 0x000000FF)
    struct.pack_into("<I", header, 96, 0x0000FF00)
    struct.pack_into("<I", header, 100, 0x00FF0000)
    struct.pack_into("<I", header, 104, 0xFF000000)
    struct.pack_into("<I", header, 108, 0x1000)
    return bytes(header) + rgba.tobytes("raw", "RGBA")


def _extract_dds_mip_levels(
    dds_bytes: bytes,
    width: int,
    height: int,
    mip_count: int,
    dxt_format: str,
) -> List[bytes]:
    if dds_bytes[:4] != b"DDS ":
        raise ValueError("Invalid DDS header from texconv")
    actual_mips = max(1, struct.unpack_from("<I", dds_bytes, 28)[0])
    count = min(mip_count, actual_mips)
    block_bytes = 8 if dxt_format == "DXT1" else 16
    levels: List[bytes] = []
    cursor = 128
    w, h = width, height
    for _ in range(count):
        n_bytes = max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * block_bytes
        if cursor + n_bytes > len(dds_bytes):
            break
        levels.append(dds_bytes[cursor : cursor + n_bytes])
        cursor += n_bytes
        w = max(1, w >> 1)
        h = max(1, h >> 1)
    return levels


def encode_pil_to_dxt_mip_chain(
    pil_img: Image.Image,
    dxt_format: str = "DXT1",
    generate_mipmaps: bool = True,
    texconv_path: Optional[Path] = None,
) -> List[bytes]:
    """
    Encode a PIL image into a list of DXT1 or DXT5 mipmap byte buffers.
    Uses Microsoft DirectXTex texconv.exe if present and image dimensions >= 4,
    with seamless fallback to the built-in NumPy vectorized encoder.
    """
    dxt_format = "DXT5" if dxt_format.upper() in ("DXT5", "DXT3") else "DXT1"
    w = max(4, (pil_img.width + 3) & ~3)
    h = max(4, (pil_img.height + 3) & ~3)
    if (pil_img.width, pil_img.height) != (w, h):
        pil_img = pil_img.resize((w, h), Image.Resampling.LANCZOS)

    # Calculate valid mip levels where both width and height remain >= 4
    max_mips = 1
    if generate_mipmaps:
        mw, mh = w, h
        while mw >= 8 and mh >= 8 and max_mips < 10:
            mw >>= 1
            mh >>= 1
            max_mips += 1

    tc = texconv_path or TEXCONV_DEFAULT_PATH
    if tc and tc.is_file():
        try:
            with tempfile.TemporaryDirectory(prefix="ewts_dxt_") as tmp_dir:
                tmp_p = Path(tmp_dir)
                in_dds = tmp_p / "input.dds"
                in_dds.write_bytes(_make_rgba_dds_bytes(pil_img))
                cmd = [
                    str(tc),
                    "-nologo",
                    "-dx9",
                    "-f",
                    dxt_format,
                    "-m",
                    str(max_mips),
                    "-o",
                    str(tmp_p),
                    "-y",
                    str(in_dds),
                ]
                res = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=20,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if res.returncode == 0 and in_dds.exists():
                    levels = _extract_dds_mip_levels(
                        in_dds.read_bytes(), w, h, max_mips, dxt_format
                    )
                    if levels:
                        return levels
        except Exception:
            pass

    # Built-in NumPy vectorized mipmap chain fallback
    levels: List[bytes] = []
    cur_img = pil_img.convert("RGBA")
    mw, mh = w, h
    for _ in range(max_mips):
        if (cur_img.width, cur_img.height) != (mw, mh):
            cur_img = cur_img.resize((mw, mh), Image.Resampling.LANCZOS)
        if dxt_format == "DXT5":
            levels.append(encode_dxt5_numpy(cur_img))
        else:
            levels.append(encode_dxt1_numpy(cur_img))
        mw = max(4, mw >> 1)
        mh = max(4, mh >> 1)
    return levels


def build_rw_texnative_chunk(
    name: str,
    pil_img: Image.Image,
    dxt_format: str = "DXT1",
    generate_mipmaps: bool = True,
    mask_name: str = "",
    filter_flags: int = 0x1102,
    version: int = RW_VERSION_SA,
) -> RwRasterEntry:
    """
    Build a complete RenderWare D3D9 0x15 TextureNative chunk from a PIL Image.
    """
    dxt_format = "DXT5" if dxt_format.upper() in ("DXT5", "DXT3") else "DXT1"
    w = max(4, (pil_img.width + 3) & ~3)
    h = max(4, (pil_img.height + 3) & ~3)
    if (pil_img.width, pil_img.height) != (w, h):
        pil_img = pil_img.resize((w, h), Image.Resampling.LANCZOS)

    levels = encode_pil_to_dxt_mip_chain(
        pil_img, dxt_format=dxt_format, generate_mipmaps=generate_mipmaps
    )
    mip_count = len(levels)
    has_alpha = dxt_format == "DXT5"
    if has_alpha and not mask_name:
        mask_name = (name + "m")[:31]

    clean_name = name[:31].encode("latin1", errors="ignore").ljust(32, b"\x00")
    clean_mask = mask_name[:31].encode("latin1", errors="ignore").ljust(32, b"\x00")

    rfmt = (0x0500 if has_alpha else 0x0200) | (0x8000 if mip_count > 1 else 0)
    fourcc = FOURCC_DXT5 if has_alpha else FOURCC_DXT1
    depth = 32 if has_alpha else 16
    comp_byte = 8 | (1 if has_alpha else 0)  # D3D9 compressed flag (8) + hasAlpha (1)

    if mip_count > 1 and (filter_flags & 0xFF) == 0x02:
        filter_flags = (filter_flags & ~0xFF) | 0x06  # Trilinear mipmap filtering

    header_88 = struct.pack(
        "<II32s32sIIHHBBBB",
        9,             # platform_id = 9 (D3D9)
        filter_flags,
        clean_name,
        clean_mask,
        rfmt,
        fourcc,
        w,
        h,
        depth,
        mip_count,
        4,             # raster_type = 4 (Texture)
        comp_byte,
    )

    mip_payload = b"".join(struct.pack("<I", len(lv)) + lv for lv in levels)
    struct_chunk = rw_pack_chunk(RW_STRUCT, header_88 + mip_payload, version)
    ext_chunk = rw_pack_chunk(RW_EXTENSION, b"", version)
    raw_chunk = rw_pack_chunk(RW_TEXNATIVE, struct_chunk + ext_chunk, version)

    return RwRasterEntry(
        index=0,
        name=name[:31],
        mask_name=mask_name[:31],
        platform_id=9,
        filter_flags=filter_flags,
        raster_format=rfmt,
        d3d_format=fourcc,
        width=w,
        height=h,
        depth=depth,
        mip_levels=mip_count,
        raster_type=4,
        compression=comp_byte,
        format_label=dxt_format,
        has_alpha=has_alpha,
        is_pot=_is_pot(w) and _is_pot(h),
        aspect_label=_aspect_str(w, h),
        top_level_bytes=len(levels[0]),
        total_chunk_bytes=len(raw_chunk),
        raw_chunk=raw_chunk,
        version=version,
    )


def replace_rasters_in_txd_bytes(
    txd_bytes: bytes,
    replacements: Dict[str, RwRasterEntry],
) -> bytes:
    """
    Replace one or more named rasters (case-insensitive) inside `txd_bytes`
    while keeping all other rasters and trailing extensions untouched.
    """
    root_ver, dev_id, entries, tail_ext = parse_txd_bytes(txd_bytes)
    rep_map = {k.lower(): v for k, v in replacements.items()}
    new_entries: List[RwRasterEntry] = []
    matched_keys = set()

    for entry in entries:
        key = entry.name.lower()
        if key in rep_map:
            rep = rep_map[key]
            # Preserve exact original casing of texture name from the TXD
            if rep.name != entry.name:
                # Patch name_32 at offset 24+8 = 32 inside raw_chunk
                patched_raw = bytearray(rep.raw_chunk)
                patched_raw[32:64] = entry.name[:31].encode("latin1", errors="ignore").ljust(32, b"\x00")
                rep.raw_chunk = bytes(patched_raw)
                rep.name = entry.name
            rep.index = entry.index
            new_entries.append(rep)
            matched_keys.add(key)
        else:
            new_entries.append(entry)

    return serialize_txd_bytes(new_entries, root_version=root_ver, device_id=2, tail_ext=tail_ext)
