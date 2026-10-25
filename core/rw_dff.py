"""
RenderWare .DFF 3D Geometry Parser, UV-Aspect & Sub-Region Analyzer,
and Static 3D Monument / Statue Baker for Eagle World Texture & Landmark Studio.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from core.rw_txd import (
    RW_EXTENSION,
    RW_STRUCT,
    RW_VERSION_SA,
    build_rw_texnative_chunk,
    decode_raster_to_pil,
    parse_txd_bytes,
    rw_pack_chunk,
    serialize_txd_bytes,
)

RW_STRING = 0x02
RW_TEXTURE = 0x06
RW_MATERIAL = 0x07
RW_MATLIST = 0x08
RW_FRAMELIST = 0x0E
RW_GEOMETRY = 0x0F
RW_CLUMP = 0x10
RW_ATOMIC = 0x14
RW_GEOMLIST = 0x1A
RW_BINMESH_PLG = 0x050E
RW_FRAME_NAME = 0x0253F2FE

# Geometry flags
RPGEOMETRYTRISTRIP = 0x00000001
RPGEOMETRYPOSITIONS = 0x00000002
RPGEOMETRYTEXTURED = 0x00000004
RPGEOMETRYPRELIT = 0x00000008
RPGEOMETRYNORMALS = 0x00000010
RPGEOMETRYLIGHT = 0x00000020
RPGEOMETRYMODULATEMATERIALCOLOR = 0x00000040
RPGEOMETRYTEXTURED2 = 0x00000080
RPGEOMETRYNATIVE = 0x01000000


def _dff_chunk_length(data: bytes, kind: int, size: int, body_offset: int) -> int:
    if kind in (RW_STRUCT, RW_CLUMP) and size > len(data) - body_offset and (size >> 24) == 1:
        return size & 0x00FFFFFF
    return size


def iter_rw_chunks(data: bytes, start: int = 0, end: Optional[int] = None):
    end = len(data) if end is None else min(len(data), end)
    pos = start
    while pos + 12 <= end:
        kind, size, ver = struct.unpack_from("<III", data, pos)
        body = pos + 12
        size = _dff_chunk_length(data, kind, size, body)
        stop = body + size
        if stop > end:
            break
        yield kind, pos, body, stop, ver
        pos = stop


def _read_rw_string(data: bytes, body: int, stop: int) -> str:
    return data[body:stop].split(b"\x00", 1)[0].decode("latin1", errors="ignore").strip()


def _parse_material_list(data: bytes, body: int, stop: int) -> List[Dict[str, Any]]:
    materials: List[Dict[str, Any]] = []
    for kind, _, mbody, mstop, _ in iter_rw_chunks(data, body, stop):
        if kind != RW_MATERIAL:
            continue
        mat_info: Dict[str, Any] = {
            "color": [255, 255, 255, 255],
            "texture": "",
            "mask": "",
        }
        for ckind, _, cbody, cstop, _ in iter_rw_chunks(data, mbody, mstop):
            if ckind == RW_STRUCT and (cstop - cbody) >= 16:
                r, g, b, a = struct.unpack_from("<4B", data, cbody + 4)
                mat_info["color"] = [r, g, b, a]
            elif ckind == RW_TEXTURE:
                str_chunks = [
                    _read_rw_string(data, sbody, sstop)
                    for skind, _, sbody, sstop, _ in iter_rw_chunks(data, cbody, cstop)
                    if skind == RW_STRING
                ]
                if len(str_chunks) >= 1:
                    mat_info["texture"] = str_chunks[0]
                if len(str_chunks) >= 2:
                    mat_info["mask"] = str_chunks[1]
        materials.append(mat_info)
    return materials


def _parse_single_geometry(data: bytes, body: int, stop: int) -> Optional[Dict[str, Any]]:
    children = list(iter_rw_chunks(data, body, stop))
    if not children or children[0][0] != RW_STRUCT:
        return None
    _, _, sbody, sstop, sver = children[0]
    if sstop - sbody < 16:
        return None

    flags, num_tris, num_verts, num_morphs = struct.unpack_from("<IIII", data, sbody)
    if (flags & RPGEOMETRYNATIVE) or num_verts == 0 or num_verts > 250000:
        return None

    # Check if RW < 3.4 surface properties (ambient, specular, diffuse floats) are in Struct
    rw_rel = ((sver >> 14) & 0x3FF00) | ((sver >> 16) & 0x3F) | 0x30000 if (sver & 0xFFFF0000) else (sver << 8)
    cursor = sbody + 16
    if rw_rel < 0x34000 and cursor + 12 <= sstop:
        cursor += 12

    num_uv_sets = (flags >> 16) & 0xFF
    if num_uv_sets == 0:
        if flags & RPGEOMETRYTEXTURED2:
            num_uv_sets = 2
        elif flags & RPGEOMETRYTEXTURED:
            num_uv_sets = 1

    if flags & RPGEOMETRYPRELIT:
        cursor += num_verts * 4

    uvs = np.zeros((num_verts, 2), dtype=np.float32)
    if num_uv_sets >= 1 and cursor + num_verts * 8 <= sstop:
        uvs = np.frombuffer(data, dtype=np.float32, count=num_verts * 2, offset=cursor).reshape(num_verts, 2).copy()
        cursor += num_uv_sets * num_verts * 8

    # Triangles in struct: each is 8 bytes: uint16 v2, v1, flags/mat, v3 (RenderWare triangle layout: v2, v1, matId, v3)
    tris = np.zeros((0, 4), dtype=np.int32)
    if num_tris > 0 and cursor + num_tris * 8 <= sstop:
        raw_tris = np.frombuffer(data, dtype=np.uint16, count=num_tris * 4, offset=cursor).reshape(num_tris, 4)
        # RenderWare RpTriangle is (vertIndex[0]=v2, vertIndex[1]=v1, matIndex=flags, vertIndex[2]=v3)
        tris = np.stack([raw_tris[:, 1], raw_tris[:, 0], raw_tris[:, 3], raw_tris[:, 2]], axis=-1).astype(np.int32)
        cursor += num_tris * 8

    # Morph target 0: bounding sphere (16 bytes: x, y, z, r), hasPosition (uint32), hasNormals (uint32)
    if cursor + 24 > sstop:
        return None
    has_pos, has_norm = struct.unpack_from("<II", data, cursor + 16)
    cursor += 24

    verts = np.zeros((num_verts, 3), dtype=np.float32)
    if has_pos and cursor + num_verts * 12 <= sstop:
        verts = np.frombuffer(data, dtype=np.float32, count=num_verts * 3, offset=cursor).reshape(num_verts, 3).copy()
        cursor += num_verts * 12

    normals = np.zeros((num_verts, 3), dtype=np.float32)
    if has_norm and cursor + num_verts * 12 <= sstop:
        normals = np.frombuffer(data, dtype=np.float32, count=num_verts * 3, offset=cursor).reshape(num_verts, 3).copy()
        cursor += num_verts * 12

    materials: List[Dict[str, Any]] = []
    for ckind, _, cbody, cstop, _ in children[1:]:
        if ckind == RW_MATLIST:
            materials = _parse_material_list(data, cbody, cstop)

    # Filter out degenerate / out-of-bounds triangles
    if len(tris) > 0:
        valid = (
            (tris[:, 0] < num_verts)
            & (tris[:, 1] < num_verts)
            & (tris[:, 2] < num_verts)
            & (tris[:, 0] != tris[:, 1])
            & (tris[:, 1] != tris[:, 2])
            & (tris[:, 0] != tris[:, 2])
        )
        tris = tris[valid]

    return {
        "vertices": verts,
        "normals": normals,
        "uvs": uvs,
        "triangles": tris,
        "materials": materials,
    }


def parse_dff_geometries(dff_bytes: bytes) -> List[Dict[str, Any]]:
    """Extract all geometries from a RenderWare .DFF file."""
    geoms: List[Dict[str, Any]] = []
    for kind, _, body, stop, _ in iter_rw_chunks(dff_bytes):
        if kind == RW_CLUMP:
            for ckind, _, cbody, cstop, _ in iter_rw_chunks(dff_bytes, body, stop):
                if ckind == RW_GEOMLIST:
                    for gkind, _, gbody, gstop, _ in iter_rw_chunks(dff_bytes, cbody, cstop):
                        if gkind == RW_GEOMETRY:
                            g = _parse_single_geometry(dff_bytes, gbody, gstop)
                            if g is not None:
                                geoms.append(g)
        elif kind == RW_GEOMETRY:
            g = _parse_single_geometry(dff_bytes, body, stop)
            if g is not None:
                geoms.append(g)
    return geoms


def _aspect_to_readable(ratio_wh: float) -> str:
    if ratio_wh <= 0 or not math.isfinite(ratio_wh):
        return "1:1"
    candidates = [
        (4.0, "4:1 (Dải ngang)"),
        (3.0, "3:1 (Ngang rộng)"),
        (2.0, "2:1 (Biển ngang chuẩn)"),
        (16.0 / 9.0, "16:9 (Màn hình ngang)"),
        (1.5, "3:2 (Ngang)"),
        (4.0 / 3.0, "4:3 (Ngang vừa)"),
        (1.0, "1:1 (Vuông)"),
        (0.75, "3:4 (Poster đứng)"),
        (0.625, "5:8 (Nhà chờ Bus đứng)"),
        (0.5, "1:2 (Banner đứng)"),
        (0.25, "1:4 (Phướn dọc hẹp)"),
    ]
    best_r, best_lbl = min(candidates, key=lambda c: abs(math.log(ratio_wh / c[0])))
    if abs(math.log(ratio_wh / best_r)) < 0.18:
        return best_lbl
    return f"{ratio_wh:.2f}:1"


def analyze_dff_materials_and_uvs(dff_bytes: bytes) -> Dict[str, Any]:
    """
    Analyze a .DFF binary to determine:
    1. Overall 3D bounding box (min, max, width_x, depth_y, height_z, vertex_count, triangle_count).
    2. Per-texture-name 3D surface dimensions (in meters), true physical display aspect ratio
       (compensating for UV stretching!), and UV crop sub-regions (for Atlas textures).
    """
    geoms = parse_dff_geometries(dff_bytes)
    if not geoms:
        return {
            "valid": False,
            "vertex_count": 0,
            "triangle_count": 0,
            "dimensions_m": [0.0, 0.0, 0.0],
            "bbox_min": [0.0, 0.0, 0.0],
            "bbox_max": [0.0, 0.0, 0.0],
            "textures": {},
        }

    all_verts = np.vstack([g["vertices"] for g in geoms if len(g["vertices"]) > 0])
    total_verts = int(all_verts.shape[0])
    total_tris = int(sum(len(g["triangles"]) for g in geoms))
    bmin = all_verts.min(axis=0)
    bmax = all_verts.max(axis=0)
    dims = bmax - bmin

    tex_stats: Dict[str, Dict[str, Any]] = {}

    for g in geoms:
        verts = g["vertices"]
        uvs = g["uvs"]
        tris = g["triangles"]
        mats = g["materials"]
        if len(tris) == 0 or not mats:
            continue

        for mat_idx, mat in enumerate(mats):
            tname = (mat.get("texture") or "").strip()
            if not tname:
                continue
            tkey = tname.lower()
            m_tris = tris[tris[:, 3] == mat_idx]
            if len(m_tris) == 0:
                continue

            v_idx = np.unique(m_tris[:, :3])
            m_verts = verts[v_idx]
            m_uvs = uvs[v_idx]

            mbmin = m_verts.min(axis=0)
            mbmax = m_verts.max(axis=0)
            mdims = mbmax - mbmin
            horiz_span_m = float(math.hypot(float(mdims[0]), float(mdims[1])))
            vert_span_m = float(mdims[2])

            umin, vmin = m_uvs.min(axis=0)
            umax, vmax = m_uvs.max(axis=0)
            du_total = max(0.01, float(umax - umin))
            dv_total = max(0.01, float(vmax - vmin))

            # Estimate physical meters per unit U and unit V from triangle edges
            p0 = verts[m_tris[:, 0]]
            p1 = verts[m_tris[:, 1]]
            p2 = verts[m_tris[:, 2]]
            uv0 = uvs[m_tris[:, 0]]
            uv1 = uvs[m_tris[:, 1]]
            uv2 = uvs[m_tris[:, 2]]

            # Compute triangle areas in 3D
            cross_vec = np.cross(p1 - p0, p2 - p0)
            tri_areas = 0.5 * np.linalg.norm(cross_vec, axis=1)
            total_area_m2 = float(tri_areas.sum())

            # Estimate effective physical aspect ratio (Width : Height) of the surface
            if vert_span_m > 0.05 and horiz_span_m > 0.05:
                surf_aspect = horiz_span_m / vert_span_m
            else:
                surf_aspect = 1.0

            # Full texture effective aspect ratio taking UV span into account:
            # If mesh crops only half the height (dv=0.5, du=1.0) on a 2:1 physical board,
            # the full texture aspect is (horiz_span / du) / (vert_span / dv).
            if du_total <= 1.25 and dv_total <= 1.25:
                tex_effective_aspect = (horiz_span_m / du_total) / max(0.05, vert_span_m / dv_total)
            else:
                tex_effective_aspect = surf_aspect

            tex_effective_aspect = float(np.clip(tex_effective_aspect, 0.15, 8.0))
            surf_aspect = float(np.clip(surf_aspect, 0.15, 8.0))

            # Normalized UV bounding box within [0, 1] for non-tiled sub-regions
            uv_crop = [
                round(float(np.clip(umin, 0.0, 1.0)), 4),
                round(float(np.clip(vmin, 0.0, 1.0)), 4),
                round(float(np.clip(umax, 0.0, 1.0)), 4),
                round(float(np.clip(vmax, 0.0, 1.0)), 4),
            ]
            is_atlas_subregion = bool(
                0.0 <= umin
                and umax <= 1.05
                and 0.0 <= vmin
                and vmax <= 1.05
                and (du_total < 0.82 or dv_total < 0.82)
            )
            is_tiled = bool((du_total > 1.35) or (dv_total > 1.35))

            rec = tex_stats.get(tkey)
            if rec is None or total_area_m2 > rec["area_m2"]:
                uv_boxes = rec["uv_boxes"] if rec else []
                if is_atlas_subregion and uv_crop not in uv_boxes and len(uv_boxes) < 12:
                    uv_boxes.append(uv_crop)
                tex_stats[tkey] = {
                    "texture_name": tname,
                    "triangle_count": int(len(m_tris)) + (rec["triangle_count"] if rec else 0),
                    "area_m2": round(total_area_m2 + (rec["area_m2"] if rec else 0.0), 2),
                    "phys_width_m": round(horiz_span_m, 2),
                    "phys_height_m": round(vert_span_m, 2),
                    "surface_aspect_wh": round(surf_aspect, 3),
                    "surface_aspect_label": _aspect_to_readable(surf_aspect),
                    "effective_tex_aspect_wh": round(tex_effective_aspect, 3),
                    "effective_tex_aspect_label": _aspect_to_readable(tex_effective_aspect),
                    "uv_span": [round(du_total, 3), round(dv_total, 3)],
                    "uv_crop": uv_crop,
                    "uv_boxes": uv_boxes,
                    "is_atlas_subregion": is_atlas_subregion,
                    "is_tiled": is_tiled,
                }
            elif is_atlas_subregion and uv_crop not in rec["uv_boxes"] and len(rec["uv_boxes"]) < 12:
                rec["uv_boxes"].append(uv_crop)

    return {
        "valid": True,
        "vertex_count": total_verts,
        "triangle_count": total_tris,
        "dimensions_m": [round(float(dims[0]), 2), round(float(dims[1]), 2), round(float(dims[2]), 2)],
        "bbox_min": [round(float(bmin[0]), 2), round(float(bmin[1]), 2), round(float(bmin[2]), 2)],
        "bbox_max": [round(float(bmax[0]), 2), round(float(bmax[1]), 2), round(float(bmax[2]), 2)],
        "textures": tex_stats,
    }


# ---------------------------------------------------------------------------
# 3D Monument / Statue Builder & Baker (Category 5 Pipeline)
# ---------------------------------------------------------------------------

STATUE_FINISH_PRESETS = {
    "anime_figure": {
        "label": "Anime Figure (Giữ màu gốc rực rỡ)",
        "tint_rgb": None,
        "contrast": 1.05,
        "brightness": 1.02,
    },
    "white_marble": {
        "label": "Tượng Đá Cẩm Thạch Trắng (Carrara Marble)",
        "tint_rgb": ((48, 52, 60), (242, 245, 250)),
        "contrast": 1.18,
        "brightness": 1.08,
    },
    "aged_bronze": {
        "label": "Tượng Đồng Cổ Điển (Aged Bronze & Patina)",
        "tint_rgb": ((28, 22, 16), (186, 148, 96)),
        "contrast": 1.22,
        "brightness": 0.98,
    },
    "golden_idol": {
        "label": "Tượng Vàng Hoàng Kim (24K Golden Idol)",
        "tint_rgb": ((55, 34, 8), (255, 218, 92)),
        "contrast": 1.25,
        "brightness": 1.08,
    },
    "jade_archon": {
        "label": "Tượng Ngọc Bích Thần Tượng (Archon Jade)",
        "tint_rgb": ((16, 48, 42), (168, 242, 218)),
        "contrast": 1.18,
        "brightness": 1.05,
    },
}


def apply_statue_finish_to_image(pil_img: Image.Image, finish: str = "anime_figure") -> Image.Image:
    preset = STATUE_FINISH_PRESETS.get(finish, STATUE_FINISH_PRESETS["anime_figure"])
    rgba = pil_img.convert("RGBA")
    r, g, b, a = rgba.split()
    rgb = Image.merge("RGB", (r, g, b))

    if preset["tint_rgb"] is not None:
        dark_col, light_col = preset["tint_rgb"]
        gray = ImageOps.grayscale(rgb)
        gray = ImageEnhance.Contrast(gray).enhance(preset["contrast"])
        rgb = ImageOps.colorize(gray, black=dark_col, white=light_col)
    else:
        if preset["contrast"] != 1.0:
            rgb = ImageEnhance.Contrast(rgb).enhance(preset["contrast"])
        if preset["brightness"] != 1.0:
            rgb = ImageEnhance.Brightness(rgb).enhance(preset["brightness"])

    nr, ng, nb = rgb.split()
    return Image.merge("RGBA", (nr, ng, nb, a))


def _auto_orient_statue_vertices(verts: np.ndarray, normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensure the character/statue mesh stands upright along the Z axis (GTA SA world up).
    Many ped/character DFFs have Z as up already, or Y as up if imported from OBJ/FBX.
    """
    if len(verts) == 0:
        return verts, normals
    spans = verts.max(axis=0) - verts.min(axis=0)
    # If Y span is strictly the tallest axis (> 1.35x Z span), rotate +90 deg around X so Y -> Z
    if spans[1] > spans[2] * 1.35 and spans[1] > spans[0] * 1.35:
        rot_v = np.stack([verts[:, 0], -verts[:, 2], verts[:, 1]], axis=-1)
        rot_n = (
            np.stack([normals[:, 0], -normals[:, 2], normals[:, 1]], axis=-1)
            if len(normals) == len(verts)
            else normals
        )
        return rot_v, rot_n
    return verts, normals


def _create_pedestal_mesh(
    center_x: float,
    center_y: float,
    base_z: float,
    radius_xy: float,
    height_z: float,
    texture_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a two-tier octagonal stone pedestal under the statue's feet.
    Returns (verts, normals, uvs, tris_with_mat0).
    """
    verts_list = []
    norms_list = []
    uvs_list = []
    tris_list = []

    tiers = [
        (base_z, base_z + height_z * 0.35, radius_xy * 1.18),
        (base_z + height_z * 0.35, base_z + height_z, radius_xy * 0.92),
    ]
    sides = 8
    for z0, z1, rad in tiers:
        base_idx = len(verts_list)
        for i in range(sides):
            ang0 = (2.0 * math.pi * i) / sides
            ang1 = (2.0 * math.pi * (i + 1)) / sides
            x0, y0 = center_x + rad * math.cos(ang0), center_y + rad * math.sin(ang0)
            x1, y1 = center_x + rad * math.cos(ang1), center_y + rad * math.sin(ang1)
            nx, ny = math.cos((ang0 + ang1) * 0.5), math.sin((ang0 + ang1) * 0.5)

            v_start = len(verts_list)
            verts_list.extend([[x0, y0, z0], [x1, y1, z0], [x1, y1, z1], [x0, y0, z1]])
            norms_list.extend([[nx, ny, 0.0]] * 4)
            uvs_list.extend([[0.0, 0.8], [0.5, 0.8], [0.5, 0.2], [0.0, 0.2]])
            tris_list.append([v_start, v_start + 1, v_start + 2, 0])
            tris_list.append([v_start, v_start + 2, v_start + 3, 0])

        # Top cap of tier
        top_center_idx = len(verts_list)
        verts_list.append([center_x, center_y, z1])
        norms_list.append([0.0, 0.0, 1.0])
        uvs_list.append([0.5, 0.5])
        for i in range(sides):
            ang = (2.0 * math.pi * i) / sides
            verts_list.append([center_x + rad * math.cos(ang), center_y + rad * math.sin(ang), z1])
            norms_list.append([0.0, 0.0, 1.0])
            uvs_list.append([0.5 + 0.3 * math.cos(ang), 0.5 + 0.3 * math.sin(ang)])
        for i in range(sides):
            v1 = top_center_idx + 1 + i
            v2 = top_center_idx + 1 + ((i + 1) % sides)
            tris_list.append([top_center_idx, v1, v2, 0])

    return (
        np.array(verts_list, dtype=np.float32),
        np.array(norms_list, dtype=np.float32),
        np.array(uvs_list, dtype=np.float32),
        np.array(tris_list, dtype=np.int32),
    )


def _pack_static_building_dff(
    verts: np.ndarray,
    normals: np.ndarray,
    uvs: np.ndarray,
    prelit_rgba: np.ndarray,
    tris: np.ndarray,
    materials: List[Dict[str, Any]],
    frame_name: str = "wibu_monument",
) -> bytes:
    """
    Pack a pure static RenderWare 3.6.0.3 (0x1803FFFF) Building .DFF
    with PRELIT vertex colors, Normals, UVs, BinMeshPLG, and zero Skin/HAnim bones.
    """
    num_verts = int(len(verts))
    num_tris = int(len(tris))

    # 1. FrameList (1 root frame with identity rotation & zero translation)
    frame_struct = struct.pack(
        "<I12fII",
        1,  # 1 frame
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 0.0,
        0xFFFFFFFF,  # parent = -1
        0,
    )
    fname_bytes = frame_name[:23].encode("latin1", errors="ignore")
    frame_ext = rw_pack_chunk(RW_EXTENSION, rw_pack_chunk(RW_FRAME_NAME, fname_bytes))
    framelist_chunk = rw_pack_chunk(RW_FRAMELIST, rw_pack_chunk(RW_STRUCT, frame_struct) + frame_ext)

    # 2. Geometry Struct
    flags = (
        RPGEOMETRYPOSITIONS
        | RPGEOMETRYTEXTURED
        | RPGEOMETRYPRELIT
        | RPGEOMETRYNORMALS
        | RPGEOMETRYLIGHT
        | RPGEOMETRYMODULATEMATERIALCOLOR
        | (1 << 16)  # 1 UV set
    )
    geom_header = struct.pack("<IIII", flags, num_tris, num_verts, 1)
    prelit_bytes = prelit_rgba.astype(np.uint8).tobytes()
    uv_bytes = uvs.astype(np.float32).tobytes()

    # RenderWare RpTriangle layout: uint16 v2, v1, matId, v3
    rw_tris = np.stack([tris[:, 1], tris[:, 0], tris[:, 3], tris[:, 2]], axis=-1).astype(np.uint16)
    tri_bytes = rw_tris.tobytes()

    # Bounding sphere
    bmin = verts.min(axis=0)
    bmax = verts.max(axis=0)
    center = (bmin + bmax) * 0.5
    radius = float(np.linalg.norm(verts - center, axis=1).max()) if num_verts > 0 else 1.0
    morph_header = struct.pack("<4fII", float(center[0]), float(center[1]), float(center[2]), radius, 1, 1)
    vert_bytes = verts.astype(np.float32).tobytes()
    norm_bytes = normals.astype(np.float32).tobytes()

    geom_struct_chunk = rw_pack_chunk(
        RW_STRUCT,
        geom_header + prelit_bytes + uv_bytes + tri_bytes + morph_header + vert_bytes + norm_bytes,
    )

    # 3. MaterialList
    mat_indices = struct.pack("<I" + "i" * len(materials), len(materials), *([-1] * len(materials)))
    mat_chunks = [rw_pack_chunk(RW_STRUCT, mat_indices)]
    for mat in materials:
        r, g, b, a = mat.get("color", [255, 255, 255, 255])
        tname = (mat.get("texture") or "").encode("latin1", errors="ignore") + b"\x00"
        if len(tname) % 4 != 0:
            tname += b"\x00" * (4 - (len(tname) % 4))
        mname = (mat.get("mask") or "").encode("latin1", errors="ignore") + b"\x00"
        if len(mname) % 4 != 0:
            mname += b"\x00" * (4 - (len(mname) % 4))

        tex_chunk = rw_pack_chunk(
            RW_TEXTURE,
            rw_pack_chunk(RW_STRUCT, struct.pack("<HH", 0x1106, 0))
            + rw_pack_chunk(RW_STRING, tname)
            + rw_pack_chunk(RW_STRING, mname)
            + rw_pack_chunk(RW_EXTENSION, b""),
        )
        mat_struct = struct.pack("<I4BIf3f", 0, r, g, b, a, 1, 1.0, 0.0, 1.0, 0.0)[:28]
        mat_chunks.append(
            rw_pack_chunk(
                RW_MATERIAL,
                rw_pack_chunk(RW_STRUCT, mat_struct)
                + tex_chunk
                + rw_pack_chunk(RW_EXTENSION, b""),
            )
        )
    matlist_chunk = rw_pack_chunk(RW_MATLIST, b"".join(mat_chunks))

    # 4. BinMesh PLG (0x50E) inside Geometry Extension (required by GTA SA D3D9 pipeline!)
    splits = []
    total_indices = 0
    for mat_idx in range(len(materials)):
        m_tris = tris[tris[:, 3] == mat_idx]
        if len(m_tris) == 0:
            continue
        idx_flat = m_tris[:, :3].reshape(-1).astype(np.uint32)
        total_indices += len(idx_flat)
        splits.append(struct.pack("<II", len(idx_flat), mat_idx) + idx_flat.tobytes())

    binmesh_payload = struct.pack("<III", 0, len(splits), total_indices) + b"".join(splits)
    geom_ext_chunk = rw_pack_chunk(RW_EXTENSION, rw_pack_chunk(RW_BINMESH_PLG, binmesh_payload))
    geom_chunk = rw_pack_chunk(RW_GEOMETRY, geom_struct_chunk + matlist_chunk + geom_ext_chunk)
    geomlist_chunk = rw_pack_chunk(RW_GEOMLIST, rw_pack_chunk(RW_STRUCT, struct.pack("<I", 1)) + geom_chunk)

    # 5. Atomic (links frame 0 and geometry 0)
    atomic_struct = struct.pack("<IIII", 0, 0, 5, 0)
    atomic_chunk = rw_pack_chunk(
        RW_ATOMIC,
        rw_pack_chunk(RW_STRUCT, atomic_struct) + rw_pack_chunk(RW_EXTENSION, b""),
    )

    # 6. Clump
    clump_struct = rw_pack_chunk(RW_STRUCT, struct.pack("<III", 1, 0, 0))
    clump_ext = rw_pack_chunk(RW_EXTENSION, b"")
    return rw_pack_chunk(
        RW_CLUMP,
        clump_struct + framelist_chunk + geomlist_chunk + atomic_chunk + clump_ext,
    )


def build_statue_dff_and_txd(
    target_monument_dff_bytes: bytes,
    source_dff_bytes: bytes,
    source_txd_bytes: bytes,
    target_txd_bytes: Optional[bytes] = None,
    height_scale: float = 1.0,
    rotation_deg: float = 0.0,
    z_offset_m: float = 0.0,
    add_pedestal: bool = True,
    statue_finish: str = "anime_figure",
    frame_name: str = "wibu_monument",
) -> Tuple[bytes, bytes, Dict[str, Any]]:
    """
    Convert any source 3D model (.DFF + .TXD, e.g. from Skin Library or custom upload)
    into a static, prelit GTA SA Monument .DFF + .TXD fitted to the target monument's bounding box!
    """
    target_info = analyze_dff_materials_and_uvs(target_monument_dff_bytes)
    src_geoms = parse_dff_geometries(source_dff_bytes)
    if not src_geoms:
        raise ValueError("Source DFF has no valid 3D geometry")

    # Merge all geometries from source DFF
    merged_verts = []
    merged_norms = []
    merged_uvs = []
    merged_tris = []
    merged_mats: List[Dict[str, Any]] = []
    v_offset = 0
    m_offset = 0

    for g in src_geoms:
        v = g["vertices"]
        n = g["normals"]
        u = g["uvs"]
        t = g["triangles"].copy()
        mats = g["materials"]
        if len(v) == 0 or len(t) == 0:
            continue
        if len(n) != len(v) or np.all(n == 0):
            n = np.zeros_like(v)
            n[:, 2] = 1.0
        t[:, :3] += v_offset
        t[:, 3] += m_offset
        merged_verts.append(v)
        merged_norms.append(n)
        merged_uvs.append(u)
        merged_tris.append(t)
        merged_mats.extend(mats if mats else [{"color": [255, 255, 255, 255], "texture": "", "mask": ""}])
        v_offset += len(v)
        m_offset += max(1, len(mats))

    verts = np.vstack(merged_verts)
    normals = np.vstack(merged_norms)
    uvs = np.vstack(merged_uvs)
    tris = np.vstack(merged_tris)

    verts, normals = _auto_orient_statue_vertices(verts, normals)

    # Rotate around Z axis if requested
    if abs(rotation_deg) > 1e-3:
        rad = math.radians(rotation_deg)
        ca, sa = math.cos(rad), math.sin(rad)
        rx = verts[:, 0] * ca - verts[:, 1] * sa
        ry = verts[:, 0] * sa + verts[:, 1] * ca
        verts = np.stack([rx, ry, verts[:, 2]], axis=-1)
        nx = normals[:, 0] * ca - normals[:, 1] * sa
        ny = normals[:, 0] * sa + normals[:, 1] * ca
        normals = np.stack([nx, ny, normals[:, 2]], axis=-1)

    # Determine target bounding box
    if target_info["valid"] and target_info["dimensions_m"][2] > 0.2:
        t_min = np.array(target_info["bbox_min"], dtype=np.float32)
        t_max = np.array(target_info["bbox_max"], dtype=np.float32)
        target_height = float(t_max[2] - t_min[2]) * max(0.1, float(height_scale))
        target_cx = float((t_min[0] + t_max[0]) * 0.5)
        target_cy = float((t_min[1] + t_max[1]) * 0.5)
        target_base_z = float(t_min[2]) + float(z_offset_m)
    else:
        target_height = 6.0 * max(0.1, float(height_scale))
        target_cx, target_cy, target_base_z = 0.0, 0.0, float(z_offset_m)

    s_min = verts.min(axis=0)
    s_max = verts.max(axis=0)
    src_height = max(0.05, float(s_max[2] - s_min[2]))
    src_cx = float((s_min[0] + s_max[0]) * 0.5)
    src_cy = float((s_min[1] + s_max[1]) * 0.5)

    ped_h = target_height * 0.14 if add_pedestal else 0.0
    figure_h = target_height - ped_h
    scale = figure_h / src_height

    verts[:, 0] = (verts[:, 0] - src_cx) * scale + target_cx
    verts[:, 1] = (verts[:, 1] - src_cy) * scale + target_cy
    verts[:, 2] = (verts[:, 2] - float(s_min[2])) * scale + target_base_z + ped_h

    if add_pedestal and ped_h > 0.05:
        fig_span_xy = max(
            float(verts[:, 0].max() - verts[:, 0].min()),
            float(verts[:, 1].max() - verts[:, 1].min()),
        )
        ped_rad = max(0.45, fig_span_xy * 0.55)
        first_tex = merged_mats[0].get("texture", "") if merged_mats else ""
        pv, pn, pu, pt = _create_pedestal_mesh(
            target_cx, target_cy, target_base_z, ped_rad, ped_h, first_tex
        )
        pt[:, :3] += len(verts)
        verts = np.vstack([verts, pv])
        normals = np.vstack([normals, pn])
        uvs = np.vstack([uvs, pu])
        tris = np.vstack([tris, pt])

    # Normalize normals and bake smooth Day/Night Prelight Vertex Colors
    n_len = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.where(n_len > 1e-6, normals / np.maximum(n_len, 1e-6), np.array([0.0, 0.0, 1.0], dtype=np.float32))
    sun_dir = np.array([0.45, -0.35, 0.82], dtype=np.float32)
    sun_dot = np.clip((normals * sun_dir).sum(axis=1), -1.0, 1.0)
    shade = np.clip(188.0 + 58.0 * sun_dot, 135.0, 250.0).astype(np.uint8)
    prelit_rgba = np.stack([shade, shade, shade, np.full_like(shade, 255)], axis=-1)

    new_dff_bytes = _pack_static_building_dff(
        verts=verts,
        normals=normals,
        uvs=uvs,
        prelit_rgba=prelit_rgba,
        tris=tris,
        materials=merged_mats,
        frame_name=frame_name,
    )

    # Build the accompanying .TXD:
    # Start from target_txd_bytes (if any, so other models sharing the monument TXD keep their textures!)
    # and add/update the statue's textures encoded as POT DXT1/DXT5 with mipmaps and statue_finish applied.
    _, _, src_entries, _ = parse_txd_bytes(source_txd_bytes)
    new_raster_map: Dict[str, Any] = {}
    for entry in src_entries:
        pil_img = decode_raster_to_pil(entry)
        pil_img = apply_statue_finish_to_image(pil_img, finish=statue_finish)
        # Ensure POT <= 512 for optimal streaming
        tw = min(512, max(64, 1 << int(round(math.log2(max(4, pil_img.width))))))
        th = min(512, max(64, 1 << int(round(math.log2(max(4, pil_img.height))))))
        if (pil_img.width, pil_img.height) != (tw, th):
            pil_img = pil_img.resize((tw, th), Image.Resampling.LANCZOS)
        dxt_fmt = "DXT5" if entry.has_alpha else "DXT1"
        new_chunk = build_rw_texnative_chunk(
            name=entry.name,
            pil_img=pil_img,
            dxt_format=dxt_fmt,
            generate_mipmaps=True,
        )
        new_raster_map[entry.name.lower()] = new_chunk

    if target_txd_bytes:
        try:
            root_ver, dev_id, tgt_entries, tail_ext = parse_txd_bytes(target_txd_bytes)
            combined_entries = []
            seen = set()
            for te in tgt_entries:
                kl = te.name.lower()
                if kl in new_raster_map:
                    combined_entries.append(new_raster_map[kl])
                    seen.add(kl)
                else:
                    combined_entries.append(te)
            for kl, ne in new_raster_map.items():
                if kl not in seen:
                    combined_entries.append(ne)
            new_txd_bytes = serialize_txd_bytes(combined_entries, root_version=root_ver, device_id=2, tail_ext=tail_ext)
        except Exception:
            new_txd_bytes = serialize_txd_bytes(list(new_raster_map.values()))
    else:
        new_txd_bytes = serialize_txd_bytes(list(new_raster_map.values()))

    final_min = verts.min(axis=0)
    final_max = verts.max(axis=0)
    final_dims = final_max - final_min

    stats = {
        "vertex_count": int(len(verts)),
        "triangle_count": int(len(tris)),
        "material_count": len(merged_mats),
        "texture_count": len(new_raster_map),
        "dimensions_m": [round(float(final_dims[0]), 2), round(float(final_dims[1]), 2), round(float(final_dims[2]), 2)],
        "dff_size_kb": round(len(new_dff_bytes) / 1024.0, 2),
        "txd_size_kb": round(len(new_txd_bytes) / 1024.0, 2),
        "finish": statue_finish,
        "add_pedestal": add_pedestal,
    }
    return new_dff_bytes, new_txd_bytes, stats
