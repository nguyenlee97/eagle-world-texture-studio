"""
Comprehensive Automated Test Suite for Eagle World Texture & Landmark Studio.
Tests:
1. RenderWare .TXD parsing, DXT1/DXT3/DXT5 decoding, and DXT1/DXT5 + Mipmap encoding.
2. RenderWare .DFF 3D geometry & UV-Aspect analysis on real `gameplay-lab` models.
3. All 4 Specialized 2D Surface Converters (CAT1..CAT4) including Dual-Sync LOD & Sub-Region Atlas.
4. 3D Monument / Statue Swapper & Baker (CAT5) converting a real Skin Library model into a static prelit monument.
5. Atomic VER2 .IMG patching, `game-write.lock` protocol, and 100% bit-for-bit SHA-256 Rollback verification.
6. FastAPI endpoints end-to-end verification.
"""

from __future__ import annotations

import hashlib
import struct
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

STUDIO_ROOT = Path(__file__).resolve().parent.parent
if str(STUDIO_ROOT) not in sys.path:
    sys.path.insert(0, str(STUDIO_ROOT))

from app import app, store
from core.catalog_store import generate_sample_wibu_artwork
from core.converters import convert_surface_raster
from core.deployer import (
    GameWriteLock,
    patch_img_archive_members,
    rollback_img_archive_members,
    sha256_bytes,
)
from core.rw_dff import (
    analyze_dff_materials_and_uvs,
    build_statue_dff_and_txd,
    parse_dff_geometries,
)
from core.rw_txd import (
    build_rw_texnative_chunk,
    decode_raster_to_pil,
    parse_txd_bytes,
    replace_rasters_in_txd_bytes,
)


def test_txd_parse_decode_and_encode_roundtrip():
    """Verify reading a real billboard TXD, decoding rasters, and encoding DXT1/DXT5 with mipmaps."""
    txd_bytes, _, _ = store.read_txd_bytes("billbrd01_lan", prefer_staged=False)
    root_ver, dev_id, entries, _ = parse_txd_bytes(txd_bytes)
    assert len(entries) >= 4

    first = entries[0]
    decoded = decode_raster_to_pil(first)
    assert decoded.size == (first.width, first.height)
    assert decoded.mode == "RGBA"

    # Encode a 512x256 DXT1 with mipmaps and verify size reduction vs GROUP_B1's 1024x512 DXT3
    sample_img = generate_sample_wibu_artwork("cyber_anime", 512, 256)
    new_chunk = build_rw_texnative_chunk(
        name=first.name,
        pil_img=sample_img,
        dxt_format="DXT1",
        generate_mipmaps=True,
    )
    assert new_chunk.width == 512
    assert new_chunk.height == 256
    assert new_chunk.format_label == "DXT1"
    assert new_chunk.mip_levels > 1
    assert new_chunk.total_chunk_bytes < 100 * 1024  # ~87 KB vs 569 KB in GROUP_B1

    # Replace inside TXD and verify re-parsing
    updated_txd = replace_rasters_in_txd_bytes(txd_bytes, {first.name: new_chunk})
    _, _, new_entries, _ = parse_txd_bytes(updated_txd)
    assert len(new_entries) == len(entries)
    re_decoded = decode_raster_to_pil(new_entries[0])
    assert re_decoded.size == (512, 256)


def test_dff_uv_and_aspect_analyzer():
    """Verify 3D DFF material & UV aspect ratio detection on real gameplay-lab models."""
    dff_bytes, _, _ = store.read_dff_bytes("bustopm", prefer_staged=False)
    info = analyze_dff_materials_and_uvs(dff_bytes)
    assert info["valid"] is True
    assert info["vertex_count"] > 10
    assert info["triangle_count"] > 10
    assert len(info["textures"]) >= 1


def test_all_four_specialized_surface_converters():
    """Verify CAT1 (Billboard + LOD), CAT2 (Mural/Graffiti), CAT3 (Neon/Sub-Region), CAT4 (Poster/Bus Stop)."""
    txd_bytes, _, _ = store.read_txd_bytes("billbrd01_lan", prefer_staged=False)
    _, _, entries, _ = parse_txd_bytes(txd_bytes)
    orig_entry = entries[0]
    orig_pil = decode_raster_to_pil(orig_entry)
    user_img = generate_sample_wibu_artwork("genshin_gold", 800, 400)

    # Create a mock LOD entry (64x32) to test Dual-Sync
    lod_orig = build_rw_texnative_chunk("lod_" + orig_entry.name[:16], orig_pil.resize((64, 32)), "DXT1", True)

    # 1. CAT1_BILLBOARDS with Dual-Sync LOD
    res_cat1 = convert_surface_raster(
        category="CAT1_BILLBOARDS",
        user_img=user_img,
        orig_entry=orig_entry,
        orig_decoded_img=orig_pil,
        dff_tex_info={"effective_tex_aspect_wh": 2.0, "effective_tex_aspect_label": "2:1 (Biển ngang chuẩn)"},
        options={"resolution_tier": "recommended", "auto_sync_lod": True, "print_weathering": 0.1},
        lod_orig_entry=lod_orig,
    )
    assert res_cat1["near_entry"].format_label == "DXT1"
    assert res_cat1["lod_entry"] is not None
    assert res_cat1["lod_entry"].width == 64
    assert res_cat1["lod_entry"].height == 32
    assert res_cat1["report"]["lod_synced"] is True

    # 2. CAT2_MURALS_BANNERS_GRAFFITI (Graffiti Alpha DXT5 & Mural Wall-Blend)
    res_cat2_graff = convert_surface_raster(
        category="CAT2_MURALS_BANNERS_GRAFFITI",
        user_img=generate_sample_wibu_artwork("graffiti_tag", 256, 256),
        orig_entry=orig_entry,
        orig_decoded_img=orig_pil,
        options={"sub_mode": "graffiti_alpha", "resolution_tier": "compact"},
    )
    assert res_cat2_graff["near_entry"].format_label == "DXT5"

    res_cat2_mural = convert_surface_raster(
        category="CAT2_MURALS_BANNERS_GRAFFITI",
        user_img=user_img,
        orig_entry=orig_entry,
        orig_decoded_img=orig_pil,
        options={"sub_mode": "mural_wall_blend", "wall_blend_strength": 0.22},
    )
    assert res_cat2_mural["near_entry"].format_label == "DXT1"

    # 3. CAT3_NEON_LED_SCREENS_CASINOS with Sub-Region Atlas Patching
    res_cat3 = convert_surface_raster(
        category="CAT3_NEON_LED_SCREENS_CASINOS",
        user_img=user_img,
        orig_entry=orig_entry,
        orig_decoded_img=orig_pil,
        options={
            "use_sub_region": True,
            "sub_region_rect": [0.0, 0.0, 0.5, 0.5],
            "led_scanline_strength": 0.15,
        },
    )
    assert "Atlas" in res_cat3["report"]["uv_compensation_note"]

    # 4. CAT4_POSTERS_SHOPS_TRANSIT_VENDING with Outer Frame Border Preservation
    res_cat4 = convert_surface_raster(
        category="CAT4_POSTERS_SHOPS_TRANSIT_VENDING",
        user_img=user_img,
        orig_entry=orig_entry,
        orig_decoded_img=orig_pil,
        dff_tex_info={"effective_tex_aspect_wh": 0.625, "effective_tex_aspect_label": "5:8 (Nhà chờ Bus đứng)"},
        options={"preserve_border_pct": 0.08, "glass_sheen_strength": 0.12},
    )
    assert "khung viền" in res_cat4["report"]["uv_compensation_note"]


def test_3d_monument_statue_baker():
    """Verify converting a real character from Skin Library into a static prelit Monument .DFF + .TXD."""
    cat = store.load_catalog()
    monuments = cat["monuments"]
    assert len(monuments) > 0
    mon = monuments[0]

    target_dff, _, _ = store.read_dff_bytes(mon["dff_name"], dff_img=mon["dff_img"], prefer_staged=False)
    target_txd, _, _ = store.read_txd_bytes(mon["txd_name"], prefer_staged=False)

    skins = store.list_skin_library_characters(limit=5)
    assert len(skins) > 0
    skin = skins[0]

    src_dff = Path(skin["dff_path"]).read_bytes()
    src_txd = Path(skin["txd_path"]).read_bytes()

    baked_dff, baked_txd, stats = build_statue_dff_and_txd(
        target_monument_dff_bytes=target_dff,
        source_dff_bytes=src_dff,
        source_txd_bytes=src_txd,
        target_txd_bytes=target_txd,
        height_scale=1.0,
        rotation_deg=45.0,
        z_offset_m=0.0,
        add_pedestal=True,
        statue_finish="white_marble",
        frame_name=mon["dff_name"],
    )

    assert len(baked_dff) > 500
    assert len(baked_txd) > 500
    assert stats["vertex_count"] > 100
    assert stats["triangle_count"] > 100

    # Verify the baked monument DFF parses cleanly as a static geometry
    geoms = parse_dff_geometries(baked_dff)
    assert len(geoms) == 1
    assert len(geoms[0]["vertices"]) == stats["vertex_count"]


def test_ver2_img_patch_and_bit_for_bit_rollback(tmp_path: Path):
    """
    Create a synthetic VER2 .IMG archive with 2 members, patch both (one smaller in-place,
    one larger appended), and verify that rollback restores the EXACT bit-for-bit SHA-256!
    """
    game_dir = tmp_path / "mock_game"
    custom_dir = game_dir / "CUSTOM"
    custom_dir.mkdir(parents=True)
    img_path = custom_dir / "TEST_WORLD.IMG"

    member1_orig = (b"ORIG_MEMBER_ONE_" * 200).ljust(4096, b"\x00")  # 2 sectors
    member2_orig = (b"ORIG_MEMBER_TWO_" * 100).ljust(2048, b"\x00")  # 1 sector

    # Header + 2 dir entries fits in sector 0 (2048 bytes)
    # Member 1 at sector 1 (len 2), Member 2 at sector 3 (len 1)
    hdr = b"VER2" + struct.pack("<I", 2)
    e1 = struct.pack("<IHH24s", 1, 2, 0, b"first.txd".ljust(24, b"\x00"))
    e2 = struct.pack("<IHH24s", 3, 1, 0, b"second.txd".ljust(24, b"\x00"))
    sector0 = (hdr + e1 + e2).ljust(2048, b"\x00")
    img_path.write_bytes(sector0 + member1_orig + member2_orig)

    orig_img_sha256 = hashlib.sha256(img_path.read_bytes()).hexdigest()

    backup_dir = tmp_path / "backup_tx"
    backup_dir.mkdir()
    lock_dir = tmp_path / "coordination" / "game-write.lock"

    # Patch first.txd with a SMALLER payload (1 sector <= 2 sectors -> in-place)
    # Patch second.txd with a LARGER payload (3 sectors > 1 sector -> appended at end)
    new_m1 = b"NEW_SMALL_M1" * 50
    new_m2 = b"NEW_LARGE_M2_" * 400

    with GameWriteLock(lock_dir=lock_dir, task_label="pytest-img-patch", resources=["CUSTOM/TEST_WORLD.IMG"]):
        assert lock_dir.exists()
        assert (lock_dir / "owner.json").exists()
        records = patch_img_archive_members(
            img_path=img_path,
            member_payloads={"first.txd": new_m1, "second.txd": new_m2},
            backup_dir=backup_dir,
            archive_rel="CUSTOM/TEST_WORLD.IMG",
        )
    assert not lock_dir.exists()
    assert len(records) == 2

    patched_sha256 = hashlib.sha256(img_path.read_bytes()).hexdigest()
    assert patched_sha256 != orig_img_sha256

    # Now execute Rollback and verify 100% bit-for-bit identical SHA-256!
    with GameWriteLock(lock_dir=lock_dir, task_label="pytest-img-rollback", resources=["CUSTOM/TEST_WORLD.IMG"]):
        rollback_img_archive_members(
            game_root=game_dir,
            backup_dir=backup_dir,
            applied_records=records,
        )
    assert not lock_dir.exists()

    restored_sha256 = hashlib.sha256(img_path.read_bytes()).hexdigest()
    assert restored_sha256 == orig_img_sha256


def test_fastapi_endpoints():
    """Verify the FastAPI routes respond cleanly."""
    client = TestClient(app)

    r_state = client.get("/api/state")
    assert r_state.status_code == 200
    st = r_state.json()
    assert st["summary"]["near_txds"] >= 1400

    r_cat = client.get("/api/catalog?category=CAT1_BILLBOARDS&limit=10")
    assert r_cat.status_code == 200
    cat_data = r_cat.json()
    assert cat_data["total"] >= 100
    first_txd = cat_data["items"][0]["txd_name"]

    r_txd = client.get(f"/api/txd/{first_txd}")
    assert r_txd.status_code == 200
    txd_detail = r_txd.json()
    assert len(txd_detail["rasters"]) >= 1
    first_raster = txd_detail["rasters"][0]["name"]

    r_png = client.get(f"/api/txd/{first_txd}/raster/{first_raster}/png")
    assert r_png.status_code == 200
    assert r_png.headers["content-type"] == "image/png"

    r_prev = client.post(
        "/api/convert/preview",
        json={
            "txd_name": first_txd,
            "raster_name": first_raster,
            "category": "CAT1_BILLBOARDS",
            "options": {"resolution_tier": "recommended", "auto_sync_lod": True},
            "sample_theme": "cyber_anime",
            "sample_title": "PYTEST WIBU",
        },
    )
    assert r_prev.status_code == 200
    prev_json = r_prev.json()
    assert prev_json["committed"] is False
    assert prev_json["after_png"].startswith("data:image/png;base64,")

    r_mon = client.get("/api/monuments")
    assert r_mon.status_code == 200
    assert r_mon.json()["total"] >= 50
