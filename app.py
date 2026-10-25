"""
Eagle World Texture & Landmark Studio — FastAPI Backend Server.

Provides:
- Multi-dimensional browsing & filtering across 1,635 World .TXD dictionaries and 93 3D Monuments.
- Live RenderWare .TXD raster decoding (DXT1/DXT3/DXT5/RGBA/PAL8 -> PNG) and .DFF 3D UV/Aspect analysis.
- 4 Category-Specific 2D Surface Converters (Billboards with Dual-Sync LOD, Murals/Graffiti/Banners
  with Wall-Grain Blend & Alpha DXT5, Neon/LED Screens with Sub-Region Atlas Patching & Scanlines,
  Posters/Bus Stops/Vending with UV-Unstretch & Frame Masking).
- Dedicated 3D Monument / Statue Swapper & Baker (convert Skin Library characters or custom .DFF+.TXD
  into static prelit GTA SA monument .DFF+.TXD with optional stepped stone pedestal & Marble/Bronze/Gold finishes).
- Staged Mod Pack management + Atomic .IMG Deploy & Bit-for-Bit Rollback with `game-write.lock`.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from core.catalog_store import (
    LAB_GAME_ROOT,
    WorldCatalogStore,
    generate_sample_wibu_artwork,
)
from core.converters import CONVERTER_PRESET_SCHEMAS, convert_surface_raster
from core.deployer import StudioDeployManager
from core.rw_dff import (
    STATUE_FINISH_PRESETS,
    analyze_dff_materials_and_uvs,
    build_statue_dff_and_txd,
)
from core.rw_txd import (
    RwRasterEntry,
    decode_raster_to_pil,
    parse_txd_bytes,
    replace_rasters_in_txd_bytes,
)

TOOL_DIR = Path(__file__).resolve().parent
DATA_DIR = TOOL_DIR / "data"
STATIC_DIR = TOOL_DIR / "static"
SKIN_VIEWER_DIR = TOOL_DIR.parent / "skin-viewer"

STATIC_DIR.mkdir(parents=True, exist_ok=True)

store = WorldCatalogStore(DATA_DIR, game_root=LAB_GAME_ROOT)
deployer = StudioDeployManager(store)

app = FastAPI(title="Eagle World Texture & Landmark Studio", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if (SKIN_VIEWER_DIR / "node_modules").exists():
    app.mount(
        "/node_modules",
        StaticFiles(directory=str(SKIN_VIEWER_DIR / "node_modules")),
        name="node_modules",
    )
if (SKIN_VIEWER_DIR / "vendor").exists():
    app.mount(
        "/vendor",
        StaticFiles(directory=str(SKIN_VIEWER_DIR / "vendor")),
        name="vendor",
    )


def _pil_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _find_matching_lod_raster(
    near_raster_name: str,
    lod_entries: List[RwRasterEntry],
) -> Optional[RwRasterEntry]:
    """
    Find the corresponding LOD raster inside a paired *_lod.txd dictionary.
    Matches exact name, `lod` + name, name + `_lod`, or substring match.
    """
    nl = near_raster_name.lower()
    for e in lod_entries:
        el = e.name.lower()
        if el == nl or el == f"lod{nl}" or el == f"lod_{nl}" or el == f"{nl}_lod":
            return e
    # Strip trailing digits or prefixes
    for e in lod_entries:
        el = e.name.lower()
        if len(nl) >= 5 and (nl in el or el.replace("lod_", "").replace("lod", "") == nl):
            return e
    return None


@app.get("/", response_class=HTMLResponse)
def index_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/state")
def api_get_state() -> Dict[str, Any]:
    cat = store.load_catalog()
    staging = store.load_staging_manifest()
    txs = deployer.list_transactions()
    return {
        "summary": cat.get("summary", {}),
        "converter_schemas": CONVERTER_PRESET_SCHEMAS,
        "statue_finishes": {k: v["label"] for k, v in STATUE_FINISH_PRESETS.items()},
        "staging": staging,
        "transactions": txs,
    }


@app.get("/api/catalog")
def api_query_catalog(
    category: str = Query("ALL"),
    region: str = Query("ALL"),
    tier: str = Query("ALL"),
    q: str = Query(""),
    include_lod: bool = Query(False),
    sort: str = Query("recommended"),
    offset: int = Query(0, ge=0),
    limit: int = Query(60, ge=1, le=300),
) -> Dict[str, Any]:
    cat = store.load_catalog()
    staging = store.load_staging_manifest()
    staged_txds = staging.get("staged_txds", {})

    q_clean = q.strip().lower()
    filtered = []

    for item in cat.get("txds", []):
        if not include_lod and item.get("is_lod"):
            continue
        if category != "ALL":
            if item["primary_category"] != category and category not in item.get("categories", []):
                continue
        if region != "ALL" and item.get("region") != region:
            continue

        short = item["txd_name"].lower()
        is_staged = short in staged_txds
        tiers_present = {x["tier"] for x in item.get("all_imgs", [])}

        if tier == "STAGED" and not is_staged:
            continue
        elif tier == "INFLATED" and item.get("inflation_ratio", 1.0) <= 1.2:
            continue
        elif tier in ("GROUP_B1", "GROUP_A", "WIBU01", "BASE_NEAR") and tier not in tiers_present:
            continue

        if q_clean:
            m_names = " ".join(m["dff"] for m in item.get("models", []))
            r_names = " ".join(r["name"] for r in item.get("sample_rasters", []))
            blob = f"{short} {item['primary_img']} {item['region']} {m_names} {r_names}".lower()
            if q_clean not in blob:
                continue

        row = dict(item)
        row["is_staged"] = is_staged
        if is_staged:
            row["staged_info"] = staged_txds[short]
        filtered.append(row)

    if sort == "inflation":
        filtered.sort(key=lambda x: (-x.get("inflation_ratio", 1.0), -x.get("file_size_kb", 0)))
    elif sort == "placements":
        filtered.sort(key=lambda x: (-x.get("total_placements", 0), x["txd_name"]))
    elif sort == "size_desc":
        filtered.sort(key=lambda x: (-x.get("file_size_kb", 0), x["txd_name"]))
    elif sort == "name":
        filtered.sort(key=lambda x: x["txd_name"])
    else:
        # recommended: staged first, then high inflation or high placements
        filtered.sort(
            key=lambda x: (
                0 if x.get("is_staged") else 1,
                1 if x.get("is_lod") else 0,
                -(1 if x.get("inflation_ratio", 1.0) > 2.0 else 0),
                -min(x.get("total_placements", 0), 25),
                -x.get("inflation_ratio", 1.0),
                x["txd_name"],
            )
        )

    total = len(filtered)
    slice_rows = filtered[offset : offset + limit]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": slice_rows,
    }


@app.get("/api/txd/{txd_name}")
def api_inspect_txd(txd_name: str) -> Dict[str, Any]:
    entry = store.get_txd_entry(txd_name)
    if not entry:
        raise HTTPException(status_code=404, detail=f"TXD '{txd_name}' not found in catalog")

    try:
        active_bytes, is_staged, active_source = store.read_txd_bytes(txd_name, prefer_staged=True)
        _, _, active_entries, _ = parse_txd_bytes(active_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read TXD '{txd_name}': {exc}")

    # Analyze primary DFF models to attach 3D physical surface dimensions & UV aspect ratios
    dff_tex_map: Dict[str, Dict[str, Any]] = {}
    analyzed_dffs: List[Dict[str, Any]] = []
    for m in entry.get("models", [])[:4]:
        try:
            dff_bytes, dff_staged, dff_src = store.read_dff_bytes(
                m["dff"], dff_img=m.get("dff_img"), prefer_staged=True
            )
            info = analyze_dff_materials_and_uvs(dff_bytes)
            if info.get("valid"):
                analyzed_dffs.append({
                    "dff": m["dff"],
                    "id": m["id"],
                    "placements": m["placements"],
                    "is_tobj": m["is_tobj"],
                    "dff_img": dff_src,
                    "is_staged": dff_staged,
                    "vertex_count": info["vertex_count"],
                    "triangle_count": info["triangle_count"],
                    "dimensions_m": info["dimensions_m"],
                })
                for tkey, tval in info.get("textures", {}).items():
                    if tkey not in dff_tex_map or tval["area_m2"] > dff_tex_map[tkey]["area_m2"]:
                        tval_copy = dict(tval)
                        tval_copy["source_dff"] = m["dff"]
                        dff_tex_map[tkey] = tval_copy
        except Exception:
            pass

    # Check paired LOD TXD rasters if any
    lod_entries: List[RwRasterEntry] = []
    paired_lod_name = entry.get("paired_lod_txd")
    if paired_lod_name:
        try:
            lod_bytes, _, _ = store.read_txd_bytes(paired_lod_name, prefer_staged=True)
            _, _, lod_entries, _ = parse_txd_bytes(lod_bytes)
        except Exception:
            pass

    staging = store.load_staging_manifest()
    staged_txd_rec = staging.get("staged_txds", {}).get(entry["txd_name"].lower(), {})
    modified_rasters = staged_txd_rec.get("modified_rasters", {})

    rasters_out = []
    for r in active_entries:
        rk = r.name.lower()
        dff_info = dff_tex_map.get(rk)
        matched_lod = _find_matching_lod_raster(r.name, lod_entries) if lod_entries else None
        rasters_out.append({
            "index": r.index,
            "name": r.name,
            "mask_name": r.mask_name,
            "width": r.width,
            "height": r.height,
            "aspect": r.aspect_label,
            "format": r.format_label,
            "has_alpha": r.has_alpha,
            "mipmaps": r.mip_levels,
            "is_pot": r.is_pot,
            "size_kb": round(r.total_chunk_bytes / 1024.0, 2),
            "is_modified": rk in modified_rasters,
            "modified_report": modified_rasters.get(rk),
            "dff_uv_info": dff_info,
            "matched_lod_raster": (
                {
                    "lod_txd": paired_lod_name,
                    "name": matched_lod.name,
                    "width": matched_lod.width,
                    "height": matched_lod.height,
                    "format": matched_lod.format_label,
                }
                if matched_lod
                else None
            ),
        })

    return {
        "entry": entry,
        "is_staged": is_staged,
        "active_source": active_source,
        "active_file_size_kb": round(len(active_bytes) / 1024.0, 2),
        "analyzed_dffs": analyzed_dffs,
        "rasters": rasters_out,
    }


@app.get("/api/txd/{txd_name}/raster/{raster_name}/png")
def api_get_raster_png(
    txd_name: str,
    raster_name: str,
    source: str = Query("active"),
) -> Response:
    prefer_staged = source != "original"
    try:
        txd_bytes, _, _ = store.read_txd_bytes(txd_name, prefer_staged=prefer_staged)
        _, _, entries, _ = parse_txd_bytes(txd_bytes)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    rkey = raster_name.lower()
    target = next((e for e in entries if e.name.lower() == rkey), None)
    if target is None:
        raise HTTPException(status_code=404, detail=f"Raster '{raster_name}' not found in '{txd_name}'")

    pil_img = decode_raster_to_pil(target)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/api/assets/txd/{txd_name}.txd")
def api_serve_raw_txd(txd_name: str, staged: int = Query(1)) -> Response:
    try:
        raw, _, _ = store.read_txd_bytes(txd_name, prefer_staged=bool(staged))
        return Response(content=raw, media_type="application/octet-stream", headers={"Cache-Control": "no-store"})
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/api/assets/dff/{dff_name}.dff")
def api_serve_raw_dff(
    dff_name: str,
    dff_img: Optional[str] = Query(None),
    staged: int = Query(1),
) -> Response:
    try:
        raw, _, _ = store.read_dff_bytes(dff_name, dff_img=dff_img, prefer_staged=bool(staged))
        return Response(content=raw, media_type="application/octet-stream", headers={"Cache-Control": "no-store"})
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))


class ConvertSurfacePayload(BaseModel):
    txd_name: str
    raster_name: str
    category: str = "CAT1_BILLBOARDS"
    options: Dict[str, Any] = {}
    image_b64: Optional[str] = None
    sample_theme: Optional[str] = None
    sample_title: Optional[str] = None
    sample_subtitle: Optional[str] = None


def _decode_b64_payload(b64_str: str) -> bytes:
    if "," in b64_str and b64_str.startswith("data:"):
        b64_str = b64_str.split(",", 1)[1]
    return base64.b64decode(b64_str)


def _run_conversion_job(
    payload: ConvertSurfacePayload,
    commit_to_staging: bool,
) -> Dict[str, Any]:
    txd_name = payload.txd_name
    raster_name = payload.raster_name
    category = payload.category
    opts = dict(payload.options or {})

    entry = store.get_txd_entry(txd_name)
    if not entry:
        raise HTTPException(status_code=404, detail=f"TXD '{txd_name}' not found")

    # Always read the ORIGINAL raster from game IMG for clean Wall-Blend / Frame-Border / Before comparison
    orig_txd_bytes, _, _ = store.read_txd_bytes(txd_name, prefer_staged=False)
    _, _, orig_entries, _ = parse_txd_bytes(orig_txd_bytes)
    rkey = raster_name.lower()
    orig_raster = next((e for e in orig_entries if e.name.lower() == rkey), None)
    if orig_raster is None:
        raise HTTPException(status_code=404, detail=f"Raster '{raster_name}' not found in '{txd_name}'")

    orig_decoded_pil = decode_raster_to_pil(orig_raster)

    # Load user image (from base64 data URL) or generate sample Wibu artwork
    if payload.image_b64:
        raw_img_bytes = _decode_b64_payload(payload.image_b64)
        user_pil = Image.open(io.BytesIO(raw_img_bytes)).convert("RGBA")
    else:
        theme = payload.sample_theme or ("graffiti_tag" if "tag" in rkey else "cyber_anime")
        user_pil = generate_sample_wibu_artwork(
            theme=theme,
            width=max(256, orig_raster.width * 2),
            height=max(256, orig_raster.height * 2),
            title_text=payload.sample_title or "WIBU WORLD",
            subtitle_text=payload.sample_subtitle or f"{entry['txd_name'].upper()} · {orig_raster.name}",
        )

    # Get DFF UV info for this raster
    dff_tex_info = None
    for m in entry.get("models", [])[:3]:
        try:
            dff_bytes, _, _ = store.read_dff_bytes(m["dff"], dff_img=m.get("dff_img"), prefer_staged=False)
            dinfo = analyze_dff_materials_and_uvs(dff_bytes)
            if rkey in dinfo.get("textures", {}):
                dff_tex_info = dinfo["textures"][rkey]
                break
        except Exception:
            pass

    # Check paired LOD TXD if auto_sync_lod is enabled
    paired_lod_name = entry.get("paired_lod_txd")
    lod_orig_entry = None
    lod_active_bytes = None
    if paired_lod_name and opts.get("auto_sync_lod", True):
        try:
            lod_active_bytes, _, _ = store.read_txd_bytes(paired_lod_name, prefer_staged=True)
            _, _, lod_entries, _ = parse_txd_bytes(lod_active_bytes)
            lod_orig_entry = _find_matching_lod_raster(orig_raster.name, lod_entries)
        except Exception:
            pass

    conv = convert_surface_raster(
        category=category,
        user_img=user_pil,
        orig_entry=orig_raster,
        orig_decoded_img=orig_decoded_pil,
        dff_tex_info=dff_tex_info,
        options=opts,
        lod_orig_entry=lod_orig_entry,
    )

    staging_manifest = None
    if commit_to_staging:
        active_txd_bytes, _, _ = store.read_txd_bytes(txd_name, prefer_staged=True)
        new_txd_bytes = replace_rasters_in_txd_bytes(
            active_txd_bytes, {orig_raster.name: conv["near_entry"]}
        )

        new_lod_txd_bytes = None
        if conv["lod_entry"] is not None and paired_lod_name and lod_active_bytes:
            new_lod_txd_bytes = replace_rasters_in_txd_bytes(
                lod_active_bytes, {conv["lod_entry"].name: conv["lod_entry"]}
            )

        staging_manifest = deployer.stage_txd_update(
            txd_name=entry["txd_name"],
            new_txd_bytes=new_txd_bytes,
            raster_report=conv["report"],
            lod_txd_name=paired_lod_name if new_lod_txd_bytes else None,
            new_lod_txd_bytes=new_lod_txd_bytes,
        )

    return {
        "committed": commit_to_staging,
        "report": conv["report"],
        "before_png": _pil_to_data_url(orig_decoded_pil),
        "after_png": _pil_to_data_url(conv["near_preview_pil"]),
        "lod_after_png": _pil_to_data_url(conv["lod_preview_pil"]) if conv["lod_preview_pil"] else None,
        "staging": staging_manifest,
    }


@app.post("/api/convert/preview")
def api_convert_preview(payload: ConvertSurfacePayload) -> Dict[str, Any]:
    return _run_conversion_job(payload=payload, commit_to_staging=False)


@app.post("/api/convert/stage")
def api_convert_stage(payload: ConvertSurfacePayload) -> Dict[str, Any]:
    return _run_conversion_job(payload=payload, commit_to_staging=True)


# ---------------------------------------------------------------------------
# Category 5: 3D Monuments & Statues Swapper Routes
# ---------------------------------------------------------------------------

@app.get("/api/monuments")
def api_list_monuments(q: str = Query(""), region: str = Query("ALL")) -> Dict[str, Any]:
    cat = store.load_catalog()
    staging = store.load_staging_manifest()
    staged_statues = staging.get("staged_statues", {})

    q_clean = q.strip().lower()
    items = []
    for m in cat.get("monuments", []):
        if region != "ALL" and m.get("region") != region:
            continue
        if q_clean:
            blob = f"{m['dff_name']} {m['txd_name']} {m['region']} {m['ide']}".lower()
            if q_clean not in blob:
                continue
        row = dict(m)
        dkey = m["dff_name"].lower()
        row["is_staged"] = dkey in staged_statues
        if row["is_staged"]:
            row["staged_info"] = staged_statues[dkey]
        items.append(row)

    return {"total": len(items), "items": items}


@app.get("/api/monuments/{model_id}")
def api_inspect_monument(model_id: int) -> Dict[str, Any]:
    mon = store.get_monument_entry(model_id)
    if not mon:
        raise HTTPException(status_code=404, detail=f"Monument ID {model_id} not found")

    try:
        orig_dff, _, _ = store.read_dff_bytes(mon["dff_name"], dff_img=mon["dff_img"], prefer_staged=False)
        orig_info = analyze_dff_materials_and_uvs(orig_dff)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read monument DFF: {exc}")

    active_dff, is_staged, active_src = store.read_dff_bytes(mon["dff_name"], dff_img=mon["dff_img"], prefer_staged=True)
    active_info = analyze_dff_materials_and_uvs(active_dff) if is_staged else orig_info

    return {
        "monument": mon,
        "is_staged": is_staged,
        "active_source": active_src,
        "original_geometry": orig_info,
        "active_geometry": active_info,
    }


@app.get("/api/skin-library")
def api_search_skin_library(q: str = Query(""), limit: int = Query(60, ge=1, le=200)) -> Dict[str, Any]:
    items = store.list_skin_library_characters(query=q, limit=limit)
    return {"count": len(items), "items": items}


class BakeMonumentPayload(BaseModel):
    model_id: int
    skin_key: Optional[str] = None
    height_scale: float = 1.0
    rotation_deg: float = 0.0
    z_offset_m: float = 0.0
    add_pedestal: bool = True
    statue_finish: str = "white_marble"
    custom_dff_b64: Optional[str] = None
    custom_txd_b64: Optional[str] = None
    custom_dff_name: Optional[str] = None


@app.post("/api/monuments/bake")
def api_bake_monument_statue(payload: BakeMonumentPayload) -> Dict[str, Any]:
    mon = store.get_monument_entry(payload.model_id)
    if not mon:
        raise HTTPException(status_code=404, detail=f"Monument ID {payload.model_id} not found")

    target_dff_bytes, _, _ = store.read_dff_bytes(mon["dff_name"], dff_img=mon["dff_img"], prefer_staged=False)
    target_txd_bytes, _, _ = store.read_txd_bytes(mon["txd_name"], prefer_staged=False)

    source_label = "Custom Upload"
    if payload.custom_dff_b64 and payload.custom_txd_b64:
        src_dff_bytes = _decode_b64_payload(payload.custom_dff_b64)
        src_txd_bytes = _decode_b64_payload(payload.custom_txd_b64)
        source_label = payload.custom_dff_name or "Custom .DFF"
    elif payload.skin_key:
        skins = store.list_skin_library_characters(query=payload.skin_key, limit=50)
        matched = next((s for s in skins if s["key"] == payload.skin_key or str(s["id"]) == str(payload.skin_key)), None)
        if not matched:
            raise HTTPException(status_code=404, detail=f"Skin '{payload.skin_key}' not found in Skin Library")
        src_dff_bytes = Path(matched["dff_path"]).read_bytes()
        src_txd_bytes = Path(matched["txd_path"]).read_bytes()
        source_label = f"{matched['name']} ({matched['key']})"
    else:
        raise HTTPException(
            status_code=400,
            detail="Hãy chọn một nhân vật từ Skin Library hoặc tải lên cặp file .DFF + .TXD tượng mới.",
        )

    try:
        new_dff, new_txd, stats = build_statue_dff_and_txd(
            target_monument_dff_bytes=target_dff_bytes,
            source_dff_bytes=src_dff_bytes,
            source_txd_bytes=src_txd_bytes,
            target_txd_bytes=target_txd_bytes,
            height_scale=payload.height_scale,
            rotation_deg=payload.rotation_deg,
            z_offset_m=payload.z_offset_m,
            add_pedestal=payload.add_pedestal,
            statue_finish=payload.statue_finish,
            frame_name=mon["dff_name"],
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Lỗi đóng băng mô hình tượng 3D: {exc}")

    stats["source_model"] = source_label
    stats["height_scale"] = payload.height_scale
    stats["rotation_deg"] = payload.rotation_deg
    stats["z_offset_m"] = payload.z_offset_m

    manifest = deployer.stage_statue_update(
        monument_id=payload.model_id,
        dff_name=mon["dff_name"],
        txd_name=mon["txd_name"],
        new_dff_bytes=new_dff,
        new_txd_bytes=new_txd,
        statue_meta=stats,
    )
    return {
        "ok": True,
        "monument": mon,
        "stats": stats,
        "staging": manifest,
    }


# ---------------------------------------------------------------------------
# Staging, Deploy & Rollback Routes
# ---------------------------------------------------------------------------

@app.delete("/api/staging/{item_type}/{name}")
def api_unstage_item(item_type: str, name: str) -> Dict[str, Any]:
    manifest = deployer.unstage_item(item_type=item_type, name=name)
    return {"ok": True, "staging": manifest}


@app.post("/api/deploy")
def api_deploy_staged() -> Dict[str, Any]:
    try:
        tx = deployer.deploy_staged_to_game(check_game_closed=True)
        return {"ok": True, "transaction": tx, "transactions": deployer.list_transactions()}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/rollback/{tx_id}")
def api_rollback_tx(tx_id: str) -> Dict[str, Any]:
    try:
        tx = deployer.rollback_transaction(tx_id=tx_id, check_game_closed=True)
        return {"ok": True, "transaction": tx, "transactions": deployer.list_transactions()}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8950, reload=False)
