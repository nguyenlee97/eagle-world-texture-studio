"""
World Surface & Monument Catalog Store + Fast .IMG Member Reader
for Eagle World Texture & Landmark Studio.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

LAB_GAME_ROOT = Path(r"C:\Users\LENOVO\Desktop\GTA\Project Eagle\gta sa gameplay-lab")
SKIN_LIBRARY_CATALOG = Path(
    r"C:\Users\LENOVO\Desktop\GTA\Skin Library\Project Eagle\numbered-skins\catalog.json"
)
SECTOR = 2048


class WorldCatalogStore:
    def __init__(self, data_dir: Path, game_root: Path = LAB_GAME_ROOT):
        self.data_dir = data_dir
        self.game_root = game_root
        self.catalog_path = data_dir / "world_catalog.json"
        self.staging_dir = data_dir / "staging"
        self.staging_manifest_path = self.staging_dir / "staging_manifest.json"
        self.preview_cache_dir = data_dir / "preview_cache"

        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.preview_cache_dir.mkdir(parents=True, exist_ok=True)

        self._catalog: Optional[Dict[str, Any]] = None
        self._txd_by_name: Dict[str, Dict[str, Any]] = {}
        self._monument_by_id: Dict[int, Dict[str, Any]] = {}
        self._img_index_cache: Dict[str, Dict[str, Tuple[int, int, str]]] = {}
        self._skin_library_cache: Optional[List[Dict[str, Any]]] = None

    def load_catalog(self) -> Dict[str, Any]:
        if self._catalog is None:
            if not self.catalog_path.exists():
                raise FileNotFoundError(f"Missing world_catalog.json at {self.catalog_path}")
            self._catalog = json.loads(self.catalog_path.read_text(encoding="utf-8"))
            self._txd_by_name = {
                item["txd_name"].lower(): item for item in self._catalog.get("txds", [])
            }
            self._monument_by_id = {
                int(item["model_id"]): item for item in self._catalog.get("monuments", [])
            }
        return self._catalog

    def get_txd_entry(self, txd_name: str) -> Optional[Dict[str, Any]]:
        self.load_catalog()
        key = txd_name.lower()
        if key.endswith(".txd"):
            key = key[:-4]
        return self._txd_by_name.get(key)

    def get_monument_entry(self, model_id: int) -> Optional[Dict[str, Any]]:
        self.load_catalog()
        return self._monument_by_id.get(int(model_id))

    def load_staging_manifest(self) -> Dict[str, Any]:
        if not self.staging_manifest_path.exists():
            return {"updated_at": None, "staged_txds": {}, "staged_statues": {}}
        try:
            data = json.loads(self.staging_manifest_path.read_text(encoding="utf-8"))
            data.setdefault("staged_txds", {})
            data.setdefault("staged_statues", {})
            return data
        except Exception:
            return {"updated_at": None, "staged_txds": {}, "staged_statues": {}}

    def save_staging_manifest(self, manifest: Dict[str, Any]) -> None:
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.staging_manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.staging_manifest_path)

    def _get_img_directory(self, img_rel: str) -> Dict[str, Tuple[int, int, str]]:
        norm_rel = img_rel.replace("\\", "/")
        key = norm_rel.lower()
        if key in self._img_index_cache:
            return self._img_index_cache[key]

        img_path = (self.game_root / norm_rel).resolve()
        if not img_path.exists():
            raise FileNotFoundError(f"IMG archive not found: {img_rel}")

        entries: Dict[str, Tuple[int, int, str]] = {}
        with img_path.open("rb") as f:
            hdr = f.read(8)
            if len(hdr) < 8 or hdr[:4] != b"VER2":
                raise ValueError(f"Invalid VER2 IMG header: {img_rel}")
            count = struct.unpack("<I", hdr[4:8])[0]
            dir_raw = f.read(count * 32)
            for i in range(count):
                off, stream_sec, arch_sec, raw_name = struct.unpack_from("<IHH24s", dir_raw, i * 32)
                sectors = arch_sec or stream_sec
                name = raw_name.split(b"\x00", 1)[0].decode("latin1", errors="ignore").strip()
                entries[name.lower()] = (off, sectors, name)

        self._img_index_cache[key] = entries
        return entries

    def invalidate_img_cache(self, img_rel: Optional[str] = None) -> None:
        if img_rel is None:
            self._img_index_cache.clear()
        else:
            self._img_index_cache.pop(img_rel.replace("\\", "/").lower(), None)

    def read_member_from_img(
        self,
        img_rel: str,
        member_name: str,
        trim_rw_size: bool = True,
    ) -> bytes:
        """Read a .TXD or .DFF member directly from an .IMG archive in gameplay-lab."""
        entries = self._get_img_directory(img_rel)
        mkey = member_name.lower()
        if mkey not in entries:
            raise KeyError(f"Member '{member_name}' not found in '{img_rel}'")
        off, sectors, _ = entries[mkey]
        img_path = (self.game_root / img_rel.replace("\\", "/")).resolve()
        with img_path.open("rb") as f:
            f.seek(off * SECTOR)
            raw = f.read(sectors * SECTOR)

        if trim_rw_size and len(raw) >= 12:
            chunk_type, chunk_size, _ = struct.unpack_from("<III", raw, 0)
            if chunk_type in (0x10, 0x16):
                if (chunk_size >> 24) == 1 and chunk_size > len(raw):
                    chunk_size &= 0x00FFFFFF
                actual_len = 12 + chunk_size
                if 12 <= actual_len <= len(raw):
                    return raw[:actual_len]
        return raw

    def read_txd_bytes(self, txd_name: str, prefer_staged: bool = True) -> Tuple[bytes, bool, str]:
        """
        Return (txd_bytes, is_staged, source_label) for a catalog TXD name.
        """
        short = txd_name.lower()
        if short.endswith(".txd"):
            short = short[:-4]

        if prefer_staged:
            staged_file = self.staging_dir / "txds" / f"{short}.txd"
            if staged_file.exists():
                return staged_file.read_bytes(), True, f"STAGED/{short}.txd"

        entry = self.get_txd_entry(short)
        if not entry:
            raise KeyError(f"TXD '{txd_name}' not found in catalog")

        img_rel = entry["primary_img"]
        member = entry["txd_filename"]
        return self.read_member_from_img(img_rel, member), False, img_rel

    def read_dff_bytes(
        self,
        dff_name: str,
        dff_img: Optional[str] = None,
        prefer_staged: bool = True,
    ) -> Tuple[bytes, bool, str]:
        """Return (dff_bytes, is_staged, source_label) for a .DFF model."""
        short = dff_name.lower()
        if short.endswith(".dff"):
            short = short[:-4]

        if prefer_staged:
            staged_dff = self.staging_dir / "statues" / f"{short}.dff"
            if staged_dff.exists():
                return staged_dff.read_bytes(), True, f"STAGED/{short}.dff"

        member = f"{short}.dff"
        if dff_img:
            try:
                return self.read_member_from_img(dff_img, member), False, dff_img
            except KeyError:
                pass

        # Search across catalog TXD model references
        self.load_catalog()
        for txd_item in self._catalog.get("txds", []):
            for m in txd_item.get("models", []):
                if m["dff"].lower() == short:
                    cand_img = m.get("dff_img") or txd_item["primary_img"]
                    try:
                        return self.read_member_from_img(cand_img, member), False, cand_img
                    except KeyError:
                        pass

        for mon in self._catalog.get("monuments", []):
            if mon["dff_name"].lower() == short:
                try:
                    return self.read_member_from_img(mon["dff_img"], member), False, mon["dff_img"]
                except KeyError:
                    pass

        raise KeyError(f"DFF '{dff_name}' could not be located in mounted IMG archives")

    def list_skin_library_characters(self, query: str = "", limit: int = 80) -> List[Dict[str, Any]]:
        """
        Load available anime character 3D models from Skin Library so the user can pick
        any character to turn into a 3D Monument / Statue!
        """
        if self._skin_library_cache is None:
            self._skin_library_cache = []
            if SKIN_LIBRARY_CATALOG.exists():
                try:
                    cat = json.loads(SKIN_LIBRARY_CATALOG.read_text(encoding="utf-8"))
                    for s in cat.get("skins", []):
                        if s.get("policyHold"):
                            continue
                        dff_p = s.get("dffPath") or ""
                        txd_p = s.get("txdPath") or ""
                        if not dff_p or not txd_p:
                            continue
                        self._skin_library_cache.append({
                            "id": s.get("id"),
                            "key": s.get("key"),
                            "name": s.get("displayName") or s.get("title") or s.get("key"),
                            "series": s.get("series") or [],
                            "dff_path": dff_p,
                            "txd_path": txd_p,
                            "size_kb": round((s.get("bytes") or 0) / 1024.0, 1),
                        })
                except Exception:
                    pass

        q = query.strip().lower()
        results = []
        for item in self._skin_library_cache:
            if q:
                blob = f"{item['name']} {' '.join(item['series'])} {item['key']}".lower()
                if q not in blob:
                    continue
            results.append(item)
            if len(results) >= limit:
                break
        return results


def generate_sample_wibu_artwork(
    theme: str = "cyber_anime",
    width: int = 512,
    height: int = 256,
    title_text: str = "PROJECT EAGLE",
    subtitle_text: str = "WIBU WORLD FESTIVAL 2026",
) -> Image.Image:
    """
    Generate a stylized, high-detail sample anime/cyberpunk promotional artwork
    for instant testing of the 4 Surface Converters inside the Studio.
    """
    THEMES = {
        "cyber_anime": ((24, 14, 58), (88, 22, 112), (255, 72, 176), (0, 238, 255)),
        "genshin_gold": ((16, 28, 54), (36, 62, 108), (255, 212, 92), (138, 236, 255)),
        "akihabara_neon": ((38, 10, 32), (18, 12, 42), (255, 56, 108), (255, 220, 64)),
        "graffiti_tag": ((0, 0, 0), (0, 0, 0), (56, 248, 152), (255, 88, 212)),
    }
    bg1, bg2, acc1, acc2 = THEMES.get(theme, THEMES["cyber_anime"])

    if theme == "graffiti_tag":
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    else:
        # Smooth gradient background
        yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        xx = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
        t = np.clip(yy * 0.7 + xx * 0.3, 0.0, 1.0)
        arr = np.zeros((height, width, 4), dtype=np.uint8)
        for c in range(3):
            arr[:, :, c] = (bg1[c] * (1.0 - t) + bg2[c] * t).astype(np.uint8)
        arr[:, :, 3] = 255
        img = Image.fromarray(arr, mode="RGBA")

    draw = ImageDraw.Draw(img)

    # Decorative geometric anime sun/halos and diagonal stripes
    cx, cy = int(width * 0.75), int(height * 0.48)
    rad = int(min(width, height) * 0.36)
    draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(*acc1, 215), outline=(*acc2, 255), width=3)
    in_r = int(rad * 0.68)
    draw.ellipse([cx - in_r, cy - in_r, cx + in_r, cy + in_r], fill=(*bg1, 230), outline=(*acc2, 240), width=2)

    # Stylized character star emblem inside the circle
    pts = []
    for i in range(8):
        ang = i * math.pi / 4.0
        r_use = in_r * (0.82 if i % 2 == 0 else 0.32)
        pts.append((cx + r_use * math.cos(ang), cy + r_use * math.sin(ang)))
    draw.polygon(pts, fill=(*acc2, 245))

    # Border & corner accents
    if theme != "graffiti_tag":
        draw.rectangle([6, 6, width - 7, height - 7], outline=(*acc2, 220), width=2)
        draw.rectangle([12, 12, width - 13, height - 13], outline=(*acc1, 160), width=1)

    # Load font
    font_lg = ImageFont.load_default()
    font_sm = ImageFont.load_default()
    for fp in ("C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/impact.ttf"):
        if Path(fp).exists():
            try:
                font_lg = ImageFont.truetype(fp, max(16, int(min(width, height) * 0.14)))
                font_sm = ImageFont.truetype(fp, max(11, int(min(width, height) * 0.075)))
                break
            except Exception:
                pass

    tx = int(width * 0.06)
    ty = int(height * 0.26)
    draw.text((tx + 2, ty + 2), title_text, fill=(0, 0, 0, 220), font=font_lg)
    draw.text((tx, ty), title_text, fill=(255, 255, 255, 255), font=font_lg)
    draw.text((tx + 1, ty + int(min(width, height) * 0.20) + 1), subtitle_text, fill=(0, 0, 0, 210), font=font_sm)
    draw.text((tx, ty + int(min(width, height) * 0.20)), subtitle_text, fill=(*acc2, 255), font=font_sm)

    return img
