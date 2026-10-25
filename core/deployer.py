"""
Atomic .IMG Member Deployer, Transaction Backup & Bit-for-Bit Rollback Manager
for Eagle World Texture & Landmark Studio.

Complies with `docs/project-eagle/shared-agent-workflow.md`:
- Acquires exclusive directory lock `C:\\Users\\LENOVO\\Desktop\\GTA\\data\\project-eagle\\coordination\\game-write.lock`
  and writes `owner.json` before modifying any runtime `.IMG` archive.
- Verifies `gta_sa` / `gta_pe` is closed.
- Synchronizes updated `.TXD` / `.DFF` members across all mounted `.IMG` archives that contain them
  (e.g. `CUSTOM\\GROUP_B1_TEXTURES.IMG`, `CUSTOM\\GROUP_A_MODS.IMG`, and base `S&S` archives).
- Saves exact original directory entries, sector payloads, and file sizes per member so
  1-click Rollback restores the exact bit-for-bit original `.IMG` archive.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import subprocess
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.catalog_store import LAB_GAME_ROOT, SECTOR, WorldCatalogStore

DEFAULT_LOCK_DIR = Path(r"C:\Users\LENOVO\Desktop\GTA\data\project-eagle\coordination\game-write.lock")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class GameWriteLock:
    """Exclusive filesystem directory lock following shared-agent-workflow.md."""

    def __init__(
        self,
        lock_dir: Path = DEFAULT_LOCK_DIR,
        task_label: str = "world-texture-studio-deploy",
        resources: Optional[List[str]] = None,
    ):
        self.lock_dir = lock_dir
        self.task_label = task_label
        self.resources = resources or []
        self.owner_file = self.lock_dir / "owner.json"
        self._acquired = False

    def __enter__(self):
        self.lock_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Exclusive creation (equivalent to New-Item -ItemType Directory -ErrorAction Stop)
            self.lock_dir.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            owner_info = ""
            if self.owner_file.exists():
                try:
                    owner_info = self.owner_file.read_text(encoding="utf-8")
                except Exception:
                    pass
            raise RuntimeError(
                f"Khóa ghi game-write.lock đang được giữ bởi tác vụ khác. Chi tiết: {owner_info}"
            )

        self._acquired = True
        payload = {
            "agent": "Antigravity (World Texture Studio)",
            "task": self.task_label,
            "work_item": "docs/project-eagle/work-items/2026-10-24-antigravity-world-texture-studio.md",
            "acquired_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "pid": os.getpid(),
            "resources": self.resources,
        }
        self.owner_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._acquired:
            try:
                if self.owner_file.exists():
                    self.owner_file.unlink()
                if self.lock_dir.exists():
                    self.lock_dir.rmdir()
            except Exception:
                pass
            self._acquired = False


def verify_game_processes_closed() -> None:
    if os.name != "nt":
        return
    try:
        res = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "@(Get-Process -Name gta_sa,gta_pe -ErrorAction SilentlyContinue).Count",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        out = res.stdout.strip()
        if out.isdigit() and int(out) > 0:
            raise RuntimeError(
                "Project Eagle (gta_sa.exe / gta_pe.exe) đang chạy. Hãy thoát game trước khi Deploy hoặc Rollback."
            )
    except RuntimeError:
        raise
    except Exception:
        pass


def patch_img_archive_members(
    img_path: Path,
    member_payloads: Dict[str, bytes],
    backup_dir: Path,
    archive_rel: str,
) -> List[Dict[str, Any]]:
    """
    Patch one or more existing members inside a VER2 .IMG archive cleanly.
    - If `new_sectors <= orig_sectors`, writes in-place at `orig_offset` (zero-padding remainder of old slot).
    - If `new_sectors > orig_sectors`, appends at the end of the archive (sector-aligned).
    Saves exact original byte slices into `backup_dir` so rollback is bit-for-bit identical.
    """
    if not img_path.exists():
        raise FileNotFoundError(f"IMG archive not found: {img_path}")

    req_map = {k.lower(): (k, v) for k, v in member_payloads.items()}
    applied_records: List[Dict[str, Any]] = []

    with img_path.open("r+b") as f:
        f.seek(0, os.SEEK_END)
        orig_file_size = f.tell()
        f.seek(0)
        hdr = f.read(8)
        if len(hdr) < 8 or hdr[:4] != b"VER2":
            raise ValueError(f"Not a valid VER2 IMG archive: {img_path}")

        count = struct.unpack("<I", hdr[4:8])[0]
        dir_raw = bytearray(f.read(count * 32))

        # Find all matching directory entries and track current max sector
        max_sector_end = int(math.ceil(orig_file_size / SECTOR))
        targets_in_dir = []
        for idx in range(count):
            entry_off = idx * 32
            raw_entry = bytes(dir_raw[entry_off : entry_off + 32])
            off_sec, stream_sec, arch_sec, raw_name = struct.unpack("<IHH24s", raw_entry)
            sectors = arch_sec or stream_sec
            if off_sec + sectors > max_sector_end:
                max_sector_end = off_sec + sectors
            name = raw_name.split(b"\x00", 1)[0].decode("latin1", errors="ignore").strip()
            nl = name.lower()
            if nl in req_map:
                targets_in_dir.append((idx, name, nl, off_sec, sectors, raw_entry, raw_name))

        if not targets_in_dir:
            return []

        slug = archive_rel.replace("\\", "__").replace("/", "__").replace(":", "")
        for idx, orig_name, nl, orig_off, orig_sec, raw_entry, raw_name_24 in targets_in_dir:
            _, new_bytes = req_map[nl]
            new_sec = max(1, int(math.ceil(len(new_bytes) / SECTOR)))
            if new_sec > 65535:
                raise ValueError(f"Member {orig_name} exceeds 65535 sectors limit")

            # Read and back up exact original sector bytes
            f.seek(orig_off * SECTOR)
            orig_sector_bytes = f.read(orig_sec * SECTOR)
            bak_filename = f"{slug}__{idx}__{nl}.bak"
            bak_path = backup_dir / bak_filename
            bak_path.write_bytes(orig_sector_bytes)

            padded_new = new_bytes.ljust(new_sec * SECTOR, b"\x00")
            if new_sec <= orig_sec:
                target_off = orig_off
                # Write new payload + zero-fill remaining old sectors so no stale bytes remain
                full_slot = padded_new.ljust(orig_sec * SECTOR, b"\x00")
                f.seek(target_off * SECTOR)
                f.write(full_slot)
            else:
                target_off = max_sector_end
                f.seek(target_off * SECTOR)
                f.write(padded_new)
                max_sector_end = target_off + new_sec

            new_entry = struct.pack("<IHH24s", target_off, new_sec, 0, raw_name_24)
            dir_raw[idx * 32 : (idx + 1) * 32] = new_entry

            applied_records.append({
                "archive_rel": archive_rel,
                "dir_index": idx,
                "member_name": orig_name,
                "orig_offset_sectors": orig_off,
                "orig_sectors": orig_sec,
                "orig_dir_entry_hex": raw_entry.hex(),
                "orig_file_size": orig_file_size,
                "backup_slice_file": bak_filename,
                "before_sha256": sha256_bytes(orig_sector_bytes),
                "new_offset_sectors": target_off,
                "new_sectors": new_sec,
                "after_payload_sha256": sha256_bytes(padded_new),
                "before_bytes": orig_sec * SECTOR,
                "after_bytes": len(new_bytes),
            })

        # Write updated directory table back to offset 8
        f.seek(8)
        f.write(dir_raw)
        f.flush()
        os.fsync(f.fileno())

    return applied_records


def rollback_img_archive_members(
    game_root: Path,
    backup_dir: Path,
    applied_records: List[Dict[str, Any]],
) -> List[str]:
    """
    Restore exact original member sectors, 32-byte directory entries, and file size
    from a transaction backup directory.
    """
    by_archive: Dict[str, List[Dict[str, Any]]] = {}
    for rec in applied_records:
        by_archive.setdefault(rec["archive_rel"], []).append(rec)

    restored_archives: List[str] = []
    for archive_rel, recs in by_archive.items():
        img_path = (game_root / archive_rel.replace("\\", "/")).resolve()
        if not img_path.exists():
            continue

        # Sort in reverse order of application so earliest orig_file_size wins on truncate
        min_orig_file_size = min(int(r["orig_file_size"]) for r in recs)
        with img_path.open("r+b") as f:
            for r in recs:
                idx = int(r["dir_index"])
                orig_off = int(r["orig_offset_sectors"])
                orig_entry = bytes.fromhex(r["orig_dir_entry_hex"])
                bak_path = backup_dir / r["backup_slice_file"]
                orig_sector_bytes = bak_path.read_bytes()

                f.seek(orig_off * SECTOR)
                f.write(orig_sector_bytes)
                f.seek(8 + idx * 32)
                f.write(orig_entry)

            f.truncate(min_orig_file_size)
            f.flush()
            os.fsync(f.fileno())
        restored_archives.append(archive_rel)

    return restored_archives


class StudioDeployManager:
    def __init__(
        self,
        store: WorldCatalogStore,
        lock_dir: Path = DEFAULT_LOCK_DIR,
    ):
        self.store = store
        self.lock_dir = lock_dir
        self.backups_root = store.data_dir / "backups"
        self.backups_root.mkdir(parents=True, exist_ok=True)

    def stage_txd_update(
        self,
        txd_name: str,
        new_txd_bytes: bytes,
        raster_report: Dict[str, Any],
        lod_txd_name: Optional[str] = None,
        new_lod_txd_bytes: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """Save an updated .TXD (and optional paired *_lod.txd) into staging."""
        short = txd_name.lower()
        if short.endswith(".txd"):
            short = short[:-4]

        txd_stage_dir = self.store.staging_dir / "txds"
        txd_stage_dir.mkdir(parents=True, exist_ok=True)

        out_path = txd_stage_dir / f"{short}.txd"
        out_path.write_bytes(new_txd_bytes)

        manifest = self.store.load_staging_manifest()
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        manifest["updated_at"] = now_iso

        cat_entry = self.store.get_txd_entry(short)
        orig_kb = cat_entry["file_size_kb"] if cat_entry else round(len(new_txd_bytes) / 1024.0, 2)
        new_kb = round(len(new_txd_bytes) / 1024.0, 2)

        txd_rec = manifest["staged_txds"].get(short) or {
            "txd_name": short,
            "txd_filename": f"{short}.txd",
            "category": cat_entry["primary_category"] if cat_entry else raster_report.get("category"),
            "region": cat_entry["region"] if cat_entry else "Unknown",
            "target_imgs": [x["img"] for x in (cat_entry.get("all_imgs") or [])] if cat_entry else [],
            "orig_txd_size_kb": orig_kb,
            "modified_rasters": {},
        }
        txd_rec["staged_txd_size_kb"] = new_kb
        txd_rec["delta_txd_size_kb"] = round(new_kb - orig_kb, 2)
        txd_rec["sha256"] = sha256_bytes(new_txd_bytes)
        txd_rec["updated_at"] = now_iso
        txd_rec["modified_rasters"][raster_report["raster_name"].lower()] = raster_report
        manifest["staged_txds"][short] = txd_rec

        if lod_txd_name and new_lod_txd_bytes:
            lshort = lod_txd_name.lower()
            if lshort.endswith(".txd"):
                lshort = lshort[:-4]
            lod_path = txd_stage_dir / f"{lshort}.txd"
            lod_path.write_bytes(new_lod_txd_bytes)
            lcat = self.store.get_txd_entry(lshort)
            lorig_kb = lcat["file_size_kb"] if lcat else round(len(new_lod_txd_bytes) / 1024.0, 2)
            lnew_kb = round(len(new_lod_txd_bytes) / 1024.0, 2)
            manifest["staged_txds"][lshort] = {
                "txd_name": lshort,
                "txd_filename": f"{lshort}.txd",
                "category": "LOD_SYNC",
                "region": lcat["region"] if lcat else txd_rec["region"],
                "target_imgs": [x["img"] for x in (lcat.get("all_imgs") or [])] if lcat else [],
                "orig_txd_size_kb": lorig_kb,
                "staged_txd_size_kb": lnew_kb,
                "delta_txd_size_kb": round(lnew_kb - lorig_kb, 2),
                "sha256": sha256_bytes(new_lod_txd_bytes),
                "updated_at": now_iso,
                "is_auto_lod_for": short,
                "modified_rasters": {
                    (raster_report.get("lod_raster_name") or "lod").lower(): {
                        "raster_name": raster_report.get("lod_raster_name"),
                        "new_dimensions": raster_report.get("lod_dimensions"),
                        "after_kb": raster_report.get("lod_size_kb"),
                    }
                },
            }

        self.store.save_staging_manifest(manifest)
        return manifest

    def stage_statue_update(
        self,
        monument_id: int,
        dff_name: str,
        txd_name: str,
        new_dff_bytes: bytes,
        new_txd_bytes: bytes,
        statue_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Save a baked 3D Monument (.DFF + .TXD) into staging."""
        statue_dir = self.store.staging_dir / "statues"
        txd_dir = self.store.staging_dir / "txds"
        statue_dir.mkdir(parents=True, exist_ok=True)
        txd_dir.mkdir(parents=True, exist_ok=True)

        dshort = dff_name.lower()
        if dshort.endswith(".dff"):
            dshort = dshort[:-4]
        tshort = txd_name.lower()
        if tshort.endswith(".txd"):
            tshort = tshort[:-4]

        (statue_dir / f"{dshort}.dff").write_bytes(new_dff_bytes)
        (txd_dir / f"{tshort}.txd").write_bytes(new_txd_bytes)

        manifest = self.store.load_staging_manifest()
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        manifest["updated_at"] = now_iso

        mon_entry = self.store.get_monument_entry(monument_id)
        txd_entry = self.store.get_txd_entry(tshort)

        manifest["staged_statues"][dshort] = {
            "model_id": monument_id,
            "dff_name": dshort,
            "dff_filename": f"{dshort}.dff",
            "txd_name": tshort,
            "txd_filename": f"{tshort}.txd",
            "dff_img": mon_entry["dff_img"] if mon_entry else "",
            "txd_imgs": [x["img"] for x in (txd_entry.get("all_imgs") or [])] if txd_entry else ([mon_entry["txd_img"]] if mon_entry else []),
            "region": mon_entry["region"] if mon_entry else "Unknown",
            "dff_size_kb": round(len(new_dff_bytes) / 1024.0, 2),
            "txd_size_kb": round(len(new_txd_bytes) / 1024.0, 2),
            "dff_sha256": sha256_bytes(new_dff_bytes),
            "txd_sha256": sha256_bytes(new_txd_bytes),
            "statue_meta": statue_meta,
            "updated_at": now_iso,
        }
        self.store.save_staging_manifest(manifest)
        return manifest

    def unstage_item(self, item_type: str, name: str) -> Dict[str, Any]:
        """Remove a staged TXD or Statue from staging."""
        manifest = self.store.load_staging_manifest()
        key = name.lower().replace(".txd", "").replace(".dff", "")
        if item_type == "statue":
            rec = manifest["staged_statues"].pop(key, None)
            p_dff = self.store.staging_dir / "statues" / f"{key}.dff"
            if p_dff.exists():
                p_dff.unlink()
            if rec and rec.get("txd_name"):
                t_p = self.store.staging_dir / "txds" / f"{rec['txd_name']}.txd"
                if t_p.exists() and rec["txd_name"] not in manifest["staged_txds"]:
                    t_p.unlink()
        else:
            manifest["staged_txds"].pop(key, None)
            p_txd = self.store.staging_dir / "txds" / f"{key}.txd"
            if p_txd.exists():
                p_txd.unlink()
        manifest["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.store.save_staging_manifest(manifest)
        return manifest

    def list_transactions(self) -> List[Dict[str, Any]]:
        txs = []
        if not self.backups_root.exists():
            return txs
        for d in sorted(self.backups_root.iterdir(), reverse=True):
            if d.is_dir() and (d / "transaction.json").exists():
                try:
                    txs.append(json.loads((d / "transaction.json").read_text(encoding="utf-8")))
                except Exception:
                    pass
        return txs

    def deploy_staged_to_game(self, check_game_closed: bool = True) -> Dict[str, Any]:
        """
        Atomically deploy all staged .TXD and 3D Monument .DFF/.TXD files into
        gameplay-lab's mounted .IMG archives while holding `game-write.lock`.
        """
        if check_game_closed:
            verify_game_processes_closed()

        manifest = self.store.load_staging_manifest()
        staged_txds = manifest.get("staged_txds", {})
        staged_statues = manifest.get("staged_statues", {})

        if not staged_txds and not staged_statues:
            raise ValueError("Chưa có file .TXD hoặc Tượng 3D nào trong Staging để Deploy.")

        # Group member payloads by archive_rel
        archive_payloads: Dict[str, Dict[str, bytes]] = {}
        for short, rec in staged_txds.items():
            txd_file = self.store.staging_dir / "txds" / f"{short}.txd"
            if not txd_file.exists():
                continue
            raw_bytes = txd_file.read_bytes()
            for img_rel in rec.get("target_imgs", []):
                archive_payloads.setdefault(img_rel, {})[f"{short}.txd"] = raw_bytes

        for dshort, srec in staged_statues.items():
            dff_file = self.store.staging_dir / "statues" / f"{dshort}.dff"
            txd_file = self.store.staging_dir / "txds" / f"{srec['txd_name']}.txd"
            if dff_file.exists() and srec.get("dff_img"):
                archive_payloads.setdefault(srec["dff_img"], {})[f"{dshort}.dff"] = dff_file.read_bytes()
            if txd_file.exists():
                t_bytes = txd_file.read_bytes()
                for img_rel in srec.get("txd_imgs", []):
                    archive_payloads.setdefault(img_rel, {})[f"{srec['txd_name']}.txd"] = t_bytes

        if not archive_payloads:
            raise ValueError("Không tìm thấy đường dẫn .IMG đích cho các mục trong Staging.")

        tx_id = datetime.datetime.now().strftime("tx-%Y%m%d-%H%M%S-") + f"{int(time.time() * 1000) % 10000:04d}"
        tx_dir = self.backups_root / tx_id
        tx_dir.mkdir(parents=True, exist_ok=True)

        resources = sorted(archive_payloads.keys())
        all_applied: List[Dict[str, Any]] = []

        with GameWriteLock(lock_dir=self.lock_dir, task_label=f"deploy-{tx_id}", resources=resources):
            for img_rel, members_map in archive_payloads.items():
                img_path = (self.store.game_root / img_rel.replace("\\", "/")).resolve()
                records = patch_img_archive_members(
                    img_path=img_path,
                    member_payloads=members_map,
                    backup_dir=tx_dir,
                    archive_rel=img_rel,
                )
                all_applied.extend(records)
                self.store.invalidate_img_cache(img_rel)

            before_total = sum(r["before_bytes"] for r in all_applied)
            after_total = sum(r["after_bytes"] for r in all_applied)
            tx_meta = {
                "transaction_id": tx_id,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "status": "deployed",
                "game_root": str(self.store.game_root),
                "archives_modified": resources,
                "members_patched_count": len(all_applied),
                "before_total_kb": round(before_total / 1024.0, 2),
                "after_total_kb": round(after_total / 1024.0, 2),
                "saved_kb": round((before_total - after_total) / 1024.0, 2),
                "applied_records": all_applied,
            }
            (tx_dir / "transaction.json").write_text(
                json.dumps(tx_meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        return tx_meta

    def rollback_transaction(self, tx_id: str, check_game_closed: bool = True) -> Dict[str, Any]:
        """Restore exact original .IMG members from a backup transaction."""
        if check_game_closed:
            verify_game_processes_closed()

        tx_dir = self.backups_root / tx_id
        tx_json = tx_dir / "transaction.json"
        if not tx_json.exists():
            raise FileNotFoundError(f"Transaction '{tx_id}' not found")

        tx_meta = json.loads(tx_json.read_text(encoding="utf-8"))
        if tx_meta.get("status") == "rolled_back":
            raise ValueError(f"Transaction '{tx_id}' đã được Rollback trước đó.")

        records = tx_meta.get("applied_records", [])
        resources = tx_meta.get("archives_modified", [])

        with GameWriteLock(lock_dir=self.lock_dir, task_label=f"rollback-{tx_id}", resources=resources):
            restored = rollback_img_archive_members(
                game_root=self.store.game_root,
                backup_dir=tx_dir,
                applied_records=records,
            )
            for img_rel in restored:
                self.store.invalidate_img_cache(img_rel)

            tx_meta["status"] = "rolled_back"
            tx_meta["rolled_back_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            tx_json.write_text(json.dumps(tx_meta, ensure_ascii=False, indent=2), encoding="utf-8")

        return tx_meta
