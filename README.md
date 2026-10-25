# 🎨 Eagle World Texture Studio

**Billboard, mural, monument & neon converter for GTA San Andreas world textures.**

A web-based tool for browsing, converting, and deploying custom textures and 3D monuments into GTA SA's world — built with FastAPI and Three.js.

## Features

### Texture Browser
- Browse & filter **1,635 world TXD dictionaries** and **93 3D monuments**
- Live RenderWare TXD raster decoding: **DXT1 / DXT3 / DXT5 / RGBA / PAL8 → PNG**

### 4 Category-Specific 2D Surface Converters
| Category | Type | Features |
|----------|------|----------|
| CAT1 | Billboards | Dual-sync LOD generation |
| CAT2 | Murals/Graffiti/Banners | Wall-grain blend & alpha DXT5 |
| CAT3 | Neon/LED Screens | Sub-region atlas patching & scanlines |
| CAT4 | Posters/Bus Stops/Vending | UV-unstretch & frame masking |

### 3D Monument Swapper & Baker
- Convert character models into **static prelit GTA SA monuments**
- Optional stone pedestal with **marble / bronze / gold / jade** finishes
- DFF geometry → prelit vertex colors baking

### Deployment
- Staged mod pack management
- Atomic IMG deploy with `game-write.lock` synchronization
- Bit-for-bit rollback support

## Architecture

```
eagle-world-texture-studio/
├── app.py                     # FastAPI backend (24 KB)
├── Start_World_Texture_Studio.bat
├── core/
│   ├── catalog_store.py       # World catalog management
│   ├── converters.py          # 4 category-specific surface converters
│   ├── deployer.py            # IMG deploy + rollback
│   ├── rw_dff.py              # RenderWare DFF 3D analysis + statue builder
│   └── rw_txd.py              # RenderWare TXD raster decode
├── static/
│   ├── index.html, app.js, style.css
└── tests/
```

## Requirements

- **Python 3.10+**
- **FastAPI** + **Pydantic** — `pip install fastapi uvicorn pydantic`
- **Pillow** — `pip install Pillow`
- **GTA San Andreas** game installation

## Usage

```bash
# Option 1: Use the launcher
Start_World_Texture_Studio.bat

# Option 2: Run directly
python app.py
```

Opens at **http://127.0.0.1:8950**

## License

MIT License — see [LICENSE](LICENSE).

## Credits

- Built as part of **Project Eagle** by [@nguyenlee97](https://github.com/nguyenlee97)
- RenderWare TXD/DFF format references from the GTA SA modding community
- Three.js for 3D monument preview
