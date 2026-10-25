import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { DFFLoader } from '/vendor/dff-loader/DFFLoader.js';
import { TXDLoader } from '/vendor/dff-loader/TXDLoader.js';

// ---------------------------------------------------------------------------
// Reusable Three.js RenderWare .DFF + .TXD 3D Viewport
// ---------------------------------------------------------------------------
class RenderWare3DViewport {
  constructor(containerEl, statsEl) {
    this.container = containerEl;
    this.statsEl = statsEl;
    this.wireframe = false;
    this.autoRotate = false;
    this.modelGroup = null;

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    this.renderer.setClearColor(0x0f1622);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.05, 2500);
    this.camera.position.set(0, 4, 12);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.autoRotateSpeed = 1.6;

    const hemi = new THREE.HemisphereLight(0xe8f4ff, 0x384558, 2.3);
    this.scene.add(hemi);
    const sun = new THREE.DirectionalLight(0xffffff, 2.0);
    sun.position.set(15, 25, 20);
    this.scene.add(sun);
    const fill = new THREE.DirectionalLight(0x9ec5ff, 1.1);
    fill.position.set(-15, 10, -15);
    this.scene.add(fill);

    this.grid = new THREE.GridHelper(40, 40, 0x3b5278, 0x1d293d);
    this.scene.add(this.grid);

    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(this.container);
    this.resize();

    this.renderer.setAnimationLoop(() => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    });
  }

  resize() {
    const w = Math.max(10, this.container.clientWidth);
    const h = Math.max(10, this.container.clientHeight);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  clearModel() {
    if (!this.modelGroup) return;
    this.scene.remove(this.modelGroup);
    this.modelGroup.traverse((obj) => {
      if (obj.geometry) obj.geometry.dispose();
      const mats = [].concat(obj.material || []);
      for (const m of mats) {
        if (m?.map) m.map.dispose();
        if (m?.alphaMap) m.alphaMap.dispose();
        m?.dispose?.();
      }
    });
    this.modelGroup = null;
  }

  async loadDffAndTxd(dffUrl, txdUrl, label = '') {
    try {
      if (this.statsEl) this.statsEl.textContent = `Đang tải mô hình 3D ${label}...`;
      const [dffRes, txdRes] = await Promise.all([
        fetch(dffUrl),
        txdUrl ? fetch(txdUrl) : Promise.resolve(null),
      ]);
      if (!dffRes.ok) throw new Error(`Không tải được .DFF (${dffRes.status})`);
      const dffBuf = await dffRes.arrayBuffer();
      const txdBuf = txdRes && txdRes.ok ? await txdRes.arrayBuffer() : null;

      const txdLoader = new TXDLoader();
      const textures = txdBuf ? txdLoader.parse(txdBuf) : new Map();
      const dffLoader = new DFFLoader();
      dffLoader.setTextureDictionary(textures);
      const parsedRoot = dffLoader.parse(dffBuf);

      this.clearModel();

      // GTA SA world DFFs use Z-up; rotate -90 deg around X so Z becomes Three.js Y-up
      const holder = new THREE.Group();
      parsedRoot.rotation.x = -Math.PI / 2;
      holder.add(parsedRoot);
      holder.updateMatrixWorld(true);

      let verts = 0;
      let tris = 0;
      holder.traverse((mesh) => {
        if (!mesh.isMesh) return;
        const g = mesh.geometry;
        if (g && !g.attributes.normal) g.computeVertexNormals();
        if (g?.attributes?.position) {
          verts += g.attributes.position.count;
          tris += (g.index ? g.index.count : g.attributes.position.count) / 3;
        }
        const mats = [].concat(mesh.material || []);
        for (const m of mats) {
          if (!m) continue;
          m.side = THREE.DoubleSide;
          m.wireframe = this.wireframe;
          if (m.map) {
            m.map.colorSpace = THREE.SRGBColorSpace;
            m.map.needsUpdate = true;
          }
        }
      });

      // Center X/Z at origin and rest base on Y=0 grid
      const box = new THREE.Box3().setFromObject(holder);
      const size = new THREE.Vector3();
      const center = new THREE.Vector3();
      box.getSize(size);
      box.getCenter(center);

      holder.position.x -= center.x;
      holder.position.z -= center.z;
      holder.position.y -= box.min.y;

      this.modelGroup = holder;
      this.scene.add(holder);

      this.fitCamera(size);
      if (this.statsEl) {
        this.statsEl.textContent =
          `🧊 ${label} | Kích thước: ${size.x.toFixed(1)}m × ${size.z.toFixed(1)}m × Cao ${size.y.toFixed(1)}m | ` +
          `${verts.toLocaleString()} đỉnh · ${Math.round(tris).toLocaleString()} tam giác · ${textures.size} textures`;
      }
    } catch (err) {
      if (this.statsEl) {
        this.statsEl.textContent = `⚠️ Không thể dựng 3D: ${err.message}`;
      }
    }
  }

  fitCamera(sizeVec) {
    const maxDim = Math.max(1.5, sizeVec.x, sizeVec.y, sizeVec.z);
    const dist = maxDim * 1.65;
    this.camera.position.set(dist * 0.7, Math.max(1.2, sizeVec.y * 0.65), dist);
    this.controls.target.set(0, Math.max(0.5, sizeVec.y * 0.45), 0);
    this.controls.minDistance = 0.2;
    this.controls.maxDistance = Math.max(150, maxDim * 8);
    this.controls.update();
  }

  toggleWireframe() {
    this.wireframe = !this.wireframe;
    this.modelGroup?.traverse((o) => {
      for (const m of [].concat(o.material || [])) {
        if (m) m.wireframe = this.wireframe;
      }
    });
    return this.wireframe;
  }

  toggleAutoRotate() {
    this.autoRotate = !this.autoRotate;
    this.controls.autoRotate = this.autoRotate;
    return this.autoRotate;
  }
}

// ---------------------------------------------------------------------------
// Application State
// ---------------------------------------------------------------------------
const state = {
  summary: {},
  converterSchemas: {},
  staging: { staged_txds: {}, staged_statues: {} },
  transactions: [],

  // Tab 1: 2D Surface Studio
  selectedCategory: 'ALL',
  catalogItems: [],
  activeTxdName: null,
  activeTxdData: null,
  activeRaster: null,
  surfaceViewMode: '2d',
  surface3DStaged: true,

  // Tab 2: 3D Monuments
  monuments: [],
  activeMonument: null,
  activeMonumentDetails: null,
  monument3DStaged: true,
  skinLibraryItems: [],
};

let surfaceViewport = null;
let monumentViewport = null;

function showToast(msg, isError = false) {
  const container = document.getElementById('toast-container');
  const el = document.createElement('div');
  el.className = 'toast';
  if (isError) el.style.borderLeftColor = '#ff5b6b';
  el.textContent = msg;
  container.appendChild(el);
  setTimeout(() => el.remove(), 4200);
}

const CAT_SHORT_LABELS = {
  CAT1_BILLBOARDS: '🪧 CAT1 · Billboard',
  CAT2_MURALS_BANNERS_GRAFFITI: '🎨 CAT2 · Mural/Tag/Banner',
  CAT3_NEON_LED_SCREENS_CASINOS: '✨ CAT3 · Neon/LED/Casino',
  CAT4_POSTERS_SHOPS_TRANSIT_VENDING: '🏪 CAT4 · Poster/Bus/Vending',
  CAT5_MONUMENTS_STATUES_LANDMARKS: '🗽 CAT5 · Tượng Đài',
};

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------
async function initApp() {
  surfaceViewport = new RenderWare3DViewport(
    document.getElementById('surface-webgl-stage'),
    document.getElementById('surface-3d-stats')
  );
  monumentViewport = new RenderWare3DViewport(
    document.getElementById('monument-webgl-stage'),
    document.getElementById('monument-3d-stats')
  );

  bindUIEvents();
  await refreshGlobalState();
  await loadCatalogList();
  await loadMonumentsList();
  await searchSkinLibrary('');
}

async function refreshGlobalState() {
  const res = await fetch('/api/state');
  const data = await res.json();
  state.summary = data.summary || {};
  state.converterSchemas = data.converter_schemas || {};
  state.staging = data.staging || { staged_txds: {}, staged_statues: {} };
  state.transactions = data.transactions || [];

  // Populate region dropdowns if first time
  const regSelect = document.getElementById('filter-region');
  const monRegSelect = document.getElementById('mon-filter-region');
  if (regSelect.options.length <= 1 && state.summary.by_region) {
    for (const [reg, cnt] of Object.entries(state.summary.by_region)) {
      const opt1 = document.createElement('option');
      opt1.value = reg;
      opt1.textContent = `${reg} (${cnt})`;
      regSelect.appendChild(opt1);

      const opt2 = document.createElement('option');
      opt2.value = reg;
      opt2.textContent = reg;
      monRegSelect.appendChild(opt2);
    }
  }

  updateTopBarCounters();
  updateConverterSchemaUI(document.getElementById('conv-category-select').value);
  renderDeployTab();
}

function updateTopBarCounters() {
  const sTxds = Object.values(state.staging.staged_txds || {});
  const sStats = Object.values(state.staging.staged_statues || {});
  const totalStaged = sTxds.length + sStats.length;

  document.getElementById('nav-txd-count').textContent = `${(state.summary.near_txds || 0).toLocaleString()} TXD`;
  document.getElementById('nav-mon-count').textContent = `${state.summary.total_monument_3d_models || 93} Tượng`;
  document.getElementById('nav-staged-count').textContent = `${totalStaged} Staged`;

  let deltaKb = 0;
  for (const t of sTxds) {
    deltaKb += Number(t.delta_txd_size_kb || 0);
  }
  const savedKb = -deltaKb;
  const savedEl = document.getElementById('global-saved-kb');
  if (savedKb >= 0) {
    savedEl.textContent = `${savedKb.toFixed(1)} KB (${(savedKb / 1024).toFixed(2)} MB)`;
  } else {
    savedEl.textContent = `+${Math.abs(savedKb).toFixed(1)} KB`;
  }
}

// ---------------------------------------------------------------------------
// Tab 1: Catalog Loading & TXD Inspection
// ---------------------------------------------------------------------------
async function loadCatalogList() {
  const cat = state.selectedCategory;
  const region = document.getElementById('filter-region').value;
  const tier = document.getElementById('filter-tier').value;
  const sort = document.getElementById('filter-sort').value;
  const includeLod = document.getElementById('filter-include-lod').checked;
  const q = document.getElementById('filter-search').value.trim();

  const params = new URLSearchParams({
    category: cat,
    region,
    tier,
    sort,
    include_lod: String(includeLod),
    q,
    limit: '150',
  });
  const res = await fetch(`/api/catalog?${params}`);
  const data = await res.json();
  state.catalogItems = data.items || [];

  document.getElementById('catalog-result-count').textContent = `${data.total.toLocaleString()} TXD`;
  renderTxdList();

  if (state.catalogItems.length > 0 && (!state.activeTxdName || !state.catalogItems.some((x) => x.txd_name === state.activeTxdName))) {
    selectTxd(state.catalogItems[0].txd_name);
  }
}

function renderTxdList() {
  const listEl = document.getElementById('txd-list');
  listEl.innerHTML = '';

  for (const item of state.catalogItems) {
    const div = document.createElement('div');
    div.className = `txd-item ${item.txd_name === state.activeTxdName ? 'active' : ''}`;
    div.onclick = () => selectTxd(item.txd_name);

    const infBadge =
      item.inflation_ratio > 1.5
        ? `<span class="badge inflated" title="Đang phình gấp ${item.inflation_ratio} lần so với bản S&S gốc">⚠️ Phình ${item.inflation_ratio}×</span>`
        : '';
    const stagedBadge = item.is_staged ? `<span class="badge staged">✅ STAGED</span>` : '';
    const lodBadge = item.paired_lod_txd ? `<span class="badge lod">🔗 LOD: ${item.paired_lod_txd}</span>` : '';
    const nightBadge = item.has_night_tobj ? `<span class="badge">🌙 Đèn đêm</span>` : '';

    div.innerHTML = `
      <div class="txd-item-top">
        <span class="txd-name">${item.txd_name}.txd</span>
        <span class="txd-size">${item.file_size_kb.toLocaleString()} KB · ${item.raster_count} ảnh</span>
      </div>
      <div class="txd-badges">
        <span class="badge cat">${CAT_SHORT_LABELS[item.primary_category] || item.primary_category}</span>
        <span class="badge">${item.region}</span>
        ${stagedBadge}
        ${infBadge}
        ${lodBadge}
        ${nightBadge}
      </div>
    `;
    listEl.appendChild(div);
  }
}

async function selectTxd(txdName) {
  state.activeTxdName = txdName;
  renderTxdList();

  const res = await fetch(`/api/txd/${encodeURIComponent(txdName)}`);
  if (!res.ok) {
    showToast(`Không đọc được TXD ${txdName}`, true);
    return;
  }
  const data = await res.json();
  state.activeTxdData = data;

  const entry = data.entry;
  document.getElementById('active-txd-title').textContent =
    `📦 ${entry.txd_filename} — ${entry.region} (${data.active_file_size_kb} KB)`;

  const badgesEl = document.getElementById('active-txd-badges');
  const imgsStr = entry.all_imgs.map((x) => `${x.img} (${x.size_kb} KB)`).join(' | ');
  badgesEl.innerHTML = `
    <span class="badge cat">${CAT_SHORT_LABELS[entry.primary_category] || entry.primary_category}</span>
    <span class="badge" title="${imgsStr}">🗄️ ${entry.all_imgs.length} kho .IMG: ${entry.primary_img}</span>
    <span class="badge">📍 ${entry.total_placements} vị trí đặt trên Map</span>
    ${entry.paired_lod_txd ? `<span class="badge lod">🔗 Cặp LOD: ${entry.paired_lod_txd}.txd</span>` : ''}
    ${data.is_staged ? `<span class="badge staged">✅ ĐANG DÙNG BẢN STAGED</span>` : ''}
  `;

  // Auto-switch converter category to match this TXD's primary category (if CAT1..CAT4)
  if (entry.primary_category && entry.primary_category !== 'CAT5_MONUMENTS_STATUES_LANDMARKS') {
    const convSel = document.getElementById('conv-category-select');
    convSel.value = entry.primary_category;
    updateConverterSchemaUI(entry.primary_category);
  }

  // Populate 3D preview DFF dropdown
  const dffSelect = document.getElementById('select-preview-dff');
  dffSelect.innerHTML = '';
  const models = data.analyzed_dffs?.length ? data.analyzed_dffs : entry.models || [];
  for (const m of models) {
    const opt = document.createElement('option');
    opt.value = m.dff;
    opt.dataset.img = m.dff_img || '';
    opt.textContent = `${m.dff}.dff (${m.placements || 0} chỗ đặt${m.dimensions_m ? ` · ${m.dimensions_m.join('×')}m` : ''})`;
    dffSelect.appendChild(opt);
  }

  renderRasterGrid();

  if (data.rasters && data.rasters.length > 0) {
    selectRaster(data.rasters[0]);
  }

  if (state.surfaceViewMode === '3d') {
    loadCurrentSurface3D();
  }
}

function renderRasterGrid() {
  const gridEl = document.getElementById('raster-grid');
  gridEl.innerHTML = '';
  if (!state.activeTxdData) return;

  const q = document.getElementById('raster-filter-input').value.trim().toLowerCase();
  const txdName = state.activeTxdData.entry.txd_name;
  const cacheBust = Date.now();

  for (const r of state.activeTxdData.rasters) {
    if (q && !r.name.toLowerCase().includes(q)) continue;
    const card = document.createElement('div');
    card.className = `raster-card ${state.activeRaster?.name === r.name ? 'active' : ''}`;
    card.onclick = () => selectRaster(r);

    const uvInfo = r.dff_uv_info;
    const uvTag = uvInfo
      ? `<div class="raster-uv-tag">📐 Khung 3D: ${uvInfo.effective_tex_aspect_label} (${uvInfo.phys_width_m}×${uvInfo.phys_height_m}m)</div>`
      : '';
    const modBadge = r.is_modified ? `<span class="badge staged">ĐÃ SỬA</span>` : '';

    const pngUrl = `/api/txd/${encodeURIComponent(txdName)}/raster/${encodeURIComponent(r.name)}/png?source=active&t=${cacheBust}`;

    card.innerHTML = `
      <div class="raster-thumb-wrap checkerboard">
        <img src="${pngUrl}" loading="lazy" alt="${r.name}" />
      </div>
      <div class="raster-title" title="${r.name}">${r.name} ${modBadge}</div>
      <div class="raster-meta">
        <span>${r.width}×${r.height} (${r.aspect})</span>
        <span>${r.format} · ${r.size_kb}KB</span>
      </div>
      ${uvTag}
    `;
    gridEl.appendChild(card);
  }
}

function selectRaster(rasterObj) {
  state.activeRaster = rasterObj;
  renderRasterGrid();

  const txdName = state.activeTxdData.entry.txd_name;
  document.getElementById('diag-raster-name').textContent = rasterObj.name;
  document.getElementById('diag-stored-dim').textContent = `${rasterObj.width}×${rasterObj.height} (${rasterObj.aspect})`;
  document.getElementById('diag-orig-fmt').textContent = `${rasterObj.format} (${rasterObj.size_kb} KB, ${rasterObj.mipmaps} mip)`;

  const uvInfo = rasterObj.dff_uv_info;
  if (uvInfo) {
    document.getElementById('diag-3d-aspect').textContent =
      `${uvInfo.effective_tex_aspect_label} (Khung ${uvInfo.phys_width_m}m × ${uvInfo.phys_height_m}m trên ${uvInfo.source_dff}.dff)`;
  } else {
    document.getElementById('diag-3d-aspect').textContent = `Theo tỷ lệ ảnh gốc (${rasterObj.aspect})`;
  }

  if (rasterObj.matched_lod_raster) {
    const ml = rasterObj.matched_lod_raster;
    document.getElementById('diag-lod-info').textContent =
      `${ml.lod_txd}.txd ➜ ${ml.name} (${ml.width}×${ml.height} ${ml.format})`;
  } else if (state.activeTxdData.entry.paired_lod_txd) {
    document.getElementById('diag-lod-info').textContent =
      `File ${state.activeTxdData.entry.paired_lod_txd}.txd (Không trùng tên raster trực tiếp)`;
  } else {
    document.getElementById('diag-lod-info').textContent = 'Không có LOD riêng';
  }

  // Populate UV Island pills for CAT3 Sub-Region converter if UV boxes detected
  const pillsEl = document.getElementById('cat3-uv-island-pills');
  pillsEl.innerHTML = '';
  if (uvInfo && uvInfo.uv_boxes && uvInfo.uv_boxes.length > 0) {
    uvInfo.uv_boxes.forEach((box, idx) => {
      const btn = document.createElement('button');
      btn.className = 'cat-pill';
      btn.textContent = `Ô UV #${idx + 1} [${box[0]},${box[1]}→${box[2]},${box[3]}]`;
      btn.onclick = () => {
        document.getElementById('opt-cat3-u0').value = box[0];
        document.getElementById('opt-cat3-v0').value = box[1];
        document.getElementById('opt-cat3-u1').value = box[2];
        document.getElementById('opt-cat3-v1').value = box[3];
      };
      pillsEl.appendChild(btn);
    });
  }

  // Show Before image in comparison deck
  const deck = document.getElementById('comparison-deck');
  deck.classList.remove('hidden');
  document.getElementById('comp-before-img').src =
    `/api/txd/${encodeURIComponent(txdName)}/raster/${encodeURIComponent(rasterObj.name)}/png?source=original&t=${Date.now()}`;
  document.getElementById('comp-before-meta').textContent =
    `${rasterObj.width}×${rasterObj.height} · ${rasterObj.format} · ${rasterObj.size_kb} KB`;

  // If already modified in staging, show active image in After slot
  if (rasterObj.is_modified) {
    document.getElementById('comp-after-img').src =
      `/api/txd/${encodeURIComponent(txdName)}/raster/${encodeURIComponent(rasterObj.name)}/png?source=active&t=${Date.now()}`;
    const rep = rasterObj.modified_report || {};
    document.getElementById('comp-after-meta').textContent =
      `${rep.new_dimensions || ''} · ${rep.new_format || ''} · ${rep.after_kb || ''} KB (Đã Stage)`;
  } else {
    document.getElementById('comp-after-img').src =
      `/api/txd/${encodeURIComponent(txdName)}/raster/${encodeURIComponent(rasterObj.name)}/png?source=active&t=${Date.now()}`;
    document.getElementById('comp-after-meta').textContent = 'Bấm "Xem Trước" hoặc "Convert & Lưu"';
  }
  document.getElementById('comp-lod-card').classList.add('hidden');
}

function updateConverterSchemaUI(catKey) {
  const schema = state.converterSchemas[catKey];
  if (schema) {
    document.getElementById('conv-schema-desc').textContent = schema.description;
  }
  document.querySelectorAll('.cat-specific-group').forEach((el) => el.classList.add('hidden'));
  const targetGroup = document.getElementById(`opts-${catKey}`);
  if (targetGroup) targetGroup.classList.remove('hidden');
}

function collectConverterOptions(catKey) {
  const opts = {
    resolution_tier: document.getElementById('conv-res-tier').value,
    fit_mode: document.getElementById('conv-fit-mode').value,
    uv_aspect_compensation: document.getElementById('conv-uv-comp').checked,
    generate_mipmaps: true,
  };

  if (catKey === 'CAT1_BILLBOARDS') {
    opts.auto_sync_lod = document.getElementById('opt-cat1-sync-lod').checked;
    opts.print_weathering = parseFloat(document.getElementById('opt-cat1-weather').value);
    opts.border_frame_px = parseInt(document.getElementById('opt-cat1-border').value, 10);
  } else if (catKey === 'CAT2_MURALS_BANNERS_GRAFFITI') {
    opts.sub_mode = document.getElementById('opt-cat2-submode').value;
    opts.wall_blend_strength = parseFloat(document.getElementById('opt-cat2-wall').value);
    opts.fabric_shade_strength = parseFloat(document.getElementById('opt-cat2-fabric').value);
    opts.auto_key_bg = document.getElementById('opt-cat2-autokey').checked;
  } else if (catKey === 'CAT3_NEON_LED_SCREENS_CASINOS') {
    opts.led_scanline_strength = parseFloat(document.getElementById('opt-cat3-scanline').value);
    opts.vibrance_boost = parseFloat(document.getElementById('opt-cat3-vibrance').value);
    opts.use_sub_region = document.getElementById('opt-cat3-usesub').checked;
    opts.sub_region_rect = [
      parseFloat(document.getElementById('opt-cat3-u0').value),
      parseFloat(document.getElementById('opt-cat3-v0').value),
      parseFloat(document.getElementById('opt-cat3-u1').value),
      parseFloat(document.getElementById('opt-cat3-v1').value),
    ];
  } else if (catKey === 'CAT4_POSTERS_SHOPS_TRANSIT_VENDING') {
    opts.preserve_border_pct = parseFloat(document.getElementById('opt-cat4-border').value);
    opts.glass_sheen_strength = parseFloat(document.getElementById('opt-cat4-glass').value);
  }
  return opts;
}

function readFileAsDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ''));
    reader.onerror = () => reject(new Error('Không đọc được file'));
    reader.readAsDataURL(file);
  });
}

async function runSurfaceConversion(commitToStaging = false) {
  if (!state.activeTxdData || !state.activeRaster) {
    showToast('Hãy chọn một Raster trong TXD trước!', true);
    return;
  }

  const txdName = state.activeTxdData.entry.txd_name;
  const rasterName = state.activeRaster.name;
  const catKey = document.getElementById('conv-category-select').value;
  const opts = collectConverterOptions(catKey);

  const fileInput = document.getElementById('conv-file-input');
  let imageB64 = null;
  if (fileInput.files && fileInput.files[0]) {
    imageB64 = await readFileAsDataURL(fileInput.files[0]);
  }

  const payload = {
    txd_name: txdName,
    raster_name: rasterName,
    category: catKey,
    options: opts,
    image_b64: imageB64,
    sample_theme: document.getElementById('conv-sample-theme').value,
    sample_title: document.getElementById('conv-sample-title').value,
  };

  const endpoint = commitToStaging ? '/api/convert/stage' : '/api/convert/preview';
  const res = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Lỗi convert' }));
    showToast(err.detail || 'Lỗi khi chạy Converter', true);
    return;
  }

  const data = await res.json();
  const rep = data.report;

  document.getElementById('comp-before-img').src = data.before_png;
  document.getElementById('comp-after-img').src = data.after_png;
  document.getElementById('comp-after-meta').textContent =
    `${rep.new_dimensions} · ${rep.new_format} (${rep.new_mipmaps} mip) · ${rep.after_kb} KB`;

  const lodCard = document.getElementById('comp-lod-card');
  if (data.lod_after_png) {
    lodCard.classList.remove('hidden');
    document.getElementById('comp-lod-img').src = data.lod_after_png;
    document.getElementById('comp-lod-meta').textContent =
      `${rep.lod_raster_name} (${rep.lod_dimensions} · ${rep.lod_size_kb} KB)`;
  } else {
    lodCard.classList.add('hidden');
  }

  const repBox = document.getElementById('conv-report-box');
  repBox.classList.remove('hidden');
  const savedBadge =
    rep.saved_pct > 0
      ? `<strong style="color:#2ee59d">Giảm ${rep.saved_pct}% dung lượng (${rep.before_kb} KB → ${rep.after_kb} KB)</strong>`
      : `<strong>${rep.before_kb} KB → ${rep.after_kb} KB</strong>`;

  repBox.innerHTML = `
    <div>🎯 <strong>Kết quả Convert (${rep.raster_name}):</strong> ${rep.orig_dimensions} (${rep.orig_format}) ➜ <strong>${rep.new_dimensions} (${rep.new_format}, ${rep.new_mipmaps} mipmaps)</strong></div>
    <div>💾 Dung lượng: ${savedBadge}</div>
    <div>📐 ${rep.uv_compensation_note}</div>
    ${rep.lod_synced ? `<div>🔗 <strong>Đã tự động đồng bộ ảnh LOD:</strong> ${rep.lod_raster_name} (${rep.lod_dimensions}, ${rep.lod_size_kb} KB)</div>` : ''}
  `;

  if (commitToStaging) {
    state.staging = data.staging;
    updateTopBarCounters();
    renderDeployTab();
    await selectTxd(txdName);
    showToast(`✅ Đã lưu ${txdName}.txd (${rasterName}) vào Staging!`);
  } else {
    showToast(`👁️ Đã tạo bản xem trước cho ${rasterName}`);
  }
}

async function loadCurrentSurface3D() {
  if (!state.activeTxdData) return;
  const dffSelect = document.getElementById('select-preview-dff');
  const dffName = dffSelect.value;
  if (!dffName) {
    document.getElementById('surface-3d-stats').textContent = 'TXD này không gắn trực tiếp với model .DFF đơn lẻ.';
    return;
  }
  const dffImg = dffSelect.selectedOptions[0]?.dataset?.img || '';
  const txdName = state.activeTxdData.entry.txd_name;
  const stagedFlag = state.surface3DStaged ? 1 : 0;

  const dffUrl = `/api/assets/dff/${encodeURIComponent(dffName)}.dff?dff_img=${encodeURIComponent(dffImg)}&staged=${stagedFlag}&t=${Date.now()}`;
  const txdUrl = `/api/assets/txd/${encodeURIComponent(txdName)}.txd?staged=${stagedFlag}&t=${Date.now()}`;
  await surfaceViewport.loadDffAndTxd(dffUrl, txdUrl, `${dffName}.dff + ${txdName}.txd`);
}

// ---------------------------------------------------------------------------
// Tab 2: 3D Monuments & Statue Swapper
// ---------------------------------------------------------------------------
async function loadMonumentsList() {
  const region = document.getElementById('mon-filter-region').value;
  const q = document.getElementById('mon-filter-search').value.trim();
  const res = await fetch(`/api/monuments?region=${encodeURIComponent(region)}&q=${encodeURIComponent(q)}`);
  const data = await res.json();
  state.monuments = data.items || [];
  document.getElementById('monument-list-count').textContent = `${data.total} tượng`;
  renderMonumentsList();

  if (state.monuments.length > 0 && !state.activeMonument) {
    selectMonument(state.monuments[0].model_id);
  }
}

function renderMonumentsList() {
  const listEl = document.getElementById('monument-list');
  listEl.innerHTML = '';

  for (const m of state.monuments) {
    const div = document.createElement('div');
    div.className = `txd-item ${state.activeMonument?.model_id === m.model_id ? 'active' : ''}`;
    div.onclick = () => selectMonument(m.model_id);

    div.innerHTML = `
      <div class="txd-item-top">
        <span class="txd-name">🗽 ${m.dff_name}.dff</span>
        <span class="txd-size">ID ${m.model_id} · ${m.placements} vị trí</span>
      </div>
      <div class="txd-badges">
        <span class="badge cat">${m.txd_name}.txd</span>
        <span class="badge">${m.region}</span>
        ${m.is_staged ? '<span class="badge staged">✅ ĐÃ THAY TƯỢNG 3D</span>' : ''}
      </div>
    `;
    listEl.appendChild(div);
  }
}

async function selectMonument(modelId) {
  const res = await fetch(`/api/monuments/${modelId}`);
  if (!res.ok) return;
  const data = await res.json();
  state.activeMonument = data.monument;
  state.activeMonumentDetails = data;
  renderMonumentsList();

  const m = data.monument;
  const g = data.original_geometry || {};
  const dims = g.dimensions_m || [0, 0, 0];

  document.getElementById('active-mon-title').textContent =
    `🗽 ${m.dff_name}.dff (TXD: ${m.txd_name}.txd) — ${m.region}`;
  document.getElementById('active-mon-badges').innerHTML = `
    <span class="badge cat">Archive: ${m.dff_img}</span>
    <span class="badge">IDE: ${m.ide}</span>
    <span class="badge">Kích thước gốc: Rộng ${dims[0]}m × Sâu ${dims[1]}m × Cao ${dims[2]}m</span>
    ${data.is_staged ? '<span class="badge staged">✅ ĐANG HIỂN THỊ TƯỢNG WIBU MỚI</span>' : ''}
  `;

  document.getElementById('mon-diag-dff').textContent = `${m.dff_name}.dff`;
  document.getElementById('mon-diag-txd').textContent = `${m.txd_name}.txd`;
  document.getElementById('mon-diag-dims').textContent = `${dims[0]}m × ${dims[1]}m × Cao ${dims[2]}m`;
  document.getElementById('mon-diag-tris').textContent = `${(g.triangle_count || 0).toLocaleString()} tris`;

  await loadCurrentMonument3D();
}

async function loadCurrentMonument3D() {
  if (!state.activeMonument) return;
  const m = state.activeMonument;
  const stagedFlag = state.monument3DStaged ? 1 : 0;
  const dffUrl = `/api/assets/dff/${encodeURIComponent(m.dff_name)}.dff?dff_img=${encodeURIComponent(m.dff_img)}&staged=${stagedFlag}&t=${Date.now()}`;
  const txdUrl = `/api/assets/txd/${encodeURIComponent(m.txd_name)}.txd?staged=${stagedFlag}&t=${Date.now()}`;
  await monumentViewport.loadDffAndTxd(dffUrl, txdUrl, `${m.dff_name}.dff`);
}

async function searchSkinLibrary(q) {
  const res = await fetch(`/api/skin-library?q=${encodeURIComponent(q)}&limit=80`);
  const data = await res.json();
  state.skinLibraryItems = data.items || [];
  const sel = document.getElementById('skin-lib-select');
  sel.innerHTML = '';
  for (const s of state.skinLibraryItems) {
    const opt = document.createElement('option');
    opt.value = s.key;
    const seriesStr = s.series?.length ? ` [${s.series.join(', ')}]` : '';
    opt.textContent = `${s.name}${seriesStr} (${s.size_kb} KB)`;
    sel.appendChild(opt);
  }
  if (sel.options.length > 0) sel.selectedIndex = 0;
}

async function bakeMonumentStatue() {
  if (!state.activeMonument) {
    showToast('Hãy chọn một Tượng Đài 3D ở cột bên trái trước!', true);
    return;
  }

  const dffFile = document.getElementById('mon-custom-dff').files?.[0];
  const txdFile = document.getElementById('mon-custom-txd').files?.[0];
  let customDffB64 = null;
  let customTxdB64 = null;
  if (dffFile && txdFile) {
    customDffB64 = await readFileAsDataURL(dffFile);
    customTxdB64 = await readFileAsDataURL(txdFile);
  }

  const payload = {
    model_id: Number(state.activeMonument.model_id),
    skin_key: document.getElementById('skin-lib-select').value || null,
    statue_finish: document.getElementById('mon-finish-select').value,
    height_scale: parseFloat(document.getElementById('mon-height-scale').value),
    rotation_deg: parseFloat(document.getElementById('mon-rotation').value),
    z_offset_m: parseFloat(document.getElementById('mon-z-offset').value),
    add_pedestal: document.getElementById('mon-add-pedestal').checked,
    custom_dff_b64: customDffB64,
    custom_txd_b64: customTxdB64,
    custom_dff_name: dffFile ? dffFile.name : null,
  };

  showToast('⏳ Đang đóng băng khung xương, tạo bệ đá và nướng ánh sáng đỉnh cho tượng 3D...');
  const res = await fetch('/api/monuments/bake', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Lỗi đúc tượng 3D' }));
    showToast(err.detail || 'Lỗi đúc tượng 3D', true);
    return;
  }

  const data = await res.json();
  state.staging = data.staging;
  updateTopBarCounters();
  renderDeployTab();

  const st = data.stats;
  const repBox = document.getElementById('mon-bake-report');
  repBox.classList.remove('hidden');
  repBox.innerHTML = `
    <div>🗿 <strong>Đã đúc thành công Tượng 3D Tĩnh:</strong> <code>${state.activeMonument.dff_name}.dff</code></div>
    <div>👤 Nguồn: <strong>${st.source_model}</strong> · Chất liệu: <strong>${st.finish}</strong></div>
    <div>📏 Kích thước sau khớp bệ: <strong>${st.dimensions_m.join('m × ')}m</strong> (${st.vertex_count.toLocaleString()} đỉnh, ${st.triangle_count.toLocaleString()} tam giác)</div>
    <div>💾 Dung lượng: DFF <strong>${st.dff_size_kb} KB</strong> + TXD <strong>${st.txd_size_kb} KB</strong></div>
  `;

  state.monument3DStaged = true;
  await loadMonumentsList();
  await selectMonument(state.activeMonument.model_id);
  showToast(`✅ Đã thay tượng ${state.activeMonument.dff_name}.dff trong Staging!`);
}

// ---------------------------------------------------------------------------
// Tab 3: Staged Mod Pack, Deploy & Rollback
// ---------------------------------------------------------------------------
function renderDeployTab() {
  const txdTbody = document.querySelector('#staged-txds-table tbody');
  txdTbody.innerHTML = '';
  const stagedTxds = Object.values(state.staging.staged_txds || {});

  if (stagedTxds.length === 0) {
    txdTbody.innerHTML = `<tr><td colspan="7" style="text-align:center;color:#92a4c4;">Chưa có file .TXD bề mặt 2D nào trong Staging.</td></tr>`;
  } else {
    for (const item of stagedTxds) {
      const tr = document.createElement('tr');
      const rNames = Object.keys(item.modified_rasters || {}).join(', ');
      const deltaColor = item.delta_txd_size_kb <= 0 ? '#2ee59d' : '#ffc43d';
      tr.innerHTML = `
        <td><strong>${item.txd_filename}</strong></td>
        <td><span class="badge cat">${item.category}</span></td>
        <td>${item.region}</td>
        <td>${(item.target_imgs || []).join('<br/>')}</td>
        <td style="color:${deltaColor}">${item.orig_txd_size_kb} KB → <strong>${item.staged_txd_size_kb} KB</strong> (${item.delta_txd_size_kb > 0 ? '+' : ''}${item.delta_txd_size_kb} KB)</td>
        <td>${Object.keys(item.modified_rasters || {}).length} (${rNames})</td>
        <td><button class="btn btn-sm btn-danger" data-unstage-txd="${item.txd_name}">Xoá khỏi Staging</button></td>
      `;
      txdTbody.appendChild(tr);
    }
  }

  const statTbody = document.querySelector('#staged-statues-table tbody');
  statTbody.innerHTML = '';
  const stagedStatues = Object.values(state.staging.staged_statues || {});

  if (stagedStatues.length === 0) {
    statTbody.innerHTML = `<tr><td colspan="7" style="text-align:center;color:#92a4c4;">Chưa có Tượng Đài 3D (.DFF+.TXD) nào trong Staging.</td></tr>`;
  } else {
    for (const s of stagedStatues) {
      const meta = s.statue_meta || {};
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>🗽 ${s.dff_filename}</strong></td>
        <td>${s.txd_filename}</td>
        <td>${meta.source_model || '-'}</td>
        <td><span class="badge cat">${meta.finish || '-'}</span></td>
        <td>${(meta.dimensions_m || []).join(' × ')}m</td>
        <td>DFF: ${s.dff_size_kb} KB + TXD: ${s.txd_size_kb} KB</td>
        <td><button class="btn btn-sm btn-danger" data-unstage-statue="${s.dff_name}">Xoá khỏi Staging</button></td>
      `;
      statTbody.appendChild(tr);
    }
  }

  // Bind unstage buttons
  document.querySelectorAll('[data-unstage-txd]').forEach((btn) => {
    btn.onclick = async () => {
      const name = btn.dataset.unstageTxd;
      await fetch(`/api/staging/txd/${encodeURIComponent(name)}`, { method: 'DELETE' });
      await refreshGlobalState();
      await loadCatalogList();
      showToast(`Đã gỡ ${name}.txd khỏi Staging.`);
    };
  });
  document.querySelectorAll('[data-unstage-statue]').forEach((btn) => {
    btn.onclick = async () => {
      const name = btn.dataset.unstageStatue;
      await fetch(`/api/staging/statue/${encodeURIComponent(name)}`, { method: 'DELETE' });
      await refreshGlobalState();
      await loadMonumentsList();
      showToast(`Đã gỡ tượng ${name}.dff khỏi Staging.`);
    };
  });

  // Render transactions list
  const txListEl = document.getElementById('transactions-list');
  txListEl.innerHTML = '';
  if (!state.transactions || state.transactions.length === 0) {
    txListEl.innerHTML = `<div style="color:#92a4c4;">Chưa có lịch sử Deploy nào.</div>`;
  } else {
    for (const tx of state.transactions) {
      const card = document.createElement('div');
      card.className = 'tx-card';
      const isRolledBack = tx.status === 'rolled_back';
      card.innerHTML = `
        <div class="tx-card-header">
          <span>🔖 ${tx.transaction_id}</span>
          <span class="badge ${isRolledBack ? '' : 'staged'}">${isRolledBack ? 'ĐÃ ROLLBACK' : 'ĐANG ÁP DỤNG'}</span>
        </div>
        <div style="font-size:11.5px;color:#92a4c4;">Thời gian: ${tx.created_at}</div>
        <div>Số member đã ghi: <strong>${tx.members_patched_count}</strong> trên <strong>${(tx.archives_modified || []).length}</strong> kho .IMG</div>
        <div>Dung lượng tiết kiệm: <strong style="color:#2ee59d">${tx.saved_kb} KB</strong> (${tx.before_total_kb} KB → ${tx.after_total_kb} KB)</div>
        ${
          !isRolledBack
            ? `<button class="btn btn-sm btn-danger" data-rollback-id="${tx.transaction_id}">⏪ Phục Hồi Nguyên Trạng (Rollback Bit-for-Bit)</button>`
            : `<div style="font-size:11px;color:#ff8590;">Đã phục hồi lúc: ${tx.rolled_back_at}</div>`
        }
      `;
      txListEl.appendChild(card);
    }

    document.querySelectorAll('[data-rollback-id]').forEach((btn) => {
      btn.onclick = async () => {
        const txId = btn.dataset.rollbackId;
        const res = await fetch(`/api/rollback/${encodeURIComponent(txId)}`, { method: 'POST' });
        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: 'Lỗi Rollback' }));
          showToast(err.detail || 'Lỗi Rollback', true);
          return;
        }
        await refreshGlobalState();
        showToast(`⏪ Đã Rollback thành công ${txId} về nguyên trạng Bit-for-Bit!`);
      };
    });
  }
}

async function deployStagedToGame() {
  const res = await fetch('/api/deploy', { method: 'POST' });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Lỗi Deploy' }));
    showToast(err.detail || 'Lỗi khi Deploy', true);
    return;
  }
  const data = await res.json();
  await refreshGlobalState();
  showToast(
    `🚀 Đã Deploy thành công ${data.transaction.members_patched_count} member vào gameplay-lab (Mã: ${data.transaction.transaction_id})!`
  );
}

// ---------------------------------------------------------------------------
// UI Event Bindings
// ---------------------------------------------------------------------------
function bindUIEvents() {
  // Top navigation tabs
  document.querySelectorAll('.main-tab').forEach((btn) => {
    btn.onclick = () => {
      document.querySelectorAll('.main-tab').forEach((b) => b.classList.remove('active'));
      document.querySelectorAll('.workspace').forEach((w) => w.classList.remove('active'));
      btn.classList.add('active');
      const target = document.getElementById(btn.dataset.tab);
      if (target) target.classList.add('active');
      surfaceViewport?.resize();
      monumentViewport?.resize();
      if (btn.dataset.tab === 'tab-monuments' && state.activeMonument) {
        loadCurrentMonument3D();
      }
    };
  });

  // Category filter pills
  document.querySelectorAll('#category-pills .cat-pill').forEach((pill) => {
    pill.onclick = () => {
      document.querySelectorAll('#category-pills .cat-pill').forEach((p) => p.classList.remove('active'));
      pill.classList.add('active');
      state.selectedCategory = pill.dataset.cat;
      if (state.selectedCategory !== 'ALL' && state.selectedCategory !== 'CAT5_MONUMENTS_STATUES_LANDMARKS') {
        document.getElementById('conv-category-select').value = state.selectedCategory;
        updateConverterSchemaUI(state.selectedCategory);
      }
      loadCatalogList();
    };
  });

  ['filter-region', 'filter-tier', 'filter-sort', 'filter-include-lod'].forEach((id) => {
    document.getElementById(id).onchange = () => loadCatalogList();
  });

  let searchTimer = null;
  document.getElementById('filter-search').oninput = () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => loadCatalogList(), 220);
  };

  document.getElementById('raster-filter-input').oninput = () => renderRasterGrid();

  // 2D vs 3D View Switcher
  document.getElementById('btn-view-2d').onclick = () => {
    state.surfaceViewMode = '2d';
    document.getElementById('btn-view-2d').classList.add('active');
    document.getElementById('btn-view-3d').classList.remove('active');
    document.getElementById('view-2d-container').classList.add('active');
    document.getElementById('view-3d-container').classList.remove('active');
  };
  document.getElementById('btn-view-3d').onclick = () => {
    state.surfaceViewMode = '3d';
    document.getElementById('btn-view-3d').classList.add('active');
    document.getElementById('btn-view-2d').classList.remove('active');
    document.getElementById('view-3d-container').classList.add('active');
    document.getElementById('view-2d-container').classList.remove('active');
    surfaceViewport?.resize();
    loadCurrentSurface3D();
  };

  document.getElementById('select-preview-dff').onchange = () => loadCurrentSurface3D();
  document.getElementById('btn-3d-toggle-staged').onclick = (e) => {
    state.surface3DStaged = !state.surface3DStaged;
    e.currentTarget.textContent = state.surface3DStaged ? '🔄 Đang xem: Bản Mới (Staged)' : '🔄 Đang xem: Bản Gốc (Original)';
    loadCurrentSurface3D();
  };
  document.getElementById('btn-3d-wireframe').onclick = () => surfaceViewport.toggleWireframe();
  document.getElementById('btn-3d-autorotate').onclick = () => surfaceViewport.toggleAutoRotate();
  document.getElementById('btn-3d-resetcam').onclick = () => loadCurrentSurface3D();

  // Converter controls
  document.getElementById('conv-category-select').onchange = (e) => updateConverterSchemaUI(e.target.value);
  document.getElementById('opt-cat1-weather').oninput = (e) => {
    document.getElementById('val-cat1-weather').textContent = `${Math.round(e.target.value * 100)}%`;
  };
  document.getElementById('opt-cat1-border').oninput = (e) => {
    document.getElementById('val-cat1-border').textContent = `${e.target.value}px`;
  };
  document.getElementById('opt-cat2-wall').oninput = (e) => {
    document.getElementById('val-cat2-wall').textContent = `${Math.round(e.target.value * 100)}%`;
  };
  document.getElementById('opt-cat2-fabric').oninput = (e) => {
    document.getElementById('val-cat2-fabric').textContent = `${Math.round(e.target.value * 100)}%`;
  };
  document.getElementById('opt-cat3-scanline').oninput = (e) => {
    document.getElementById('val-cat3-scanline').textContent = `${Math.round(e.target.value * 100)}%`;
  };
  document.getElementById('opt-cat3-vibrance').oninput = (e) => {
    document.getElementById('val-cat3-vibrance').textContent = `${Math.round(e.target.value * 100)}%`;
  };
  document.getElementById('opt-cat3-usesub').onchange = (e) => {
    document.getElementById('cat3-subregion-box').classList.toggle('hidden', !e.target.checked);
  };
  document.getElementById('opt-cat4-border').oninput = (e) => {
    document.getElementById('val-cat4-border').textContent = `${Math.round(e.target.value * 100)}%`;
  };
  document.getElementById('opt-cat4-glass').oninput = (e) => {
    document.getElementById('val-cat4-glass').textContent = `${Math.round(e.target.value * 100)}%`;
  };

  document.getElementById('btn-conv-preview').onclick = () => runSurfaceConversion(false);
  document.getElementById('btn-conv-stage').onclick = () => runSurfaceConversion(true);

  // Tab 2: Monument controls
  document.getElementById('mon-filter-region').onchange = () => loadMonumentsList();
  document.getElementById('mon-filter-search').oninput = () => loadMonumentsList();
  document.getElementById('skin-lib-search').oninput = (e) => searchSkinLibrary(e.target.value);

  document.getElementById('mon-height-scale').oninput = (e) => {
    document.getElementById('val-mon-scale').textContent = `${parseFloat(e.target.value).toFixed(2)}×`;
  };
  document.getElementById('mon-rotation').oninput = (e) => {
    document.getElementById('val-mon-rot').textContent = `${e.target.value}°`;
  };
  document.getElementById('mon-z-offset').oninput = (e) => {
    document.getElementById('val-mon-zoff').textContent = `${parseFloat(e.target.value).toFixed(1)}m`;
  };

  document.getElementById('btn-mon-toggle-staged').onclick = (e) => {
    state.monument3DStaged = !state.monument3DStaged;
    e.currentTarget.textContent = state.monument3DStaged
      ? '🔄 Đang xem: Tượng Mới (Staged)'
      : '🔄 Đang xem: Tượng Gốc (Original)';
    loadCurrentMonument3D();
  };
  document.getElementById('btn-mon-wireframe').onclick = () => monumentViewport.toggleWireframe();
  document.getElementById('btn-mon-autorotate').onclick = () => monumentViewport.toggleAutoRotate();
  document.getElementById('btn-mon-resetcam').onclick = () => loadCurrentMonument3D();
  document.getElementById('btn-bake-monument').onclick = () => bakeMonumentStatue();

  // Deploy buttons
  document.getElementById('btn-quick-deploy').onclick = () => {
    document.querySelector('[data-tab="tab-deploy"]').click();
  };
  document.getElementById('btn-deploy-main').onclick = () => deployStagedToGame();
}

initApp();
