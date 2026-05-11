import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAMES = new Set([
  "WAN22MultiImageI2VGuide",
  "WAN22ImageGuideManager",
  "WAN22GenerateAllInOne",
]);
const GUIDE_ROW_HEIGHT = 32;
const DEFAULT_PAYLOAD = { version: 1, guides: [] };
const ICONS = {
  plus: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 5v14"/><path d="M5 12h14"/></svg>`,
  folderPlus: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 6h6l2 2h10v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M12 14h6"/><path d="M15 11v6"/></svg>`,
  folderMinus: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 6h6l2 2h10v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M12 14h6"/></svg>`,
  refresh: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 12a8 8 0 0 1-13.7 5.7"/><path d="M4 12A8 8 0 0 1 17.7 6.3"/><path d="M7 18H3v-4"/><path d="M17 6h4v4"/></svg>`,
  save: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 3h12l2 2v16H5z"/><path d="M8 3v6h8V3"/><path d="M8 21v-7h8v7"/></svg>`,
  folderArrow: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 6h6l2 2h10v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M12 15h6"/><path d="M15 12l3 3-3 3"/></svg>`,
  folder: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 6h6l2 2h10v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>`,
  folderTree: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 6h6l2 2h9a2 2 0 0 1 2 2v2"/><path d="M6 12v6a2 2 0 0 0 2 2h5"/><path d="M10 15h4l1.5 1.5H21v3.5H10z"/></svg>`,
  grid: `<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>`,
  eye: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6-10-6-10-6z"/><circle cx="12" cy="12" r="3"/></svg>`,
  eyeOff: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6-10-6-10-6z"/><circle cx="12" cy="12" r="3"/><path d="M3 3l18 18"/></svg>`,
  up: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 19V5"/><path d="M5 12l7-7 7 7"/></svg>`,
  down: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 5v14"/><path d="M19 12l-7 7-7-7"/></svg>`,
  remove: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M18 6 6 18"/><path d="M6 6l12 12"/></svg>`,
};

function injectStyles() {
  if (document.getElementById("wan22-guide-styles")) return;
  const style = document.createElement("style");
  style.id = "wan22-guide-styles";
  style.textContent = `
    .wan22-guide-root { box-sizing: border-box; width: 100%; color: #ddd; font: 12px Arial, sans-serif; padding: 2px 0 12px; }
    .wan22-guide-root * { box-sizing: border-box; }
    .wan22-guide-toolbar { display: flex; gap: 5px; flex-wrap: wrap; margin-bottom: 6px; }
    .wan22-guide-toolbar button, .wan22-guide-row button, .wan22-guide-dialog button {
      background: #333; color: #ddd; border: 1px solid #555; border-radius: 4px; padding: 4px 7px; cursor: pointer;
    }
    .wan22-guide-toolbar button:hover, .wan22-guide-row button:hover, .wan22-guide-dialog button:hover { background: #444; }
    .wan22-guide-toolbar button { width: 28px; height: 28px; padding: 2px; display: inline-flex; align-items: center; justify-content: center; }
    .wan22-guide-toolbar svg, .wan22-guide-browser-icon-button svg, .wan22-guide-columns-control svg {
      width: 17px; height: 17px; stroke: currentColor; fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round;
    }
    .wan22-guide-list { max-height: ${GUIDE_ROW_HEIGHT * 5 + 2}px; overflow: auto; border: 1px solid #303030; border-radius: 4px; }
    .wan22-guide-empty { padding: 8px; color: #aaa; border: 1px solid #303030; border-radius: 4px; }
    .wan22-guide-row { display: grid; grid-template-columns: 18px minmax(0, 1fr) 58px 50px 26px 26px 28px; gap: 4px; align-items: center; min-height: ${GUIDE_ROW_HEIGHT}px; padding: 4px; border-bottom: 1px solid #303030; }
    .wan22-guide-row:last-child { border-bottom: 0; }
    .wan22-guide-row input { min-width: 0; background: #181818; color: #ddd; border: 1px solid #555; border-radius: 3px; padding: 2px 3px; }
    .wan22-guide-row button { min-width: 0; height: 24px; padding: 1px 4px; display: inline-flex; align-items: center; justify-content: center; }
    .wan22-guide-row button svg { width: 15px; height: 15px; stroke: currentColor; fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
    .wan22-guide-name { overflow: hidden; white-space: nowrap; text-overflow: ellipsis; cursor: zoom-in; }
    .wan22-guide-hover-preview { position: fixed; z-index: 10000; pointer-events: none; display: none; max-width: 520px; max-height: 380px; overflow: auto; background: #191919; border: 1px solid #555; border-radius: 6px; box-shadow: 0 8px 30px rgba(0,0,0,.45); padding: 8px; }
    .wan22-guide-hover-preview.visible { display: block; }
    .wan22-guide-preview-item { display: grid; grid-template-columns: 74px 1fr; gap: 8px; padding: 5px; border-bottom: 1px solid #303030; }
    .wan22-guide-preview-item:last-child { border-bottom: 0; }
    .wan22-guide-preview-item img { max-width: 74px; max-height: 74px; object-fit: contain; background: #111; border: 1px solid #333; }
    .wan22-guide-warning { color: #e6b85c; }
    .wan22-guide-dialog { position: fixed; inset: 0; z-index: 10001; display: flex; align-items: center; justify-content: center; background: rgba(0,0,0,.58); }
    .wan22-guide-panel { width: 760px; max-width: 94vw; max-height: 88vh; overflow: auto; background: #222; border: 1px solid #555; border-radius: 6px; padding: 14px; color: #ddd; }
    .wan22-guide-panel h3 { margin: 0 0 10px; font-size: 15px; }
    .wan22-guide-controls { display: grid; grid-template-columns: minmax(140px, 1fr) minmax(150px, 1fr) auto minmax(130px, 180px); gap: 8px; align-items: center; margin-bottom: 10px; }
    .wan22-guide-controls select, .wan22-guide-controls input { background: #151515; color: #ddd; border: 1px solid #555; border-radius: 4px; padding: 6px; }
    .wan22-guide-browser-options { display: flex; align-items: center; gap: 8px; margin: -2px 0 10px; }
    .wan22-guide-browser-icon-button { width: 32px; height: 32px; padding: 4px !important; display: inline-flex; align-items: center; justify-content: center; }
    .wan22-guide-browser-icon-button svg, .wan22-guide-columns-control svg { width: 18px; height: 18px; }
    .wan22-guide-columns-control { display: grid; grid-template-columns: 22px minmax(0, 1fr) 18px; gap: 6px; align-items: center; color: #ddd; }
    .wan22-guide-columns-control input { width: 100%; min-width: 0; padding: 0; }
    .wan22-guide-columns-control span { text-align: right; color: #aaa; font-size: 11px; }
    .wan22-guide-grid { display: grid; grid-template-columns: repeat(var(--wan22-cols, 5), minmax(0, 1fr)); gap: 8px; max-height: 54vh; overflow: auto; padding: 2px; }
    .wan22-guide-tile { min-width: 0; text-align: left; background: #181818; border: 1px solid #444; border-radius: 5px; padding: 5px; color: #ddd; cursor: pointer; }
    .wan22-guide-tile.selected { border-color: #8ab4f8; background: #202a36; }
    .wan22-guide-tile img { display: block; width: 100%; aspect-ratio: 1 / 1; object-fit: contain; background: #101010; border: 1px solid #303030; border-radius: 3px; transition: opacity .12s ease; }
    .wan22-guide-grid.hide-images .wan22-guide-tile img { opacity: 0; }
    .wan22-guide-panel:hover .wan22-guide-grid.hide-images .wan22-guide-tile img { opacity: 1; }
    .wan22-guide-grid.show-images .wan22-guide-tile img { opacity: 1; }
    .wan22-guide-tile span { display: block; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; margin-top: 5px; font-size: 11px; }
    .wan22-guide-actions { display: flex; gap: 8px; justify-content: flex-end; margin-top: 12px; }
    .wan22-guide-muted { color: #aaa; }
    .wan22-guide-preview { position: fixed; z-index: 10002; inset: 0; display: flex; align-items: center; justify-content: center; background: rgba(0,0,0,.72); padding: 24px; }
    .wan22-guide-preview div { position: relative; max-width: 94vw; max-height: 94vh; background: #151515; border: 1px solid #555; border-radius: 6px; padding: 10px; }
    .wan22-guide-preview img { display: block; max-width: calc(94vw - 20px); max-height: calc(94vh - 56px); object-fit: contain; }
    .wan22-guide-preview p { margin: 7px 0 0; max-width: calc(94vw - 20px); overflow: hidden; white-space: nowrap; text-overflow: ellipsis; color: #aaa; }
  `;
  document.head.appendChild(style);
}

function getWidget(node, name) {
  return node.widgets?.find((widget) => widget.name === name);
}

function getWidgetValue(node, name, fallback) {
  const widget = getWidget(node, name);
  return widget ? widget.value : fallback;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#39;",
  })[char]);
}

function hideWidget(widget) {
  if (!widget) return;
  widget.type = "hidden";
  widget.label = "";
  widget.hidden = true;
  widget.options = { ...(widget.options || {}), hidden: true };
  widget.draw = () => {};
  widget.computeSize = () => [0, -4];
  for (const element of [widget.element, widget.inputEl]) {
    if (!element?.style) continue;
    element.style.display = "none";
    element.style.width = "0px";
    element.style.height = "0px";
  }
}

function parsePayload(raw) {
  try {
    const payload = JSON.parse(raw || "{}");
    return { ...DEFAULT_PAYLOAD, ...payload, guides: Array.isArray(payload.guides) ? payload.guides : [] };
  } catch {
    return { ...DEFAULT_PAYLOAD, guides: [] };
  }
}

function calcFrame(node, guide) {
  const length = Math.max(1, Number(getWidgetValue(node, "length", 49)) || 49);
  const fps = Math.max(0.001, Number(getWidgetValue(node, "fps", 24)) || 24);
  const mode = getWidgetValue(node, "timing_mode", "frame");
  let frame = mode === "seconds" ? Math.round(Number(guide.position || 0) * fps) : Math.round(Number(guide.position || 0));
  if (frame < 0) frame = length + frame;
  return Math.max(0, Math.min(length - 1, frame));
}

function readPayload(node) {
  const widget = getWidget(node, "guides_json");
  const raw = node.properties?.wan22_guides_json || widget?.value || JSON.stringify(DEFAULT_PAYLOAD);
  return parsePayload(raw);
}

function writePayload(node, payload) {
  const widget = getWidget(node, "guides_json");
  payload.version = 1;
  payload.guides = (payload.guides || []).map((guide) => ({ ...guide, calculated_frame: calcFrame(node, guide) }));
  for (const key of ["fps", "timing_mode", "resize_mode", "duplicate_policy", "pad_color", "global_strength", "start_images_strength", "structural_repulsion_boost", "width", "height", "length"]) {
    const widgetValue = getWidgetValue(node, key, undefined);
    if (widgetValue !== undefined) payload[key] = widgetValue;
  }
  const text = JSON.stringify(payload);
  if (widget) widget.value = text;
  node.properties = node.properties || {};
  node.properties.wan22_guides_json = text;
  node.properties.wan22_guides = payload.guides;
  app.graph.setDirtyCanvas(true, true);
}

function nextPosition(node, payload) {
  const length = Math.max(1, Number(getWidgetValue(node, "length", 49)) || 49);
  if (!payload.guides.length) return 0;
  const last = Math.max(...payload.guides.map((guide) => calcFrame(node, guide)));
  return Math.min(length - 1, last + 8);
}

function updateNodeSize(node) {
  const count = readPayload(node).guides.length;
  const panelHeight = 34 + (count ? Math.min(count, 5) * GUIDE_ROW_HEIGHT + 8 : 40);
  const size = node.computeSize?.() || node.size || [390, 260];
  node.setSize([Math.max(size[0], 420), Math.max(size[1], panelHeight + 160)]);
}

function renderGuidePanel(node) {
  const state = node._wan22Guides;
  if (!state) return;
  const payload = readPayload(node);
  const root = state.container;
  root.innerHTML = "";

  const toolbar = document.createElement("div");
  toolbar.className = "wan22-guide-toolbar";
  toolbar.innerHTML = `
    <button data-action="add" title="Open image browser to add a manual guide image" aria-label="Add guide image">${ICONS.plus}</button>
    <button data-action="folder-add" title="Add a configured image folder alias" aria-label="Add folder">${ICONS.folderPlus}</button>
    <button data-action="folder-remove" title="Remove a configured image folder alias" aria-label="Remove folder">${ICONS.folderMinus}</button>
    <button data-action="refresh" title="Refresh folder and image listings" aria-label="Refresh folders and images">${ICONS.refresh}</button>
    <button data-action="save" title="Save current manual guide list as a guide set" aria-label="Save guide set">${ICONS.save}</button>
    <button data-action="load" title="Load a saved guide set" aria-label="Load guide set">${ICONS.folderArrow}</button>
  `;
  toolbar.addEventListener("click", async (event) => {
    const button = event.target.closest("button[data-action]");
    if (!button) return;
    const action = button.dataset.action;
    if (action === "add") openImageBrowser(node);
    if (action === "folder-add") addConfiguredFolder(node);
    if (action === "folder-remove") removeConfiguredFolder();
    if (action === "refresh") refreshGuideFolders();
    if (action === "save") saveGuideSet(node);
    if (action === "load") loadGuideSet(node);
  });
  root.append(toolbar);

  if (!payload.guides.length) {
    const empty = document.createElement("div");
    empty.className = "wan22-guide-empty";
    empty.textContent = "No inserted guide frames";
    root.append(empty);
    updateNodeSize(node);
    return;
  }

  const list = document.createElement("div");
  list.className = "wan22-guide-list";
  payload.guides.forEach((guide, index) => {
    const row = document.createElement("div");
    row.className = "wan22-guide-row";

    const enabled = document.createElement("input");
    enabled.type = "checkbox";
    enabled.checked = guide.enabled !== false;
    enabled.onchange = () => {
      guide.enabled = enabled.checked;
      writePayload(node, payload);
    };

    const name = document.createElement("div");
    name.className = "wan22-guide-name";
    name.textContent = guide.filename;
    name.addEventListener("mouseenter", (event) => showPreviewForGuide(node, guide, event));
    name.addEventListener("mousemove", (event) => positionPreview(node, event));
    name.addEventListener("mouseleave", () => hidePreview(node));

    const position = document.createElement("input");
    position.type = "number";
    position.step = getWidgetValue(node, "timing_mode", "frame") === "seconds" ? "0.01" : "1";
    position.value = guide.position ?? 0;
    position.title = "Frame index or seconds, depending on timing_mode";
    position.onchange = () => {
      guide.position = Number(position.value) || 0;
      writePayload(node, payload);
      renderGuidePanel(node);
    };

    const strength = document.createElement("input");
    strength.type = "number";
    strength.min = "0";
    strength.max = "1";
    strength.step = "0.01";
    strength.value = guide.strength ?? 1;
    strength.title = "Guide strength. 1 locks the frame, 0 ignores it.";
    strength.onchange = () => {
      guide.strength = Math.max(0, Math.min(1, Number(strength.value) || 0));
      writePayload(node, payload);
    };

    const up = document.createElement("button");
    up.innerHTML = ICONS.up;
    up.title = "Move guide up";
    up.setAttribute("aria-label", "Move guide up");
    up.onclick = () => {
      if (index <= 0) return;
      [payload.guides[index - 1], payload.guides[index]] = [payload.guides[index], payload.guides[index - 1]];
      writePayload(node, payload);
      renderGuidePanel(node);
    };

    const down = document.createElement("button");
    down.innerHTML = ICONS.down;
    down.title = "Move guide down";
    down.setAttribute("aria-label", "Move guide down");
    down.onclick = () => {
      if (index >= payload.guides.length - 1) return;
      [payload.guides[index + 1], payload.guides[index]] = [payload.guides[index], payload.guides[index + 1]];
      writePayload(node, payload);
      renderGuidePanel(node);
    };

    const remove = document.createElement("button");
    remove.innerHTML = ICONS.remove;
    remove.title = "Remove guide";
    remove.setAttribute("aria-label", "Remove guide");
    remove.onclick = () => {
      payload.guides.splice(index, 1);
      writePayload(node, payload);
      renderGuidePanel(node);
    };

    row.append(enabled, name, position, strength, up, down, remove);
    list.append(row);
  });
  root.append(list);
  updateNodeSize(node);
}

async function fetchJson(url, options) {
  const response = await api.fetchApi(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

async function addConfiguredFolder(node) {
  const alias = prompt("Folder alias");
  if (!alias) return;
  const path = prompt("Folder path");
  if (!path) return;
  try {
    await fetchJson("/wan22_guides/folders", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ alias, path }),
    });
    openImageBrowser(node);
  } catch (error) {
    alert(error.message);
  }
}

async function removeConfiguredFolder() {
  try {
    const data = await fetchJson("/wan22_guides/folders");
    const aliases = (data.folders || []).map((folder) => folder.alias);
    const alias = prompt(`Folder alias to remove:\n${aliases.join("\n")}`);
    if (!alias) return;
    await fetchJson(`/wan22_guides/folders?alias=${encodeURIComponent(alias)}`, { method: "DELETE" });
  } catch (error) {
    alert(error.message);
  }
}

async function refreshGuideFolders() {
  try {
    await fetchJson("/wan22_guides/refresh", { method: "POST" });
  } catch (error) {
    alert(error.message);
  }
}

async function openImageBrowser(node) {
  let folders = [];
  try {
    folders = (await fetchJson("/wan22_guides/folders")).folders || [];
  } catch (error) {
    alert(error.message);
    return;
  }
  const payload = readPayload(node);
  const dialog = document.createElement("div");
  dialog.className = "wan22-guide-dialog";
  const panel = document.createElement("div");
  panel.className = "wan22-guide-panel";
  panel.innerHTML = `<h3>WAN 2.2 Guide Images</h3>`;

  const controls = document.createElement("div");
  controls.className = "wan22-guide-controls";
  controls.innerHTML = `
    <select class="folder" title="Choose configured image folder"></select>
    <input class="search" type="search" placeholder="Search images..." title="Search loaded image filenames and relative paths">
    <button class="scope wan22-guide-browser-icon-button" type="button" title="Recursive folder view" aria-label="Recursive folder view"></button>
    <label class="wan22-guide-columns-control" title="Thumbnail columns per row">
      ${ICONS.grid}
      <input class="columns" type="range" min="2" max="8" step="1" value="5">
      <span class="columns-value">5</span>
    </label>
  `;

  const folderSelect = controls.querySelector(".folder");
  const searchInput = controls.querySelector(".search");
  const scopeButton = controls.querySelector(".scope");
  const columns = controls.querySelector(".columns");
  const columnsValue = controls.querySelector(".columns-value");
  for (const folder of folders) {
    const option = document.createElement("option");
    option.value = folder.alias;
    option.textContent = `${folder.alias}${folder.exists ? "" : " (missing)"} - ${folder.image_count} images`;
    folderSelect.append(option);
  }

  const options = document.createElement("div");
  options.className = "wan22-guide-browser-options";
  options.innerHTML = `
    <button class="hover-hide wan22-guide-browser-icon-button" type="button" title="Hide images until hovering over window" aria-label="Hide images until hovering over window"></button>
    <span class="wan22-guide-muted">Image previews</span>
  `;
  const hoverHideButton = options.querySelector(".hover-hide");

  const grid = document.createElement("div");
  grid.className = "wan22-guide-grid hide-images";
  const meta = document.createElement("div");
  meta.className = "wan22-guide-muted";
  meta.style.marginTop = "8px";
  const actions = document.createElement("div");
  actions.className = "wan22-guide-actions";
  const cancelButton = document.createElement("button");
  cancelButton.textContent = "Cancel";
  const addButton = document.createElement("button");
  addButton.textContent = "Add Selected";
  actions.append(cancelButton, addButton);
  panel.append(controls, options, grid, meta, actions);
  dialog.append(panel);
  document.body.append(dialog);

  const selected = new Map();
  let availableImages = [];
  let recursive = true;
  let hideImagesUntilHover = true;

  function syncScopeButton() {
    scopeButton.innerHTML = recursive ? ICONS.folderTree : ICONS.folder;
    scopeButton.title = recursive ? "Recursive folder view" : "Current folder only";
    scopeButton.setAttribute("aria-label", scopeButton.title);
  }

  function syncGridVisibility() {
    hoverHideButton.innerHTML = hideImagesUntilHover ? ICONS.eyeOff : ICONS.eye;
    hoverHideButton.title = hideImagesUntilHover ? "Hide images until hovering over window" : "Show image previews";
    hoverHideButton.setAttribute("aria-label", hoverHideButton.title);
    grid.classList.toggle("hide-images", hideImagesUntilHover);
    grid.classList.toggle("show-images", !hideImagesUntilHover);
  }

  function syncColumns() {
    grid.style.setProperty("--wan22-cols", columns.value);
    columnsValue.textContent = columns.value;
  }

  function renderImageGrid() {
    grid.innerHTML = "";
    const query = searchInput.value.trim().toLowerCase();
    const images = query
      ? availableImages.filter((image) => image.filename.toLowerCase().includes(query))
      : availableImages;
    meta.textContent = `${images.length} of ${availableImages.length} images`;
    for (const image of images) {
      const tile = document.createElement("button");
      tile.className = "wan22-guide-tile";
      if (selected.has(image.filename)) tile.classList.add("selected");
      tile.innerHTML = `<img loading="lazy"><span></span>`;
      tile.querySelector("img").src = image.thumb_url;
      tile.querySelector("span").textContent = image.filename;
      tile.title = `${image.filename} (${image.width}x${image.height})\nClick to select. Ctrl-click for large preview.`;
      tile.onclick = (event) => {
        if (event.ctrlKey || event.metaKey) {
          showPreview({ folder_alias: folderSelect.value, ...image });
          return;
        }
        if (selected.has(image.filename)) {
          selected.delete(image.filename);
          tile.classList.remove("selected");
        } else {
          selected.set(image.filename, image);
          tile.classList.add("selected");
        }
      };
      grid.append(tile);
    }
  }

  cancelButton.onclick = () => dialog.remove();
  addButton.onclick = () => {
    let position = nextPosition(node, payload);
    for (const image of selected.values()) {
      payload.guides.push({
        folder_alias: folderSelect.value,
        filename: image.filename,
        width: image.width,
        height: image.height,
        position,
        calculated_frame: position,
        strength: 1.0,
        enabled: true,
      });
      position = Math.min(Math.max(1, Number(getWidgetValue(node, "length", 49)) || 49) - 1, position + 8);
    }
    writePayload(node, payload);
    dialog.remove();
    renderGuidePanel(node);
  };
  columns.oninput = syncColumns;
  searchInput.oninput = renderImageGrid;
  scopeButton.onclick = () => {
    recursive = !recursive;
    syncScopeButton();
    loadImages();
  };
  hoverHideButton.onclick = () => {
    hideImagesUntilHover = !hideImagesUntilHover;
    syncGridVisibility();
  };

  async function loadImages() {
    grid.innerHTML = "";
    selected.clear();
    meta.textContent = "Loading...";
    try {
      const data = await fetchJson(`/wan22_guides/images?alias=${encodeURIComponent(folderSelect.value)}&recursive=${recursive ? "1" : "0"}`);
      availableImages = data.images || [];
      renderImageGrid();
    } catch (error) {
      availableImages = [];
      meta.textContent = error.message;
    }
  }
  folderSelect.onchange = loadImages;
  syncScopeButton();
  syncGridVisibility();
  syncColumns();
  loadImages();
}

function showPreview(guide) {
  const dialog = document.createElement("div");
  dialog.className = "wan22-guide-preview";
  const panel = document.createElement("div");
  const img = document.createElement("img");
  img.src = `/wan22_guides/image?alias=${encodeURIComponent(guide.folder_alias)}&filename=${encodeURIComponent(guide.filename)}`;
  const caption = document.createElement("p");
  caption.textContent = `${guide.folder_alias}/${guide.filename}`;
  panel.append(img, caption);
  dialog.append(panel);
  dialog.onclick = () => dialog.remove();
  document.body.append(dialog);
}

function setupPreview(node) {
  const panel = document.createElement("div");
  panel.className = "wan22-guide-hover-preview";
  document.body.appendChild(panel);
  node._wan22Guides.preview = panel;

  const onRemoved = node.onRemoved;
  node.onRemoved = function () {
    panel.remove();
    onRemoved?.apply(this, arguments);
  };
}

function positionPreview(node, event) {
  const panel = node._wan22Guides?.preview;
  if (!panel) return;
  panel.style.left = `${Math.min(window.innerWidth - 540, event.clientX + 14)}px`;
  panel.style.top = `${Math.min(window.innerHeight - 400, event.clientY + 14)}px`;
}

function hidePreview(node) {
  node._wan22Guides?.preview?.classList.remove("visible");
}

function showPreviewForGuide(node, guide, event) {
  const panel = node._wan22Guides?.preview;
  if (!panel) return;
  const width = Number(getWidgetValue(node, "width", 1280)) || 1280;
  const height = Number(getWidgetValue(node, "height", 704)) || 704;
  const targetRatio = width / height;
  const guideRatio = guide.width && guide.height ? guide.width / guide.height : targetRatio;
  const ratioWarning = Boolean(guide.width && guide.height && Math.abs(targetRatio - guideRatio) > 0.02);
  const thumbUrl = `/wan22_guides/thumb?alias=${encodeURIComponent(guide.folder_alias)}&filename=${encodeURIComponent(guide.filename)}`;
  const timingSuffix = getWidgetValue(node, "timing_mode", "frame") === "seconds" ? "s" : "f";
  panel.innerHTML = `
    <div class="wan22-guide-preview-item">
      <img src="${thumbUrl}" alt="">
      <div>
        <div>${escapeHtml(guide.filename)}</div>
        <div class="wan22-guide-muted">${escapeHtml(guide.folder_alias)}</div>
        <div>${guide.position ?? 0}${timingSuffix} -> ${calcFrame(node, guide)}f</div>
        <div>${guide.width || "?"}x${guide.height || "?"}</div>
        <div>strength ${guide.strength ?? 1}</div>
        <div>${guide.enabled === false ? "disabled" : "enabled"}</div>
        ${ratioWarning ? `<div class="wan22-guide-warning">aspect ratio differs</div>` : ""}
      </div>
    </div>`;
  positionPreview(node, event);
  panel.classList.add("visible");
}

async function saveGuideSet(node) {
  const name = prompt("Guide set name");
  if (!name) return;
  try {
    await fetchJson(`/wan22_guides/guide_sets/${encodeURIComponent(name)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(readPayload(node)),
    });
  } catch (error) {
    alert(error.message);
  }
}

async function loadGuideSet(node) {
  try {
    const data = await fetchJson("/wan22_guides/guide_sets");
    const name = prompt(`Guide set name:\n${(data.guide_sets || []).join("\n")}`);
    if (!name) return;
    const payload = await fetchJson(`/wan22_guides/guide_sets/${encodeURIComponent(name)}`);
    writePayload(node, payload);
    renderGuidePanel(node);
  } catch (error) {
    alert(error.message);
  }
}

function setupNode(node) {
  if (node._wan22Guides) return;
  const widget = getWidget(node, "guides_json");
  if (!widget) return;
  hideWidget(widget);
  widget.serializeValue = () => widget.value || JSON.stringify(DEFAULT_PAYLOAD);
  const container = document.createElement("div");
  container.className = "wan22-guide-root";
  node._wan22Guides = { container };
  setupPreview(node);
  const domWidget = node.addDOMWidget("wan22_guide_manager", "div", container, { serialize: false });
  domWidget.serialize = false;
  domWidget.computeSize = (width) => {
    const count = readPayload(node).guides.length;
    return [width, 42 + (count ? Math.min(count, 5) * GUIDE_ROW_HEIGHT + 8 : 40)];
  };
  writePayload(node, readPayload(node));
  renderGuidePanel(node);

  for (const name of ["fps", "timing_mode", "length", "width", "height"]) {
    const input = getWidget(node, name);
    if (!input || input.__wan22GuideHooked) continue;
    input.__wan22GuideHooked = true;
    const prior = input.callback;
    input.callback = function (...args) {
      if (prior) prior.apply(this, args);
      writePayload(node, readPayload(node));
      renderGuidePanel(node);
    };
  }
}

app.registerExtension({
  name: "helto.wan22.imageGuides",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODE_NAMES.has(nodeData.name)) return;
    const original = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      original?.apply(this, arguments);
      injectStyles();
      setupNode(this);
    };
    const originalSerialize = nodeType.prototype.onSerialize;
    nodeType.prototype.onSerialize = function (info) {
      writePayload(this, readPayload(this));
      info.wan22_guides = readPayload(this).guides;
      originalSerialize?.apply(this, arguments);
    };
    const originalConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      originalConfigure?.apply(this, arguments);
      if (info?.wan22_guides && !this.properties?.wan22_guides_json) {
        this.properties = this.properties || {};
        this.properties.wan22_guides_json = JSON.stringify({ version: 1, guides: info.wan22_guides });
      }
      setTimeout(() => {
        setupNode(this);
        renderGuidePanel(this);
      }, 0);
    };
  },
});
