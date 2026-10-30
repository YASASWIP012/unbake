const $ = (id) => document.getElementById(id);

const state = { recipes: [], recipe: null, photo: null, canRender: false };
const COLORS = { x: "#f2c14e", y: "#6fcf97", zoom: "#7fb2ff", roll: "#eb6f6f" };
const SPLICE_LAYERS = ["camera", "transform", "look"];

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${response.status}`);
  }
  return response.json();
}

function curve(values, color, scale) {
  if (!values.length) return "";
  const points = values.map((v, i) => {
    const x = (i / (values.length - 1)) * 100;
    const y = 50 - (v / scale) * 44;
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  });
  return `<path d="M${points.join("L")}" stroke="${color}"/>`;
}

function curveChart(series) {
  const scale = Math.max(1e-3, ...Object.values(series).flat().map(Math.abs));
  const paths = Object.entries(series).map(([k, v]) => curve(v, COLORS[k] ?? COLORS.x, scale));
  return `<svg class="curve" viewBox="0 0 100 100" preserveAspectRatio="none">${paths.join("")}</svg>`;
}

function escapeHtml(text) {
  const node = document.createElement("span");
  node.textContent = text ?? "";
  return node.innerHTML;
}

function spliceMenu(layer) {
  const options = state.recipes
    .filter((r) => r.id !== state.recipe.id)
    .map((r) => `<option value="${r.id}">${escapeHtml(r.name)}</option>`)
    .join("");
  if (!options) return "";
  return `<select data-splice="${layer}" aria-label="Take this layer from another recipe">
    <option value="">Take from…</option>${options}</select>`;
}

function card(layer, title, body) {
  const on = state.recipe[layer].enabled;
  const menu = SPLICE_LAYERS.includes(layer) ? spliceMenu(layer) : "";
  return `<article class="card ${on ? "" : "off"}">
    <header><h3>${title}</h3>${menu}
      <input type="checkbox" data-toggle="${layer}" aria-label="Use this layer" ${on ? "checked" : ""}>
    </header>${body}</article>`;
}

function renderRecipe() {
  const r = state.recipe;
  const box = $("recipe");
  box.hidden = !r;
  $("exportLink").hidden = !r;
  if (!r) return;
  $("exportLink").href = `/api/recipes/${r.id}/export`;

  const camera = card("camera", "Camera", `
    <p>${r.camera.move.replaceAll("_", " ").replaceAll("+", " + ")}, ${r.camera.easing.replaceAll("_", " ")}</p>
    ${curveChart({ x: r.camera.x, y: r.camera.y, zoom: r.camera.zoom, roll: r.camera.roll })}
    <div class="legend"><span style="color:${COLORS.x}">pan</span><span style="color:${COLORS.y}">tilt</span>
      <span style="color:${COLORS.zoom}">zoom</span><span style="color:${COLORS.roll}">roll</span></div>
    <label>Size of move <input type="range" min="0" max="2" step="0.05" value="${r.camera.gain}" data-field="camera.gain"></label>`);

  const transform = card("transform", "Transform", `
    <p>${r.transform.present ? `The picture changes by ${Math.round(r.transform.amount * 100)}%` : "No big change measured"}</p>
    <label>Becomes</label>
    <textarea rows="2" data-field="transform.after" placeholder="What the subject turns into">${escapeHtml(r.transform.after)}</textarea>`);

  const look = card("look", "Look", `
    <div class="swatches">${r.look.palette.map((c) => `<i style="background:${c}"></i>`).join("")}</div>
    <label>Strength <input type="range" min="0" max="1" step="0.05" value="${r.look.strength}" data-field="look.strength"></label>`);

  const pacing = card("pacing", "Pacing", `
    ${curveChart({ x: r.pacing.energy })}
    <p>${r.pacing.ramps.length ? `${r.pacing.ramps.length} speed ramp(s)` : "Even pace"}</p>`);

  const sound = card("sound", "Sound", `
    <textarea rows="2" data-field="sound.description" placeholder="Describe the sound">${escapeHtml(r.sound.description)}</textarea>`);

  const warnings = r.unsupported.map((w) => `<p class="warn">${escapeHtml(w)}</p>`).join("");
  box.innerHTML = `<input class="title" data-field="name" value="${escapeHtml(r.name)}" aria-label="Recipe name">
    ${warnings}${camera}${transform}${look}${pacing}${sound}`;
}

function renderShelf() {
  const shelf = $("shelf");
  shelf.replaceChildren(
    ...state.recipes.map((r) => {
      const chip = document.createElement("button");
      chip.className = `chip ${state.recipe?.id === r.id ? "active" : ""}`;
      chip.textContent = r.name;
      chip.onclick = () => select(r);
      return chip;
    }),
  );
}

function select(recipe) {
  state.recipe = recipe;
  renderShelf();
  renderRecipe();
  updateButtons();
}

function updateButtons() {
  const ready = Boolean(state.recipe && state.photo);
  $("previewButton").disabled = !ready;
  $("bakeButton").disabled = !ready || !state.canRender;
}

function log(text) {
  const item = document.createElement("li");
  item.textContent = text;
  $("log").append(item);
}

let saveTimer;
function saveSoon() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(async () => {
    const r = state.recipe;
    await api(`/api/recipes/${r.id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(r),
    });
  }, 300);
}

$("recipe").addEventListener("input", (event) => {
  const field = event.target.dataset.field;
  if (!field) return;
  const path = field.split(".");
  const value = event.target.type === "range" ? Number(event.target.value) : event.target.value;
  const owner = path.length === 2 ? state.recipe[path[0]] : state.recipe;
  owner[path.at(-1)] = value;
  saveSoon();
});

$("recipe").addEventListener("change", async (event) => {
  const { toggle, splice } = event.target.dataset;
  if (toggle) {
    state.recipe[toggle].enabled = event.target.checked;
    saveSoon();
    renderRecipe();
  }
  if (splice && event.target.value) {
    const child = await api("/api/splice", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ base: state.recipe.id, layers: { [splice]: event.target.value } }),
    });
    state.recipes.unshift(child);
    select(child);
  }
});

async function unbake(file) {
  const label = $("clipDrop").querySelector("strong");
  label.textContent = "Measuring…";
  try {
    const body = new FormData();
    body.append("clip", file);
    const recipe = await api("/api/unbake", { method: "POST", body });
    state.recipes.unshift(recipe);
    select(recipe);
  } catch (error) {
    alert(error.message);
  } finally {
    label.textContent = "Drop a clip";
  }
}

function setPhoto(file) {
  state.photo = file;
  $("photoView").src = URL.createObjectURL(file);
  $("photoView").hidden = false;
  $("photoLabel").hidden = true;
  updateButtons();
}

function bindDrop(zone, input, handler) {
  input.addEventListener("change", () => input.files[0] && handler(input.files[0]));
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("over"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("over");
    if (e.dataTransfer.files[0]) handler(e.dataTransfer.files[0]);
  });
}

function showVideo(url) {
  const video = $("result");
  video.src = url;
  video.hidden = false;
  video.play().catch(() => {});
}

function renderReport(report) {
  const box = $("report");
  box.hidden = false;
  const bars = Object.entries(report.layers).map(([name, layer]) => `
    <div class="bar ${report.failing.includes(name) ? "fail" : ""}">
      <span>${name}</span><div><i style="width:${layer.score}%"></i></div><span>${Math.round(layer.score)}</span>
      <small>${escapeHtml(layer.detail)}</small>
    </div>`);
  box.innerHTML = `<div class="overall">${Math.round(report.overall)}<small> / 100 match</small></div>${bars.join("")}`;
}

$("previewButton").onclick = async () => {
  const body = new FormData();
  body.append("photo", state.photo);
  $("previewButton").disabled = true;
  try {
    const { url, report } = await api(`/api/recipes/${state.recipe.id}/preview`, { method: "POST", body });
    showVideo(url);
    renderReport(report);
  } catch (error) {
    alert(error.message);
  } finally {
    updateButtons();
  }
};

$("bakeButton").onclick = async () => {
  const body = new FormData();
  body.append("photo", state.photo);
  body.append("subject", $("subject").value);
  $("log").replaceChildren();
  $("bakeButton").disabled = true;
  try {
    const { run } = await api(`/api/recipes/${state.recipe.id}/bake`, { method: "POST", body });
    const source = new EventSource(`/api/jobs/${run}/events`);
    source.onmessage = (message) => {
      const event = JSON.parse(message.data);
      if (event.kind === "keyframe") log("Making the end keyframe");
      if (event.kind === "pass") log(`Render pass ${event.number}`);
      if (event.kind === "fix") log(`Adjusting ${event.notes.join("; ")}`);
      if (event.kind === "scored") {
        log(`Pass ${event.pass} scored ${Math.round(event.report.overall)}`);
        renderReport(event.report);
      }
      if (event.kind === "done") {
        source.close();
        renderReport(event.best.report);
        showVideo(`/api/runs/${event.run}/${event.best.video}`);
        updateButtons();
      }
      if (event.kind === "error") {
        source.close();
        log(event.message);
        updateButtons();
      }
    };
  } catch (error) {
    alert(error.message);
    updateButtons();
  }
};

$("exampleButton").onclick = async () => {
  const blob = await (await fetch("example.jpg")).blob();
  setPhoto(new File([blob], "example.jpg", { type: "image/jpeg" }));
};

bindDrop($("clipDrop"), $("clipInput"), unbake);
bindDrop($("photoDrop"), $("photoInput"), setPhoto);

const [status, recipes] = await Promise.all([api("/api/status"), api("/api/recipes")]);
state.canRender = status.can_render;
state.recipes = recipes;
$("engine").textContent = status.can_render ? "Renders on Comfy" : "Camera preview only: add a Comfy key to render";
renderShelf();
if (recipes.length) select(recipes[0]);
