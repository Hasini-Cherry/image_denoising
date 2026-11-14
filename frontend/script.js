const API_BASE = "http://127.0.0.1:8000";

const elements = {
  apiState: document.getElementById("apiState"),
  themeToggle: document.getElementById("themeToggle"),
  dropZone: document.getElementById("dropZone"),
  fileInput: document.getElementById("fileInput"),
  manualModeBtn: document.getElementById("manualModeBtn"),
  autoModeBtn: document.getElementById("autoModeBtn"),
  noiseType: document.getElementById("noiseType"),
  intensity: document.getElementById("intensity"),
  intensityValue: document.getElementById("intensityValue"),
  rank: document.getElementById("rank"),
  rankValue: document.getElementById("rankValue"),
  rankHint: document.getElementById("rankHint"),
  recommendedRange: document.getElementById("recommendedRange"),
  processBtn: document.getElementById("processBtn"),
  downloadBtn: document.getElementById("downloadBtn"),
  message: document.getElementById("message"),
  originalImg: document.getElementById("originalImg"),
  noisyImg: document.getElementById("noisyImg"),
  denoisedImg: document.getElementById("denoisedImg"),
  originalMeta: document.getElementById("originalMeta"),
  noisyMeta: document.getElementById("noisyMeta"),
  denoisedMeta: document.getElementById("denoisedMeta"),
  originalMetrics: document.getElementById("originalMetrics"),
  noisyMetrics: document.getElementById("noisyMetrics"),
  denoisedMetrics: document.getElementById("denoisedMetrics"),
  noisyPsnr: document.getElementById("noisyPsnr"),
  noisySsim: document.getElementById("noisySsim"),
  bestK: document.getElementById("bestK"),
  processingTime: document.getElementById("processingTime"),
  autoSection: document.getElementById("autoSection"),
  autoGrid: document.getElementById("autoGrid"),
  autoSummary: document.getElementById("autoSummary"),
};

const state = {
  mode: "manual",
  originalDataUrl: null,
  noisyDataUrl: null,
  denoisedDataUrl: null,
  autoResults: [],
  maxRank: 1,
};

function setMessage(text, type = "") {
  elements.message.textContent = text;
  elements.message.className = `message ${type}`.trim();
}

function setBusy(isBusy) {
  elements.processBtn.disabled = isBusy || !state.originalDataUrl;
  elements.processBtn.textContent = isBusy ? "Processing..." : state.mode === "auto" ? "Process Auto" : "Process Manual";
}

function formatMeta(metadata) {
  const resized = metadata.resized ? " resized" : "";
  return `${metadata.width}x${metadata.height}${resized}`;
}

function formatMetrics(metrics) {
  if (!metrics) return "PSNR: -- | SSIM: --";
  return `PSNR: ${Number(metrics.psnr).toFixed(2)} dB | SSIM: ${Number(metrics.ssim).toFixed(4)}`;
}

function showImage(img, dataUrl) {
  const frame = img.closest(".image-frame");
  img.removeAttribute("src");
  frame.classList.remove("has-image");
  if (!dataUrl) return;
  requestAnimationFrame(() => {
    img.src = dataUrl;
    frame.classList.add("has-image");
  });
}

function resetResults(keepOriginal = true) {
  state.noisyDataUrl = null;
  state.denoisedDataUrl = null;
  state.autoResults = [];
  showImage(elements.noisyImg, null);
  showImage(elements.denoisedImg, null);
  elements.noisyMeta.textContent = "Waiting";
  elements.denoisedMeta.textContent = "Waiting";
  elements.noisyMetrics.textContent = "PSNR: -- | SSIM: --";
  elements.denoisedMetrics.textContent = "PSNR: -- | SSIM: --";
  elements.noisyPsnr.textContent = "--";
  elements.noisySsim.textContent = "--";
  elements.bestK.textContent = "--";
  elements.processingTime.textContent = "-- ms";
  elements.downloadBtn.disabled = true;
  elements.autoGrid.innerHTML = "";
  elements.autoSummary.textContent = "Fixed percentage k selection";
  if (!keepOriginal) {
    state.originalDataUrl = null;
    showImage(elements.originalImg, null);
    elements.originalMeta.textContent = "Waiting";
    elements.originalMetrics.textContent = "Baseline image";
  }
}

function isAcceptedFile(file) {
  const acceptedTypes = ["image/jpeg", "image/png"];
  const name = file.name.toLowerCase();
  return acceptedTypes.includes(file.type) || name.endsWith(".jpg") || name.endsWith(".jpeg") || name.endsWith(".png");
}

async function apiFetch(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    let message = `Request failed with status ${response.status}.`;
    try {
      const error = await response.json();
      message = error.detail || message;
    } catch {
      message = await response.text();
    }
    throw new Error(message);
  }
  return response.json();
}

async function dataUrlToBlob(dataUrl) {
  const response = await fetch(dataUrl);
  return response.blob();
}

function intensityBand() {
  const value = Number(elements.intensity.value);
  if (value <= 0.2) return "low";
  if (value <= 0.5) return "medium";
  return "high";
}

function recommendedPercentRange() {
  const band = intensityBand();
  if (band === "low") return [0.15, 0.3];
  if (band === "medium") return [0.1, 0.2];
  return [0.05, 0.15];
}

function updateRecommendedRange() {
  if (!state.originalDataUrl) {
    elements.recommendedRange.textContent = "Recommended k range: --";
    return;
  }
  const [low, high] = recommendedPercentRange();
  const lowK = Math.max(1, Math.round(state.maxRank * low));
  const highK = Math.max(lowK, Math.round(state.maxRank * high));
  elements.recommendedRange.textContent = `Recommended k range: ${lowK} - ${highK}`;
}

function applyRankHint() {
  const k = Number(elements.rank.value);
  const maxRank = Number(elements.rank.max);
  elements.rankValue.textContent = String(k);
  updateRecommendedRange();
  if (!state.originalDataUrl) {
    elements.rankHint.textContent = "Upload an image to set the valid k range.";
    return;
  }
  if (k <= Math.max(2, Math.floor(maxRank * 0.04))) {
    elements.rankHint.textContent = "Very low k: stronger compression, possible distortion.";
    return;
  }
  if (k >= Math.max(1, Math.floor(maxRank * 0.85))) {
    elements.rankHint.textContent = "Very high k: preserves detail, denoising may be weak.";
    return;
  }
  elements.rankHint.textContent = `Valid range: 1 to ${maxRank}.`;
}

function setMode(mode) {
  state.mode = mode;
  document.body.classList.toggle("auto-mode", mode === "auto");
  elements.manualModeBtn.classList.toggle("active", mode === "manual");
  elements.autoModeBtn.classList.toggle("active", mode === "auto");
  elements.autoSection.classList.toggle("active", mode === "auto");
  resetResults(true);
  setBusy(false);
  setMessage(mode === "auto" ? "Auto mode will generate 5 denoised outputs from one SVD." : "Manual mode gives direct k control.");
}

function toggleTheme() {
  const isLight = document.body.classList.toggle("light-mode");
  elements.themeToggle.textContent = isLight ? "Light" : "Dark";
  localStorage.setItem("svd-theme", isLight ? "light" : "dark");
}

function restoreTheme() {
  const saved = localStorage.getItem("svd-theme");
  const isLight = saved === "light";
  document.body.classList.toggle("light-mode", isLight);
  elements.themeToggle.textContent = isLight ? "Light" : "Dark";
}

async function handleFile(file) {
  if (!file) return;
  if (!isAcceptedFile(file)) {
    setMessage("Invalid file. Please upload JPG, JPEG, or PNG.", "error");
    return;
  }

  setMessage("Uploading and validating image...");
  setBusy(true);
  resetResults(false);

  try {
    const result = await apiFetch("/upload", {
      method: "POST",
      headers: { "Content-Type": file.type || "application/octet-stream" },
      body: file,
    });
    state.originalDataUrl = result.image;
    state.maxRank = result.metadata.max_rank;
    showImage(elements.originalImg, state.originalDataUrl);
    elements.originalMeta.textContent = formatMeta(result.metadata);
    elements.originalMetrics.textContent = "Reference for PSNR and SSIM";

    elements.rank.max = String(state.maxRank);
    const [low] = recommendedPercentRange();
    elements.rank.value = String(Math.max(1, Math.round(state.maxRank * low)));
    applyRankHint();
    elements.processBtn.disabled = false;

    const resizedMessage = result.metadata.resized
      ? `Image resized from ${result.metadata.original_width}x${result.metadata.original_height} to ${result.metadata.width}x${result.metadata.height}.`
      : "Image ready.";
    setMessage(resizedMessage, result.metadata.resized ? "warn" : "");
  } catch (error) {
    resetResults(false);
    elements.processBtn.disabled = true;
    setMessage(error.message, "error");
  } finally {
    setBusy(false);
  }
}

async function processImage() {
  if (!state.originalDataUrl) {
    setMessage("Upload a valid image first.", "error");
    return;
  }
  if (state.mode === "auto") {
    await processAuto();
  } else {
    await processManual();
  }
}

async function processManual() {
  setBusy(true);
  setMessage("Adding noise, denoising, and evaluating manual k...");
  const started = performance.now();
  try {
    const originalBlob = await dataUrlToBlob(state.originalDataUrl);
    const noiseResult = await apiFetch(
      `/add-noise?noise_type=${encodeURIComponent(elements.noiseType.value)}&intensity=${encodeURIComponent(elements.intensity.value)}`,
      {
        method: "POST",
        headers: { "Content-Type": "image/png" },
        body: originalBlob,
      },
    );
    state.noisyDataUrl = noiseResult.image;
    showImage(elements.noisyImg, state.noisyDataUrl);
    elements.noisyMeta.textContent = `${elements.noiseType.options[elements.noiseType.selectedIndex].text} ${Number(elements.intensity.value).toFixed(2)}`;

    const noisyMetrics = await apiFetch("/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ original: state.originalDataUrl, denoised: state.noisyDataUrl }),
    });
    elements.noisyMetrics.textContent = formatMetrics(noisyMetrics);
    elements.noisyPsnr.textContent = `${noisyMetrics.psnr.toFixed(2)} dB`;
    elements.noisySsim.textContent = noisyMetrics.ssim.toFixed(4);

    const noisyBlob = await dataUrlToBlob(state.noisyDataUrl);
    const denoiseResult = await apiFetch(`/denoise?k=${encodeURIComponent(elements.rank.value)}`, {
      method: "POST",
      headers: { "Content-Type": "image/png" },
      body: noisyBlob,
    });
    state.denoisedDataUrl = denoiseResult.image;
    showImage(elements.denoisedImg, state.denoisedDataUrl);
    elements.denoisedMeta.textContent = `k=${denoiseResult.denoise.k}`;
    elements.downloadBtn.disabled = false;

    const denoisedMetrics = await apiFetch("/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ original: state.originalDataUrl, denoised: state.denoisedDataUrl }),
    });
    elements.denoisedMetrics.textContent = formatMetrics(denoisedMetrics);

    const serverTime = Number(noiseResult.processing_time_ms) + Number(denoiseResult.processing_time_ms);
    elements.processingTime.textContent = `${serverTime.toFixed(2)} ms`;
    elements.processingTime.title = `Browser round trip: ${(performance.now() - started).toFixed(2)} ms`;
    elements.bestK.textContent = String(denoiseResult.denoise.k);

    const warnings = denoiseResult.denoise.warnings || [];
    setMessage(warnings.length ? warnings.join(" ") : "Manual denoising complete.", warnings.length ? "warn" : "");
  } catch (error) {
    setMessage(error.message, "error");
  } finally {
    setBusy(false);
  }
}

async function processAuto() {
  setBusy(true);
  setMessage("Running auto mode with one SVD and five k reconstructions...");
  try {
    const originalBlob = await dataUrlToBlob(state.originalDataUrl);
    const result = await apiFetch(
      `/auto-denoise?noise_type=${encodeURIComponent(elements.noiseType.value)}&intensity=${encodeURIComponent(elements.intensity.value)}`,
      {
        method: "POST",
        headers: { "Content-Type": "image/png" },
        body: originalBlob,
      },
    );

    state.noisyDataUrl = result.noisy_image;
    state.autoResults = result.results.map((item, index) => ({ ...item, best: index === result.best_index }));
    showImage(elements.noisyImg, state.noisyDataUrl);
    elements.noisyMeta.textContent = `${elements.noiseType.options[elements.noiseType.selectedIndex].text} ${Number(elements.intensity.value).toFixed(2)}`;
    elements.noisyMetrics.textContent = formatMetrics(result.noisy_metrics);
    elements.noisyPsnr.textContent = `${result.noisy_metrics.psnr.toFixed(2)} dB`;
    elements.noisySsim.textContent = result.noisy_metrics.ssim.toFixed(4);
    elements.bestK.textContent = String(result.best_k);
    elements.processingTime.textContent = `${Number(result.processing_time_ms).toFixed(2)} ms`;
    elements.autoSummary.textContent = `Ranks: 20%, 30%, 40%, 50%, 60%; best k=${result.best_k}`;

    renderAutoGrid(state.autoResults);
    setMessage("Auto denoising complete. Best result is highlighted by highest SSIM.");
  } catch (error) {
    setMessage(error.message, "error");
  } finally {
    setBusy(false);
  }
}

function renderAutoGrid(results) {
  elements.autoGrid.innerHTML = "";
  results.forEach((result, index) => {
    const card = document.createElement("article");
    card.className = `image-card ${result.best ? "best" : ""}`;
    card.innerHTML = `
      <div class="card-head">
        <h2>${result.best ? "Best " : ""}Denoised ${index + 1}</h2>
        <span>k=${result.k}</span>
      </div>
      <div class="image-frame">
        <img alt="Auto denoised output ${index + 1}" />
        <span class="empty-label">No image</span>
      </div>
      <div class="metric-strip">${formatMetrics(result.metrics)}</div>
      <div class="download-group" aria-label="Download k ${result.k}">
        <button class="download-mini" type="button" data-format="png">PNG</button>
        <button class="download-mini" type="button" data-format="jpg">JPG</button>
        <button class="download-mini" type="button" data-format="jpeg">JPEG</button>
      </div>
    `;
    const img = card.querySelector("img");
    card.querySelectorAll(".download-mini").forEach((button) => {
      button.addEventListener("click", () => {
        const format = button.dataset.format;
        downloadImageAs(result.image, `svd-auto-k-${result.k}.${format}`, format);
      });
    });
    elements.autoGrid.appendChild(card);
    showImage(img, result.image);
  });
}

function downloadDataUrl(dataUrl, filename) {
  if (!dataUrl) return;
  const link = document.createElement("a");
  link.href = dataUrl;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
}

function downloadImageAs(dataUrl, filename, format) {
  if (!dataUrl) return;
  if (format === "png") {
    downloadDataUrl(dataUrl, filename);
    return;
  }

  const image = new Image();
  image.onload = () => {
    const canvas = document.createElement("canvas");
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0);
    downloadDataUrl(canvas.toDataURL("image/jpeg", 0.95), filename);
  };
  image.onerror = () => setMessage("Could not prepare this image for download.", "error");
  image.src = dataUrl;
}

function downloadDenoised() {
  downloadDataUrl(state.denoisedDataUrl, "svd-denoised.png");
}

async function checkApi() {
  try {
    await apiFetch("/health");
    elements.apiState.textContent = "API: online";
    elements.apiState.classList.remove("offline");
  } catch {
    elements.apiState.textContent = "API: offline";
    elements.apiState.classList.add("offline");
    setMessage("Start the FastAPI backend at http://127.0.0.1:8000 before processing.", "warn");
  }
}

elements.fileInput.addEventListener("change", (event) => handleFile(event.target.files[0]));
elements.processBtn.addEventListener("click", processImage);
elements.downloadBtn.addEventListener("click", downloadDenoised);
elements.themeToggle.addEventListener("click", toggleTheme);
elements.manualModeBtn.addEventListener("click", () => setMode("manual"));
elements.autoModeBtn.addEventListener("click", () => setMode("auto"));
elements.intensity.addEventListener("input", () => {
  elements.intensityValue.textContent = Number(elements.intensity.value).toFixed(2);
  applyRankHint();
});
elements.rank.addEventListener("input", applyRankHint);

["dragenter", "dragover"].forEach((eventName) => {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.add("drag-over");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.remove("drag-over");
  });
});

elements.dropZone.addEventListener("drop", (event) => {
  handleFile(event.dataTransfer.files[0]);
});

restoreTheme();
setMode("manual");
applyRankHint();
checkApi();
