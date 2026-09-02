const $ = (id) => document.getElementById(id);
let selectedBlob = null;
let selectedName = "";
let recorder = null;
let timerHandle = null;
let activeListenPath = "raw";

async function loadStatus() {
  try {
    const state = await fetch("/api/status").then((r) => r.json());
    const ok = state.google && state.ai_coustics;
    $("status").className = ok ? "status hidden" : "status missing";
    $("status").lastElementChild.textContent = "Credentials needed · see README";
  } catch {
    $("status").className = "status missing";
    $("status").lastElementChild.textContent = "Server unavailable";
  }
}

function setFile(blob, name) {
  selectedBlob = blob; selectedName = name;
  $("selectedName").textContent = name;
  $("selectedMeta").textContent = `${(blob.size / 1024 / 1024).toFixed(2)} MB · WAV`;
  $("selectedFile").classList.remove("hidden");
  $("runButton").disabled = false;
}

function clearFile() {
  selectedBlob = null; selectedName = ""; $("fileInput").value = "";
  $("selectedFile").classList.add("hidden"); $("runButton").disabled = true;
}

function setLiveText(path, active) {
  const text = [...active.finals[path], active.interim[path]].filter(Boolean).join(" ");
  const target = path === "raw" ? $("rawText") : $("enhancedText");
  target.textContent = text;
  if (!text) target.innerHTML = '<span class="empty">Waiting for speech...</span>';
}

function sendAudioBlocks(active, input) {
  const merged = new Float32Array(active.pending.length + input.length);
  merged.set(active.pending); merged.set(input, active.pending.length);
  let offset = 0;
  while (offset + active.blockSize <= merged.length) {
    const block = merged.slice(offset, offset + active.blockSize);
    if (active.socket.readyState === WebSocket.OPEN) active.socket.send(block.buffer);
    offset += active.blockSize;
  }
  active.pending = merged.slice(offset);
}

function beginAudioCapture(active) {
  if (active.processor) return;
  active.source = active.context.createMediaStreamSource(active.stream);
  active.processor = active.context.createScriptProcessor(2048, 1, 1);
  active.silent = active.context.createGain(); active.silent.gain.value = 0;
  active.processor.onaudioprocess = (event) => sendAudioBlocks(active, new Float32Array(event.inputBuffer.getChannelData(0)));
  active.source.connect(active.processor); active.processor.connect(active.silent); active.silent.connect(active.context.destination);
  active.started = Date.now();
  $("recordButton").disabled = false; $("recordButton").classList.add("recording"); $("recordButton").setAttribute("aria-pressed", "true"); $("recordButton").setAttribute("aria-label", "Stop live comparison"); $("recordLabel").textContent = "Listening";
  timerHandle = setInterval(() => { const elapsed = (Date.now() - active.started) / 1000; $("recordHelp").textContent = `${elapsed.toFixed(1)} seconds · click to stop`; if (elapsed >= 600) stopLiveComparison(); }, 100);
}

async function releaseMicrophone(active) {
  if (active.cleaned) return;
  active.cleaned = true;
  clearInterval(timerHandle);
  if (active.processor) active.processor.disconnect();
  if (active.source) active.source.disconnect();
  if (active.silent) active.silent.disconnect();
  active.stream.getTracks().forEach((track) => track.stop());
  if (active.context.state !== "closed") await active.context.close();
}

function resetLiveUi() {
  $("inputCard").classList.remove("live-active"); $("recordButton").classList.remove("recording"); $("recordButton").disabled = false; $("recordButton").setAttribute("aria-pressed", "false"); $("recordButton").setAttribute("aria-label", "Start live comparison"); $("recordLabel").textContent = "Start live comparison"; $("recordHelp").textContent = "Transcribe as you speak";
}

async function startLiveComparison() {
  $("errorBox").classList.add("hidden");
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
  const context = new AudioContext(); await context.resume();
  const blockSize = Math.round(context.sampleRate * 0.015);
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${location.host}/ws/live`);
  const active = { socket, stream, context, blockSize, pending: new Float32Array(), processor: null, source: null, silent: null, finals: { raw: [], enhanced: [] }, interim: { raw: "", enhanced: "" }, started: null, cleaned: false };
  recorder = active;
  $("inputCard").classList.add("live-active"); $("recordButton").disabled = true; $("recordLabel").textContent = "Connecting"; $("recordHelp").textContent = "Loading Quail and Gemini Live";
  $("results").classList.remove("hidden"); $("results").classList.add("live"); $("resultEyebrow").textContent = "Live comparison"; $("resultTitle").textContent = "Listening side by side"; $("rawTime").textContent = "Live"; $("enhancedTime").textContent = "Live"; $("rawModel").textContent = "Gemini 3.5 Transcribe Live";
  setLiveText("raw", active); setLiveText("enhanced", active);
  socket.addEventListener("open", () => socket.send(JSON.stringify({ sample_rate: context.sampleRate, block_size: blockSize, enhancement_level: Number($("enhancementLevel").value), language_codes: $("languages").value, vocabulary: $("vocabulary").value, tyto: $("tyto").checked })));
  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "status") {
      $("recordHelp").textContent = message.text;
      if (message.status === "ready") {
        if (message.transcription_available === false) {
          $("resultTitle").textContent = "Tyto live · Gemini blocked";
          $("rawText").innerHTML = '<span class="empty">Gemini unavailable</span>';
          $("enhancedText").innerHTML = '<span class="empty">Gemini unavailable</span>';
        } else $("quailModel").textContent = `Quail VF 2.2 · ${message.quail_delay_ms}ms`;
        beginAudioCapture(active);
      }
    }
    if (message.type === "transcript") {
      if (message.final) { active.finals[message.path].push(message.text); active.interim[message.path] = ""; }
      else active.interim[message.path] = message.text;
      setLiveText(message.path, active);
    }
    if (message.type === "tyto") {
      $("tytoMetric").classList.remove("hidden");
      $("tytoRisk").textContent = message.risk_score.toFixed(2);
      const dimensions = ["noise", "interfering_speech", "speaker_reverb", "packet_loss", "codec_degradation"];
      const cause = dimensions.sort((a, b) => message[b] - message[a])[0];
      $("tytoDetail").textContent = `highest: ${cause.replaceAll("_", " ")}`;
    }
    if (message.type === "tyto_warning") $("tytoDetail").textContent = message.message;
    if (message.type === "warning") showError(message.message);
    if (message.type === "error") { showError(message.message); releaseMicrophone(active); resetLiveUi(); recorder = null; socket.close(); }
    if (message.type === "complete") { $("resultTitle").textContent = "Live result"; resetLiveUi(); recorder = null; socket.close(); }
  });
  socket.addEventListener("error", async () => { if (recorder === active) { showError("Could not connect to live transcription."); await releaseMicrophone(active); resetLiveUi(); recorder = null; } });
}

async function stopLiveComparison() {
  const active = recorder;
  if (!active) return;
  $("recordButton").disabled = true; $("recordLabel").textContent = "Finishing"; $("recordHelp").textContent = "Waiting for final transcript";
  if (active.pending.length) {
    const finalBlock = new Float32Array(active.blockSize); finalBlock.set(active.pending);
    if (active.socket.readyState === WebSocket.OPEN) active.socket.send(finalBlock.buffer);
    active.pending = new Float32Array();
  }
  if (active.socket.readyState === WebSocket.OPEN) active.socket.send(JSON.stringify({ action: "stop" }));
  await releaseMicrophone(active);
}

function words(text) { return text.toLowerCase().match(/[\p{L}\p{N}'-]+/gu) || []; }
function lcs(a, b) {
  const rows = Array.from({ length: a.length + 1 }, () => new Uint16Array(b.length + 1));
  for (let i = 1; i <= a.length; i++) for (let j = 1; j <= b.length; j++) rows[i][j] = a[i - 1] === b[j - 1] ? rows[i - 1][j - 1] + 1 : Math.max(rows[i - 1][j], rows[i][j - 1]);
  return rows[a.length][b.length];
}
function alignment(a, b) {
  const rows = Array.from({ length: a.length + 1 }, () => new Uint16Array(b.length + 1));
  for (let i = 1; i <= a.length; i++) for (let j = 1; j <= b.length; j++) rows[i][j] = a[i - 1] === b[j - 1] ? rows[i - 1][j - 1] + 1 : Math.max(rows[i - 1][j], rows[i][j - 1]);
  const left = new Set(), right = new Set(); let i = a.length, j = b.length;
  while (i && j) { if (a[i - 1] === b[j - 1]) { left.add(--i); right.add(--j); } else if (rows[i - 1][j] >= rows[i][j - 1]) i--; else j--; }
  return [left, right];
}
function editDistance(a, b) {
  let prior = Array.from({length:b.length + 1}, (_,i) => i);
  for (let i=1;i<=a.length;i++) { const row=[i]; for (let j=1;j<=b.length;j++) row[j] = Math.min(row[j-1]+1, prior[j]+1, prior[j-1]+(a[i-1]===b[j-1]?0:1)); prior=row; }
  return prior[b.length];
}
function escapeHtml(value) { const div = document.createElement("div"); div.textContent = value; return div.innerHTML; }
function renderTranscript(target, result, common = null) {
  if (!result.text) { target.innerHTML = '<span class="empty">No speech detected.</span>'; return; }
  const labelled = result.segments.some((segment) => segment.speaker);
  if (!common) { target.innerHTML = labelled ? result.segments.map((segment) => `<span class="speaker">${escapeHtml(segment.speaker || "Speaker")}</span>${escapeHtml(segment.text)}`).join("") : escapeHtml(result.text); return; }
  let index = 0;
  target.innerHTML = (result.text.match(/[\p{L}\p{N}'-]+|[^\p{L}\p{N}'-]+/gu) || []).map((token) => /[\p{L}\p{N}]/u.test(token) ? `<span class="${common.has(index++) ? "" : "changed"}">${escapeHtml(token)}</span>` : escapeHtml(token)).join("");
}
function showError(error) { $("errorText").textContent = error; $("errorBox").classList.remove("hidden"); }

async function runComparison() {
  if (!selectedBlob) return;
  $("errorBox").classList.add("hidden"); $("inputCard").classList.add("hidden"); $("results").classList.add("hidden"); $("processing").classList.remove("hidden");
  const started = Date.now(); $("processingStep").textContent = "Enhancing audio, then transcribing both paths in parallel...";
  const clock = setInterval(() => { const sec = Math.floor((Date.now() - started) / 1000); $("processingTime").textContent = `${String(Math.floor(sec / 60)).padStart(2,"0")}:${String(sec % 60).padStart(2,"0")}`; }, 250);
  const form = new FormData();
  form.append("audio_file", selectedBlob, selectedName); form.append("enhancement_level", $("enhancementLevel").value); form.append("language_codes", $("languages").value); form.append("vocabulary", $("vocabulary").value);
  for (const id of ["diarization", "timestamps", "tyto"]) form.append(id === "timestamps" ? "word_timestamps" : id, $(id).checked ? "true" : "false");
  try {
    const response = await fetch("/api/compare", { method: "POST", body: form });
    const data = await response.json(); if (!response.ok) throw new Error(data.detail || "Comparison failed");
    const rawWords = words(data.raw.text), enhancedWords = words(data.enhanced.text), [rawCommon, enhancedCommon] = alignment(rawWords, enhancedWords);
    renderTranscript($("rawText"), data.raw, rawCommon); renderTranscript($("enhancedText"), data.enhanced, enhancedCommon);
    $("rawAudio").src = data.raw.audio_url; $("enhancedAudio").src = data.enhanced.audio_url;
    activeListenPath = "raw"; $("listenRaw").classList.add("active"); $("listenEnhanced").classList.remove("active"); $("abPlay").textContent = "▶ Play";
    $("rawTime").textContent = `${(data.raw.elapsed_ms / 1000).toFixed(1)}s response`; $("enhancedTime").textContent = `${(data.enhanced.elapsed_ms / 1000).toFixed(1)}s response`; $("quailModel").textContent = data.quail.model;
    const shared = rawCommon.size, maxWords = Math.max(rawWords.length, enhancedWords.length), referenceWords = words($("reference").value);
    if (referenceWords.length) { const rawWer = editDistance(referenceWords, rawWords) / referenceWords.length * 100, enhancedWer = editDistance(referenceWords, enhancedWords) / referenceWords.length * 100; $("primaryMetricLabel").textContent = "Word error rate"; $("wordDelta").textContent = `${rawWer.toFixed(1)}→${enhancedWer.toFixed(1)}%`; $("primaryMetricDetail").textContent = `${(rawWer-enhancedWer).toFixed(1)} points improvement`; }
    else { const delta = enhancedWords.length - rawWords.length; $("primaryMetricLabel").textContent = "Word delta"; $("wordDelta").textContent = `${delta > 0 ? "+" : ""}${delta}`; $("primaryMetricDetail").textContent = "Quail vs control"; }
    $("similarity").textContent = maxWords ? `${Math.round(shared / maxWords * 100)}%` : "-";
    $("quailTime").textContent = `${data.quail.processing_ms}ms`; $("quailDelay").textContent = `${data.quail.audio_delay_ms}ms signal delay`;
    if (data.tyto) { $("tytoMetric").classList.remove("hidden"); $("tytoRisk").textContent = data.tyto.risk_score.toFixed(2); const cause = ["noise","interfering_speech","speaker_reverb","speaker_loudness","packet_loss","codec_degradation"].sort((a,b) => data.tyto[b]-data.tyto[a])[0]; $("tytoDetail").textContent = `highest: ${cause.replaceAll("_"," ")}`; } else $("tytoMetric").classList.add("hidden");
    $("results").classList.remove("hidden", "live"); $("rawModel").textContent = "Gemini 3.5 Transcribe"; $("resultEyebrow").textContent = "Result"; $("resultTitle").textContent = "Side by side"; $("results").scrollIntoView({ behavior:"smooth", block:"start" });
  } catch (error) { $("inputCard").classList.remove("hidden"); showError(error.message); }
  finally { clearInterval(clock); $("processing").classList.add("hidden"); }
}

$("recordButton").addEventListener("click", async () => { try { recorder ? await stopLiveComparison() : await startLiveComparison(); } catch (error) { showError(`Microphone error: ${error.message}`); if (recorder) { await releaseMicrophone(recorder); recorder = null; } resetLiveUi(); } });
$("fileInput").addEventListener("change", (event) => { const file = event.target.files[0]; if (file) setFile(file, file.name); });
const upload = document.querySelector(".upload-zone");
for (const event of ["dragenter","dragover"]) upload.addEventListener(event, (e) => { e.preventDefault(); upload.classList.add("drag"); });
for (const event of ["dragleave","drop"]) upload.addEventListener(event, (e) => { e.preventDefault(); upload.classList.remove("drag"); });
upload.addEventListener("drop", (event) => { const file = event.dataTransfer.files[0]; if (file) setFile(file, file.name); });
upload.addEventListener("keydown", (event) => { if (["Enter", " "].includes(event.key)) { event.preventDefault(); $("fileInput").click(); } });
// the whole record zone is clickable, matching the upload label; ignore clicks that already hit the button
$("recordZone").addEventListener("click", (event) => { if (!event.target.closest("#recordButton") && !$("recordButton").disabled) $("recordButton").click(); });
$("clearFile").addEventListener("click", clearFile); $("runButton").addEventListener("click", runComparison); $("dismissError").addEventListener("click", () => $("errorBox").classList.add("hidden"));
$("enhancementLevel").addEventListener("input", (event) => $("levelValue").textContent = event.target.value);
$("newTest").addEventListener("click", async () => { if (recorder) await stopLiveComparison(); $("results").classList.add("hidden"); $("results").classList.remove("live"); $("resultEyebrow").textContent = "Result"; $("resultTitle").textContent = "Side by side"; $("inputCard").classList.remove("hidden"); $("inputCard").scrollIntoView({behavior:"smooth"}); });
function chooseListen(path) { const from = activeListenPath === "raw" ? $("rawAudio") : $("enhancedAudio"), to = path === "raw" ? $("rawAudio") : $("enhancedAudio"), playing = !from.paused; to.currentTime = from.currentTime; from.pause(); if (playing) to.play(); activeListenPath = path; $("listenRaw").classList.toggle("active", path === "raw"); $("listenEnhanced").classList.toggle("active", path === "enhanced"); }
$("listenRaw").addEventListener("click", () => chooseListen("raw")); $("listenEnhanced").addEventListener("click", () => chooseListen("enhanced"));
$("abPlay").addEventListener("click", () => { const audio = activeListenPath === "raw" ? $("rawAudio") : $("enhancedAudio"); if (audio.paused) { audio.play(); $("abPlay").textContent = "Ⅱ Pause"; } else { audio.pause(); $("abPlay").textContent = "▶ Play"; } });
loadStatus();
if ("serviceWorker" in navigator) navigator.serviceWorker.getRegistrations().then((registrations) => registrations.forEach((registration) => registration.unregister()));
