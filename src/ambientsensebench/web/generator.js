"use strict";

const featureLabels = {
  wake_time: "Wake time (hour)",
  sleep_hours: "Sleep duration (hours)",
  meal_count: "Meal events",
  social_proxy: "Door events",
  night_activity: "Night activity",
  mobility_score: "Mobility score",
  kitchen_activity_score: "Kitchen activity score",
  room_transition_entropy: "Transition entropy",
  evening_routine_consistency: "Evening consistency",
};

const el = (id) => document.getElementById(id);
let defaults = null;
let currentPayload = null;

/* ------------------------------------------------------------------ SVG charts */

function svgElement(name, attributes = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
  return element;
}

function scale(value, minimum, maximum, low, high) {
  if (maximum === minimum) return (low + high) / 2;
  return low + ((value - minimum) / (maximum - minimum)) * (high - low);
}

function numericRange(values, fallback = [0, 1]) {
  const finite = values.filter(Number.isFinite);
  if (!finite.length) return fallback;
  const minimum = Math.min(...finite);
  const maximum = Math.max(...finite);
  const padding = minimum === maximum ? 1 : (maximum - minimum) * 0.08;
  return [minimum - padding, maximum + padding];
}

function drawChart(svg, values, options) {
  const width = 960;
  const height = 260;
  const padding = { top: 18, right: 20, bottom: 32, left: 50 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const [minimum, maximum] = options.range || numericRange(values);
  const span = values.length - 1 || 1;
  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  if (options.trainingDays) {
    svg.append(svgElement("rect", {
      x: padding.left, y: padding.top,
      width: (options.trainingDays / span) * plotWidth, height: plotHeight,
      class: "training-band",
    }));
  }
  if (options.changeBands) {
    options.changeBands.forEach(([start, end]) => {
      svg.append(svgElement("rect", {
        x: padding.left + (start / span) * plotWidth, y: padding.top,
        width: ((end - start + 1) / span) * plotWidth, height: plotHeight,
        class: "change-band",
      }));
    });
  }
  for (let index = 0; index <= 4; index += 1) {
    const y = padding.top + (index / 4) * plotHeight;
    const value = maximum - ((maximum - minimum) * index) / 4;
    svg.append(svgElement("line", { x1: padding.left, y1: y, x2: width - padding.right, y2: y, class: "grid-line" }));
    const label = svgElement("text", { x: padding.left - 8, y: y + 4, "text-anchor": "end" });
    label.textContent = value.toFixed(options.decimals ?? 2);
    svg.append(label);
  }
  const line = values.map((value, index) => {
    const x = padding.left + (index / span) * plotWidth;
    const y = padding.top + plotHeight - scale(value, minimum, maximum, 0, plotHeight);
    return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(" ");
  svg.append(svgElement("path", { d: line, class: options.className }));

  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(f * span));
  [...new Set(ticks)].forEach((index) => {
    const x = padding.left + (index / span) * plotWidth;
    svg.append(svgElement("line", { x1: x, y1: padding.top + plotHeight, x2: x, y2: padding.top + plotHeight + 5, class: "axis" }));
    const label = svgElement("text", { x, y: height - 9, "text-anchor": "middle" });
    label.textContent = `Day ${index + 1}`;
    svg.append(label);
  });

  if (options.alerts) {
    options.alerts.forEach((isAlert, index) => {
      if (!isAlert) return;
      const x = padding.left + (index / span) * plotWidth;
      const y = padding.top + plotHeight - scale(values[index], minimum, maximum, 0, plotHeight);
      svg.append(svgElement("circle", { cx: x, cy: y, r: 3.2, class: "alert" }));
    });
  }
}

function contiguousChangeBands(records) {
  const bands = [];
  let start = null;
  records.forEach((record, index) => {
    if (record.label === "distress" && start === null) start = index;
    if (record.label !== "distress" && start !== null) { bands.push([start, index - 1]); start = null; }
  });
  if (start !== null) bands.push([start, records.length - 1]);
  return bands;
}

/* ------------------------------------------------------------------ episodes UI */

function episodeCard(episode) {
  const card = document.createElement("div");
  card.className = "episode-card";
  card.innerHTML = `
    <div class="card-top">
      <strong>Episode</strong>
      <button type="button" class="remove">Remove</button>
    </div>
    <div class="grid">
      <label>Mode
        <select data-key="mode">
          <option value="episode">Episode (rise + recover)</option>
          <option value="monotone">Monotone (rise + stay)</option>
        </select>
      </label>
      <label>Peak severity
        <input type="number" data-key="max_severity" min="0.1" max="1" step="0.05" />
      </label>
      <label>Onset day
        <input type="number" data-key="onset" min="1" step="1" />
      </label>
      <label class="offset-field">Offset day
        <input type="number" data-key="offset" min="1" step="1" />
      </label>
      <label>Smoothness τ (days)
        <input type="number" data-key="tau" min="0.5" max="60" step="0.5" />
      </label>
    </div>`;
  card.querySelector('[data-key="mode"]').value = episode.mode;
  card.querySelector('[data-key="max_severity"]').value = episode.max_severity;
  card.querySelector('[data-key="onset"]').value = episode.onset;
  card.querySelector('[data-key="offset"]').value = episode.offset || episode.onset + 30;
  card.querySelector('[data-key="tau"]').value = episode.tau;

  const toggleOffset = () => {
    const isMonotone = card.querySelector('[data-key="mode"]').value === "monotone";
    card.querySelector(".offset-field").style.display = isMonotone ? "none" : "flex";
  };
  toggleOffset();
  card.querySelector('[data-key="mode"]').addEventListener("change", toggleOffset);
  card.querySelector(".remove").addEventListener("click", () => card.remove());
  return card;
}

function setEpisodes(episodes) {
  const list = el("episode-list");
  list.replaceChildren();
  episodes.forEach((episode) => list.append(episodeCard(episode)));
}

function readEpisodes() {
  return [...document.querySelectorAll(".episode-card")].map((card) => {
    const value = (key) => card.querySelector(`[data-key="${key}"]`).value;
    const mode = value("mode");
    const episode = {
      mode,
      onset: Number(value("onset")),
      tau: Number(value("tau")),
      max_severity: Number(value("max_severity")),
    };
    if (mode === "episode") episode.offset = Number(value("offset"));
    return episode;
  });
}

/* ------------------------------------------------------------------ rendering results */

function stat(value, label) {
  return `<div class="stat"><div class="value">${value}</div><div class="label">${label}</div></div>`;
}

function renderSummary(payload) {
  const m = payload.metrics;
  const dp = payload.dp_epsilon ? `ε = ${payload.dp_epsilon}` : "off";
  el("summary").innerHTML = [
    stat(payload.n_days, "days"),
    stat(payload.distress_days, "change days"),
    stat(payload.event_count.toLocaleString(), "raw events"),
    stat(m.alerts, "IF alerts"),
    stat(m.threshold, "threshold"),
    stat(dp, "privacy"),
    stat(`${payload.generation_seconds}s`, "generation time"),
  ].join("");
}

function renderDownloads(payload) {
  const names = {
    events: "events.csv (raw)",
    features: "daily_features.csv",
    features_dp: "daily_features_dp.csv",
    labels: "scenario_labels.csv",
  };
  el("downloads").innerHTML = Object.entries(payload.downloads)
    .map(([key, file]) => `<a href="/api/download?job=${payload.job_id}&file=${encodeURIComponent(file)}" download>${names[key] || file}</a>`)
    .join("");
}

function renderDayDetail() {
  if (!currentPayload) return;
  const record = currentPayload.records[Number(el("day").value) - 1];
  const feature = el("feature").value;
  el("day-output").textContent = `${record.date} (${record.label})`;
  const score = record.anomaly_score === null ? "training baseline" : record.anomaly_score.toFixed(4);
  el("day-line").textContent =
    `${featureLabels[feature]}: ${record[feature]} · severity ${record.severity} · score ${score}${record.alarm ? " · ALERT" : ""}`;
}

function renderCharts() {
  const records = currentPayload.records;
  const feature = el("feature").value;
  const bands = contiguousChangeBands(records);
  drawChart(el("severity-chart"), records.map((r) => r.severity), {
    range: [0, 1], className: "severity", decimals: 1,
    trainingDays: currentPayload.metrics.training_days, changeBands: bands,
    alerts: records.map((r) => r.alarm),
  });
  drawChart(el("feature-chart"), records.map((r) => Number(r[feature])), {
    className: "feature-line", decimals: 2,
    trainingDays: currentPayload.metrics.training_days, changeBands: bands,
  });
  renderDayDetail();
}

function renderPayload(payload) {
  currentPayload = payload;
  el("results").hidden = false;
  renderSummary(payload);
  renderDownloads(payload);

  const featureSelect = el("feature");
  featureSelect.replaceChildren();
  payload.features.forEach((feature) => {
    const option = document.createElement("option");
    option.value = feature;
    option.textContent = featureLabels[feature];
    featureSelect.append(option);
  });

  const daySlider = el("day");
  daySlider.max = payload.n_days;
  daySlider.value = 1;
  renderCharts();
  el("results").scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ------------------------------------------------------------------ init + submit */

function renderSensors() {
  const container = el("sensor-list");
  container.replaceChildren();
  defaults.sensors.forEach((sensor) => {
    const label = document.createElement("label");
    label.innerHTML = `<input type="checkbox" value="${sensor.id}" checked /> ${sensor.label}`;
    container.append(label);
  });
}

function renderPresets() {
  const select = el("preset");
  select.replaceChildren();
  Object.entries(defaults.presets).forEach(([id, preset]) => {
    const option = document.createElement("option");
    option.value = id;
    option.textContent = `${id} — ${preset.label}`;
    select.append(option);
  });
  select.addEventListener("change", () => {
    setEpisodes(defaults.presets[select.value].episodes);
  });
}

function activeSensors() {
  const boxes = [...document.querySelectorAll("#sensor-list input:checked")];
  const selected = boxes.map((box) => box.value);
  return selected.length === defaults.sensors.length ? null : selected;
}

async function generate() {
  const button = el("generate");
  const status = el("status");
  status.className = "status";
  const body = {
    n_days: Number(el("days").value),
    seed: Number(el("seed").value),
    episodes: readEpisodes(),
    dp_epsilon: el("dp-epsilon").value || null,
    active_sensors: activeSensors(),
  };
  if (!Number.isInteger(body.seed)) { status.textContent = "Seed must be an integer."; status.className = "status error"; return; }

  button.disabled = true;
  status.textContent = "Generating raw events, extracting features and scoring alerts…";
  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Generation failed");
    status.textContent = `Generated ${payload.n_days}-day dataset with ${payload.event_count.toLocaleString()} events (seed ${payload.seed}).`;
    renderPayload(payload);
  } catch (error) {
    status.textContent = error.message;
    status.className = "status error";
  } finally {
    button.disabled = false;
  }
}

async function init() {
  try {
    defaults = await (await fetch("/api/defaults")).json();
  } catch (error) {
    el("status").textContent = "Could not load configuration from the server.";
    el("status").className = "status error";
    return;
  }
  renderSensors();
  renderPresets();
  setEpisodes(defaults.presets.P01.episodes);

  el("add-episode").addEventListener("click", () => {
    el("episode-list").append(episodeCard({ mode: "episode", onset: 120, offset: 160, tau: 3, max_severity: 1.0 }));
  });
  el("generate").addEventListener("click", generate);
  el("feature").addEventListener("change", renderCharts);
  el("day").addEventListener("input", renderDayDetail);
}

init();
