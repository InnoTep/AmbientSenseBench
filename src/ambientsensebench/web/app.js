const scenarioSelect = document.getElementById("scenario");
const seedInput = document.getElementById("seed");
const generateButton = document.getElementById("generate");
const statusText = document.getElementById("status");
const summary = document.getElementById("summary");
const visuals = document.getElementById("visuals");
const featureSelect = document.getElementById("feature");
const dayInput = document.getElementById("day");
const dayOutput = document.getElementById("day-output");
const severityChart = document.getElementById("severity-chart");
const featureChart = document.getElementById("feature-chart");

let currentPayload = null;

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
  const height = 270;
  const padding = { top: 20, right: 22, bottom: 34, left: 52 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const [minimum, maximum] = options.range || numericRange(values);
  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  if (options.trainingDays) {
    svg.append(svgElement("rect", {
      x: padding.left,
      y: padding.top,
      width: (options.trainingDays / (values.length - 1)) * plotWidth,
      height: plotHeight,
      class: "training-band",
    }));
  }

  if (options.changeBands) {
    options.changeBands.forEach(([start, end]) => {
      svg.append(svgElement("rect", {
        x: padding.left + (start / (values.length - 1)) * plotWidth,
        y: padding.top,
        width: ((end - start + 1) / (values.length - 1)) * plotWidth,
        height: plotHeight,
        class: "change-band",
      }));
    });
  }

  for (let index = 0; index <= 4; index += 1) {
    const y = padding.top + (index / 4) * plotHeight;
    const value = maximum - ((maximum - minimum) * index) / 4;
    svg.append(svgElement("line", { x1: padding.left, y1: y, x2: width - padding.right, y2: y, class: "grid" }));
    const label = svgElement("text", { x: padding.left - 8, y: y + 4, "text-anchor": "end" });
    label.textContent = value.toFixed(options.decimals ?? 2);
    svg.append(label);
  }

  const line = values.map((value, index) => {
    const x = padding.left + (index / (values.length - 1)) * plotWidth;
    const y = padding.top + plotHeight - scale(value, minimum, maximum, 0, plotHeight);
    return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(" ");
  svg.append(svgElement("path", { d: line, class: options.className }));

  [0, 90, 180, 270, 364].filter((value) => value < values.length).forEach((index) => {
    const x = padding.left + (index / (values.length - 1)) * plotWidth;
    svg.append(svgElement("line", { x1: x, y1: padding.top + plotHeight, x2: x, y2: padding.top + plotHeight + 5, class: "axis" }));
    const label = svgElement("text", { x, y: height - 10, "text-anchor": "middle" });
    label.textContent = `Day ${index + 1}`;
    svg.append(label);
  });

  if (options.alerts) {
    options.alerts.forEach((isAlert, index) => {
      if (!isAlert) return;
      const x = padding.left + (index / (values.length - 1)) * plotWidth;
      const y = padding.top + plotHeight - scale(values[index], minimum, maximum, 0, plotHeight);
      svg.append(svgElement("circle", { cx: x, cy: y, r: 3.5, class: "alert" }));
    });
  }
}

function contiguousChangeBands(records) {
  const bands = [];
  let start = null;
  records.forEach((record, index) => {
    if (record.label === "distress" && start === null) start = index;
    if (record.label !== "distress" && start !== null) {
      bands.push([start, index - 1]);
      start = null;
    }
  });
  if (start !== null) bands.push([start, records.length - 1]);
  return bands;
}

function renderDayDetail() {
  if (!currentPayload) return;
  const record = currentPayload.records[Number(dayInput.value) - 1];
  const feature = featureSelect.value;
  const score = record.anomaly_score === null ? "training baseline" : record.anomaly_score.toFixed(4);
  dayOutput.textContent = `${record.date} · ${record.label} · ${featureLabels[feature]}: ${record[feature]} · score: ${score}${record.alarm ? " · alert" : ""}`;
}

function renderCharts() {
  const records = currentPayload.records;
  const feature = featureSelect.value;
  drawChart(severityChart, records.map((record) => record.severity), {
    range: [0, 1],
    className: "severity",
    decimals: 1,
    trainingDays: currentPayload.metrics.training_days,
    changeBands: contiguousChangeBands(records),
    alerts: records.map((record) => record.alarm),
  });
  drawChart(featureChart, records.map((record) => Number(record[feature])), {
    className: "feature-line",
    decimals: 2,
    trainingDays: currentPayload.metrics.training_days,
    changeBands: contiguousChangeBands(records),
  });
  renderDayDetail();
}

function renderPayload(payload) {
  currentPayload = payload;
  featureSelect.replaceChildren();
  payload.features.forEach((feature) => {
    const option = document.createElement("option");
    option.value = feature;
    option.textContent = featureLabels[feature];
    featureSelect.append(option);
  });
  summary.hidden = false;
  visuals.hidden = false;
  document.getElementById("change-days").textContent = payload.records.filter((record) => record.label === "distress").length;
  document.getElementById("alerts").textContent = payload.metrics.alerts;
  document.getElementById("threshold").textContent = payload.metrics.threshold;
  document.getElementById("scenario-description").textContent = payload.description;
  dayInput.value = 1;
  statusText.textContent = `${payload.scenario} generated with seed ${payload.seed}. The shaded region marks simulated routine change; orange points are fixed-threshold alerts.`;
  renderCharts();
}

async function loadScenarios() {
  const response = await fetch("/api/scenarios");
  const payload = await response.json();
  Object.entries(payload.scenarios).forEach(([id, description]) => {
    const option = document.createElement("option");
    option.value = id;
    option.textContent = `${id} — ${description}`;
    scenarioSelect.append(option);
  });
}

generateButton.addEventListener("click", async () => {
  const seed = Number(seedInput.value);
  if (!Number.isInteger(seed)) {
    statusText.textContent = "Seed must be an integer.";
    return;
  }
  generateButton.disabled = true;
  statusText.textContent = "Generating scenario and scoring alerts…";
  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario: scenarioSelect.value, seed }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Scenario generation failed");
    renderPayload(payload);
  } catch (error) {
    statusText.textContent = error.message;
  } finally {
    generateButton.disabled = false;
  }
});

featureSelect.addEventListener("change", renderCharts);
dayInput.addEventListener("input", renderDayDetail);
loadScenarios().catch((error) => { statusText.textContent = error.message; });
