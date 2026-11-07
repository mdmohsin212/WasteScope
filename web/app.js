"use strict";

const $ = (id) => document.getElementById(id);
let selected = null,
  busy = false,
  timer = null;

function remember(id) {
  try {
    id
      ? localStorage.setItem("wastescope-job", id)
      : localStorage.removeItem("wastescope-job");
  } catch (_) {}
}

function setBusy(value) {
  busy = value;
  $("file").disabled = value;
  $("submit").disabled = value || !selected;
  $("submit").firstChild.textContent = value
    ? "Processing media "
    : "Analyze media ";
}

function status(title, message, error = false) {
  $("job-status").hidden = false;
  $("job-status").classList.toggle("error", error);
  $("state").textContent = title;
  $("status-message").textContent = message;
}

async function request(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    const error = new Error(
      typeof data.detail === "string"
        ? data.detail
        : `Request failed (${response.status}).`,
    );
    error.code = response.status;
    throw error;
  }
  return data;
}

function choose(file) {
  if (busy || !file) return;

  const suffix = file.name.split(".").pop().toLowerCase();
  const supported = ["jpg", "jpeg", "png", "webp", "mp4", "mov", "avi", "mkv"];

  if (
    !supported.includes(suffix) ||
    !file.size ||
    file.size > 200 * 1024 * 1024
  ) {
    selected = null;
    $("selection").textContent = "No valid file selected";
    status(
      "Check your file",
      "Choose a supported, nonempty image or video up to 200 MiB.",
      true,
    );
  } else {
    selected = file;
    $("selection").textContent =
      `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MiB`;
    $("job-status").hidden = true;
  }
  setBusy(false);
}

function metric(value, label) {
  const item = document.createElement("div");
  item.className = "metric";

  const number = document.createElement("strong");
  const caption = document.createElement("span");
  number.textContent = value;
  caption.textContent = label;
  item.append(number, caption);
  $("metrics").append(item);
}

function showResult(job, summary) {
  const video = job.mode === "video";
  $("empty").hidden = true;
  $("result").hidden = false;
  $("result-name").textContent = job.filename;
  $("result-badge").textContent = "Analysis complete";

  const name = video ? "annotated.mp4" : "annotated.png";
  const media = video ? $("video") : $("image");
  media.src = `${job.files[name]}?inline=true`;
  media.hidden = false;

  $("metrics").replaceChildren();
  const counts = video ? summary.final_counts : summary.detection_counts;

  metric(
    Object.values(counts).reduce((a, b) => a + b, 0),
    video ? "Unique containers" : "Container detections",
  );
  metric(
    video
      ? summary.frames_with_overflow_predictions
      : summary.overflow_instances,
    video ? "Frames with overflow predictions" : "Overflow masks",
  );
  metric(
    `${Number(summary.end_to_end_seconds).toFixed(1)}s`,
    "Processing time",
  );

  $("count-note").textContent = video
    ? `${summary.frames} frames processed. Counts are cumulative unique tracks after temporal filtering.`
    : "Container counts apply to this image.";

  $("counts").replaceChildren();
  for (const [name, count] of Object.entries(counts)) {
    const row = document.createElement("div");
    const label = document.createElement("dt");
    const value = document.createElement("dd");
    label.textContent = name.replace(/^container_/, "").replaceAll("_", " ");
    value.textContent = count;
    row.append(label, value);
    $("counts").append(row);
  }

  $("downloads").replaceChildren();
  for (const [name, url] of Object.entries(job.files)) {
    const link = document.createElement("a");
    link.href = url;
    link.textContent = name;
    $("downloads").append(link);
  }
}

function resetResult() {
  $("video").pause();
  for (const id of ["image", "video"]) {
    $(id).hidden = true;
    $(id).removeAttribute("src");
  }
  $("video").load();
  $("result").hidden = true;
  $("empty").hidden = false;
  $("frame-progress").textContent = "";
  $("progress").removeAttribute("value");
}

async function poll(id) {
  try {
    const job = await request(`/jobs/${encodeURIComponent(id)}`);
    const progress = job.progress || {};

    if (["failed", "interrupted"].includes(job.state)) {
      status(
        "Analysis stopped",
        job.error || "The job could not finish.",
        true,
      );
      $("progress").value = 0;
      $("result-badge").textContent = "Not completed";
      remember(null);
      setBusy(false);
      return;
    }

    if (job.state === "complete") {
      const summary = await request(job.result_url);
      showResult(job, summary);
      $("progress").max = 1;
      $("progress").value = 1;
      $("frame-progress").textContent =
        `${summary.frames} frame${summary.frames === 1 ? "" : "s"}`;
      status("Complete", "Your preview and downloads are ready.");
      setBusy(false);
      return;
    }

    const frames = progress.frames_processed || 0;
    const total = progress.total_frames;

    $("frame-progress").textContent = total
      ? `${frames} / ${total} frames`
      : `${frames} frames`;

    if (total) {
      $("progress").max = total;
      $("progress").value = frames;
    } else {
      $("progress").removeAttribute("value");
    }

    const rendering = progress.phase === "rendering";
    status(
      job.state === "queued"
        ? "Waiting in queue"
        : rendering
          ? "Preparing your video"
          : "Analyzing media",
      job.state === "queued"
        ? "Your job will start when the worker is available."
        : rendering
          ? "Adding final track IDs, counts and masks to the video."
          : "Finding containers and overflow. Larger videos take longer.",
    );
    $("result-badge").textContent = "Processing";
  } catch (error) {
    if (error.code === 404) {
      status(
        "Run unavailable",
        "This saved job no longer exists. Choose a file to start again.",
        true,
      );
      remember(null);
      setBusy(false);
      return;
    }
    status("Reconnecting", "Unable to read progress. Retrying automatically…");
  }

  timer = setTimeout(() => poll(id), 2000);
}

$("file").addEventListener("change", (event) => choose(event.target.files[0]));

for (const name of ["dragenter", "dragover", "dragleave", "drop"]) {
  $("dropzone").addEventListener(name, (event) => {
    event.preventDefault();
    $("dropzone").classList.toggle(
      "dragging",
      !busy && ["dragenter", "dragover"].includes(name),
    );
    if (name === "drop") choose(event.dataTransfer.files[0]);
  });
}

$("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy || !selected) return;

  clearTimeout(timer);
  resetResult();
  setBusy(true);
  status("Uploading", "Sending your file for analysis…");
  $("result-badge").textContent = "Uploading";

  const body = new FormData();
  body.append("file", selected);

  try {
    const job = await request("/jobs", { method: "POST", body });
    remember(job.job_id);
    await poll(job.job_id);
  } catch (error) {
    status("Upload failed", error.message, true);
    $("progress").value = 0;
    $("result-badge").textContent = "Awaiting upload";
    setBusy(false);
  }
});

let saved;
try {
  saved = localStorage.getItem("wastescope-job");
} catch (_) {}

if (saved && /^[0-9a-f]{32}$/.test(saved)) {
  setBusy(true);
  status("Restoring your last run", "Checking its latest status…");
  poll(saved);
}
