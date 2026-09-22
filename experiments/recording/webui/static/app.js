const POLL_MS = 700;

function fmtHMS(seconds) {
  if (seconds === null || seconds === undefined || isNaN(seconds)) return "--:--:--";
  seconds = Math.max(0, Math.round(seconds));
  const h = String(Math.floor(seconds / 3600)).padStart(2, "0");
  const m = String(Math.floor((seconds % 3600) / 60)).padStart(2, "0");
  const s = String(seconds % 60).padStart(2, "0");
  return `${h}:${m}:${s}`;
}

function pct(n, d) {
  if (!d) return 0;
  return Math.max(0, Math.min(100, (100 * n) / d));
}

function statusClass(status) {
  if (!status || status === "idle") return "idle";
  if (status === "valid") return "valid";
  if (["corrupted", "audio_overflow", "interrupted"].includes(status)) return "bad";
  return "warn";
}

function setLamp(el, lit, color) {
  el.classList.remove("lit-red", "lit-amber", "lit-green");
  if (lit) el.classList.add(color);
}

async function poll() {
  const dot = document.getElementById("conn-dot");
  const connText = document.getElementById("conn-text");
  try {
    const res = await fetch("/api/status", { cache: "no-store" });
    const d = await res.json();

    dot.classList.remove("off");
    dot.classList.add("on");
    connText.textContent = d.found ? "LIVE" : "NO SESSION";

    if (!d.found) return;

    document.getElementById("participant").textContent = d.participant ?? "--";
    document.getElementById("scenario").textContent = d.scenario ?? "--";
    document.getElementById("session").textContent = d.session ?? "--";

    const overallPct = pct(d.overall_completed, d.overall_total);
    document.getElementById("overall-dial").style.setProperty("--pct", overallPct.toFixed(1));
    document.getElementById("overall-pct").textContent = overallPct.toFixed(1) + "%";
    document.getElementById("overall-frac").textContent = `${d.overall_completed} / ${d.overall_total}`;

    const classPct = pct(d.class_completed, d.class_total);
    document.getElementById("class-dial").style.setProperty("--pct", classPct.toFixed(1));
    document.getElementById("class-pct").textContent = classPct.toFixed(1) + "%";
    document.getElementById("class-frac").textContent = `${d.class_completed} / ${d.class_total}`;

    const glyph = document.getElementById("target-glyph");
    if (d.current_label) {
      glyph.textContent = d.current_label;
      glyph.classList.remove("idle");
    } else {
      glyph.textContent = "-";
      glyph.classList.add("idle");
    }
    document.getElementById("next-glyph").textContent = d.next_label ?? "-";

    const strip = document.getElementById("status-strip");
    const label = d.is_complete ? "COMPLETE" : (d.current_status || "idle").toUpperCase();
    strip.textContent = label;
    strip.className = "status-strip " + (d.is_complete ? "valid" : statusClass(d.current_status));

    document.getElementById("elapsed").textContent = fmtHMS(d.elapsed_s);
    document.getElementById("remaining").textContent = d.is_complete ? "00:00:00" : fmtHMS(d.remaining_s);
    document.getElementById("seed-strategy").textContent =
      `${d.random_seed ?? "--"} / ${d.randomization_strategy ?? "--"}`;

    document.getElementById("peak-bar").style.width = ((d.last_peak ?? 0) * 100).toFixed(1) + "%";
    document.getElementById("rms-bar").style.width = ((d.last_rms ?? 0) * 100).toFixed(1) + "%";
    document.getElementById("clip-bar").style.width = ((d.last_clipping_ratio ?? 0) * 100).toFixed(1) + "%";

    const w = d.warnings || {};
    setLamp(document.getElementById("lamp-overflow"), w.overflow, "lit-red");
    setLamp(document.getElementById("lamp-clipping"), w.clipping, "lit-amber");
    setLamp(document.getElementById("lamp-silence"), w.silence, "lit-amber");
    setLamp(document.getElementById("lamp-complete"), d.is_complete, "lit-green");
  } catch (err) {
    dot.classList.remove("on");
    dot.classList.add("off");
    connText.textContent = "OFFLINE";
  } finally {
    setTimeout(poll, POLL_MS);
  }
}

poll();
