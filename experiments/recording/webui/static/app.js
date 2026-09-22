const POLL_MS = 700;

function fmtHMS(seconds) {
  if (seconds === null || seconds === undefined || isNaN(seconds)) return "--:--:--";
  seconds = Math.max(0, Math.round(seconds));
  const h = String(Math.floor(seconds / 3600)).padStart(2, "0");
  const m = String(Math.floor((seconds % 3600) / 60)).padStart(2, "0");
  const s = String(seconds % 60).padStart(2, "0");
  return `${h}:${m}:${s}`;
}

async function poll() {
  try {
    const res = await fetch("/api/status", { cache: "no-store" });
    const d = await res.json();
    if (!d.found) {
      document.getElementById("status-strip").textContent = "no session";
      return;
    }

    document.getElementById("participant").textContent = d.participant ?? "--";
    document.getElementById("scenario").textContent = d.scenario ?? "--";
    document.getElementById("session").textContent = d.session ?? "--";

    const glyph = document.getElementById("target-glyph");
    if (d.current_label) {
      glyph.textContent = d.current_label;
      glyph.classList.remove("idle");
    } else {
      glyph.textContent = "-";
      glyph.classList.add("idle");
    }

    document.getElementById("status-strip").textContent = d.is_complete
      ? "complete"
      : (d.current_status || "idle");
    document.getElementById("overall-frac").textContent = `${d.overall_completed} / ${d.overall_total}`;
    document.getElementById("next-glyph").textContent = d.next_label ?? "-";
    document.getElementById("elapsed").textContent = fmtHMS(d.elapsed_s);
    document.getElementById("remaining").textContent = d.is_complete ? "00:00:00" : fmtHMS(d.remaining_s);
  } catch (err) {
    document.getElementById("status-strip").textContent = "offline";
  } finally {
    setTimeout(poll, POLL_MS);
  }
}

poll();
