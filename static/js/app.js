/**
 * app.js — Diabetic Retinopathy Prediction System
 * Minimal JavaScript for UI interactions
 */

// ─── Sidebar Toggle (mobile) ──────────────────────────────────────────────────
const menuToggle = document.getElementById("menuToggle");
const sidebar    = document.getElementById("sidebar");

if (menuToggle && sidebar) {
  menuToggle.addEventListener("click", () => {
    sidebar.classList.toggle("open");
  });

  // Close sidebar when clicking outside on mobile
  document.addEventListener("click", (e) => {
    if (window.innerWidth <= 900 &&
        !sidebar.contains(e.target) &&
        e.target !== menuToggle) {
      sidebar.classList.remove("open");
    }
  });
}

// ─── Auto-dismiss Flash Messages ─────────────────────────────────────────────
document.querySelectorAll(".flash").forEach(el => {
  setTimeout(() => {
    el.style.transition = "opacity 0.5s ease";
    el.style.opacity = "0";
    setTimeout(() => el.remove(), 500);
  }, 4000);
});

// ─── Animate confidence bars on result page ───────────────────────────────────
window.addEventListener("load", () => {
  // Trigger CSS transitions on probability bars
  document.querySelectorAll(".prob-fill, .conf-bar").forEach(bar => {
    const w = bar.style.width;
    bar.style.width = "0%";
    requestAnimationFrame(() => {
      setTimeout(() => { bar.style.width = w; }, 80);
    });
  });

  // Animate mini confidence fills in tables
  document.querySelectorAll(".mini-conf-fill").forEach(bar => {
    const w = bar.style.width;
    bar.style.width = "0%";
    requestAnimationFrame(() => {
      setTimeout(() => { bar.style.width = w; }, 100);
    });
  });
});

// ─── Stat card counter animation ─────────────────────────────────────────────
function animateCount(el, target, duration = 1200) {
  let start = 0;
  const step = target / (duration / 16);
  const timer = setInterval(() => {
    start += step;
    if (start >= target) { start = target; clearInterval(timer); }
    // Preserve % suffix if present
    const suffix = el.dataset.suffix || "";
    el.textContent = Math.floor(start) + suffix;
  }, 16);
}

document.querySelectorAll(".stat-value").forEach(el => {
  const raw = el.textContent.trim();
  const num = parseFloat(raw.replace("%", ""));
  if (!isNaN(num) && num > 0) {
    const suffix = raw.includes("%") ? "%" : "";
    el.dataset.suffix = suffix;
    el.textContent = "0" + suffix;
    animateCount(el, num);
  }
});

// ─── Table row hover highlight ────────────────────────────────────────────────
document.querySelectorAll(".data-table tbody tr").forEach(row => {
  row.addEventListener("mouseenter", () => {
    row.style.background = "rgba(30,136,229,0.04)";
  });
  row.addEventListener("mouseleave", () => {
    row.style.background = "";
  });
});

// ─── Confirm delete dialogs ───────────────────────────────────────────────────
document.querySelectorAll("form[data-confirm]").forEach(form => {
  form.addEventListener("submit", e => {
    if (!confirm(form.dataset.confirm)) e.preventDefault();
  });
});

// ─── Set current date in topbar if not set by Jinja ──────────────────────────
const dateBadge = document.querySelector(".date-badge");
if (dateBadge && dateBadge.textContent.trim() === "") {
  const now = new Date();
  dateBadge.textContent = now.toLocaleDateString("en-IN", {
    day: "2-digit", month: "short", year: "numeric"
  });
}
