(function () {
  const root = document.documentElement;
  const stored = localStorage.getItem("sp-theme");
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const initial = stored || (prefersDark ? "dark" : "light");
  root.setAttribute("data-theme", initial);

  window.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("themeToggle");
    if (btn) {
      btn.addEventListener("click", () => {
        const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
        root.setAttribute("data-theme", next);
        localStorage.setItem("sp-theme", next);
      });
    }

    // mobile sidebar
    const menuBtn = document.getElementById("menuToggle");
    const sidebar = document.getElementById("sidebar");
    if (menuBtn && sidebar) {
      menuBtn.addEventListener("click", () => sidebar.classList.toggle("open"));
    }

    // count-up animation for stat numbers with data-countup
    document.querySelectorAll("[data-countup]").forEach((el) => {
      const target = parseFloat(el.dataset.countup);
      const suffix = el.dataset.suffix || "";
      const decimals = el.dataset.decimals ? parseInt(el.dataset.decimals) : 0;
      const duration = 900;
      const start = performance.now();
      function tick(now) {
        const p = Math.min((now - start) / duration, 1);
        const eased = 1 - Math.pow(1 - p, 3);
        const val = target * eased;
        el.textContent = val.toLocaleString(undefined, {
          minimumFractionDigits: decimals,
          maximumFractionDigits: decimals,
        }) + suffix;
        if (p < 1) requestAnimationFrame(tick);
      }
      requestAnimationFrame(tick);
    });
  });
})();
