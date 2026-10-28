// Apply the saved theme before the stylesheet can paint. Storage can be blocked.
(() => {
  let theme = "system";
  try {
    const value = localStorage.getItem("localmelo-console.theme");
    if (["system", "light", "dark"].includes(value)) theme = value;
  } catch { /* The system theme is the safe fallback. */ }
  document.documentElement.dataset.theme = theme;
})();
