document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("[data-lang-button]").forEach((button) => {
    button.addEventListener("click", () => {
      const lang = button.getAttribute("data-lang-button");
      document.body.classList.toggle("lang-en", lang === "en");
      document.body.classList.toggle("lang-zh", lang === "zh");
      document.querySelectorAll("[data-lang-button]").forEach((item) => {
        item.classList.toggle("active", item === button);
      });
    });
  });
});
