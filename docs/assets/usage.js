// Copy buttons for the terminal blocks. Without this script (or without the
// clipboard) the buttons stay hidden and the blocks still read and select.
(function () {
  if (!navigator.clipboard) return;
  document.querySelectorAll("figure.usage[data-copy]").forEach(function (figure) {
    var button = figure.querySelector(".usage-copy");
    if (!button) return;
    var label = button.textContent;
    button.hidden = false;
    button.addEventListener("click", function () {
      navigator.clipboard.writeText(figure.dataset.copy).then(
        function () { button.textContent = button.dataset.done; },
        function () { button.textContent = button.dataset.fail; }
      ).then(function () {
        setTimeout(function () { button.textContent = label; }, 1800);
      });
    });
  });
})();
