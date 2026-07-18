// Sidebar toggle for mobile + highlight the current page in the TOC.
// No external input, no eval; pure DOM. Runs after DOMContentLoaded.
(function () {
  "use strict";

  function ready(fn) {
    if (document.readyState !== "loading") fn();
    else document.addEventListener("DOMContentLoaded", fn);
  }

  ready(function () {
    var nav = document.querySelector("nav.toc");

    // mobile hamburger
    var btn = document.querySelector(".menu-btn");
    if (btn && nav) {
      btn.addEventListener("click", function () {
        nav.classList.toggle("open");
      });
      // close when a link is tapped
      nav.addEventListener("click", function (e) {
        if (e.target.tagName === "A") nav.classList.remove("open");
      });
    }

    // mark current page link (compare by file name only)
    if (nav) {
      var here = location.pathname.split("/").pop() || "index.html";
      var links = nav.querySelectorAll("a");
      for (var i = 0; i < links.length; i++) {
        var href = (links[i].getAttribute("href") || "").split("/").pop();
        if (href === here) links[i].classList.add("current");
      }
    }
  });
})();
