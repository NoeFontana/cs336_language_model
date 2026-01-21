window.MathJax = {
    tex: {
        inlineMath: [["\\(", "\\)"]],
        displayMath: [["\\[", "\\]"]],
        processEscapes: true,
        processEnvironments: true
    },
    options: {
        ignoreHtmlClass: ".*|",
        processHtmlClass: "arithmatex"
    }
};

document.addEventListener("DOMContentLoaded", () => {
    if (typeof MathJax !== "undefined" && MathJax.typesetPromise) {
        MathJax.typesetPromise();
    }
});

/* Support for mkdocs-material's instant loading if it gets enabled later */
if (typeof document.subscribe === "function") {
    document.subscribe(() => {
        if (typeof MathJax !== "undefined" && MathJax.typesetPromise) {
            MathJax.typesetPromise();
        }
    });
}
