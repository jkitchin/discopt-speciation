window.MathJax = {
  tex: {
    // \( \[ from arithmatex (Markdown pages); $ $$ from notebook markdown cells
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    // ignore elements with a class; unclassed children (the <p> in a notebook
    // cell) inherit "process" from a processHtmlClass ancestor
    ignoreHtmlClass: ".+",
    processHtmlClass: "arithmatex|jp-RenderedMarkdown"
  }
};

document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
