---
layout: page
title: Taylor-Green Vortex
description: Simulations of the Taylor-Green vortex on GPUs.
importance: 1
category: work
github: https://github.com/khilavm/JAX-taylor_green_vortex
---

Simulations of the Taylor-Green vortex, a canonical benchmark problem in fluid dynamics. Below is an $N=64$ run on a molab GPU. Jupyter code for the same run on a colab T4 is hosted on GitHub with a link at the bottom of the page. Both were implemented in JAX. Further below on this page is a run on an NVIDIA H200 using nvmath and CuPy with $N=512$.

<style>
  .tgv-demo {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
    width: 100%;
    height: clamp(420px, 62vh, 600px);
    margin: 1.5rem 0 0.75rem;
    border-radius: 8px;
    overflow: hidden;
    text-align: center;
    text-decoration: none;
    color: #fff !important;
    border: 1px solid rgba(0, 0, 0, 0.12);
    background-color: #0b1021;
    background-image:
      linear-gradient(rgba(11, 16, 33, 0.62), rgba(11, 16, 33, 0.72)),
      url('{{ "/assets/img/taylor_green_vortex.png" | relative_url }}');
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
    transition: filter 0.15s ease;
  }
  html[data-theme="dark"] .tgv-demo {
    border-color: rgba(255, 255, 255, 0.15);
  }
  .tgv-demo:hover {
    color: #fff;
    filter: brightness(1.05);
  }
  .tgv-demo:hover .tgv-demo__play {
    transform: scale(1.08);
    background: var(--global-theme-color, #b509ac);
  }
  .tgv-demo__play {
    width: 64px;
    height: 64px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.6rem;
    line-height: 1;
    background: rgba(0, 0, 0, 0.6);
    border: 2px solid #fff;
    padding-left: 4px;
    transition: transform 0.15s ease, background 0.15s ease;
  }
  .tgv-demo__label,
  .tgv-demo__sublabel {
    color: #fff !important;
    background: rgba(11, 16, 33, 0.8);
    border-radius: 5px;
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.9);
  }
  .tgv-demo__label {
    font-weight: 700;
    font-size: 1.1rem;
    padding: 0.2rem 0.7rem;
  }
  .tgv-demo__sublabel {
    font-size: 0.85rem;
    padding: 0.15rem 0.6rem;
  }
  .tgv-demo__hint {
    font-size: 0.85rem;
    opacity: 0.7;
    margin-bottom: 1.25rem;
  }
  .tgv-links {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
  }
  /* Bootstrap's .btn:hover hardcodes a near-black colour, which disappears in
     dark mode. Follow the theme's own convention for link rows instead. */
  .tgv-links a.btn {
    color: var(--global-text-color);
    border: 1px solid var(--global-text-color);
  }
  .tgv-links a.btn:hover {
    color: var(--global-theme-color);
    border-color: var(--global-theme-color);
  }
  /* The notebook renders inside a same-origin iframe, so its internals cannot be
     styled from here. Only the frame itself is ours. */
  .jupyter-notebook-iframe-container iframe {
    background: #fff;
    border-radius: 6px;
  }
</style>

<a class="tgv-demo" href="https://molab.marimo.io/notebooks/nb_ftnExDz99uEZH7NpY9WsQv/app" target="_blank" rel="noopener noreferrer" aria-label="Open the interactive Taylor-Green vortex demo on molab (opens in a new tab)">
  <span class="tgv-demo__play" aria-hidden="true">&#9654;</span>
  <span class="tgv-demo__label">Open interactive demo &#8599;</span>
  <span class="tgv-demo__sublabel">JAX pseudospectral solver &middot; runs live on a molab GPU</span>
</a>

<p class="tgv-demo__hint">Opens in a new tab. Adjust the sliders, then click <strong>Run simulation</strong>. The GPU kernel takes a few seconds to warm up on first run.</p>

---

## The full simulation notebook

The notebook below is the $N = 512$ production run on an NVIDIA H200 using nvmath and CuPy with the full derivation,
solver, diagnostics and with all output as it was produced. It is a rendered copy. Use the molab demo above to actually run a simulation at lower $N$. GitHub code links are present at the bottom of the page.

{::nomarkdown}
{% assign jupyter_path = "assets/jupyter/taylor_green_vortex.ipynb" | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/taylor_green_vortex.ipynb %}{% endcapture %}
{% if notebook_exists == "true" %}
{% jupyter_notebook jupyter_path %}
{% else %}
<p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}

<script>
  // The plugin sizes the iframe once, on load. This notebook typesets its LaTeX
  // with MathJax *after* that, which grows the document and clips the bottom.
  // Re-measure whenever the framed document changes size.
  (function () {
    var frame = document.querySelector(".jupyter-notebook-iframe-container iframe");
    if (!frame) return;
    function sync() {
      try {
        var doc = frame.contentWindow.document;
        if (!doc || !doc.documentElement) return;
        frame.parentElement.style.paddingBottom = doc.documentElement.scrollHeight + 10 + "px";
      } catch (e) {
        /* cross-origin; nothing we can do */
      }
    }
    frame.addEventListener("load", function () {
      sync();
      try {
        new ResizeObserver(sync).observe(frame.contentWindow.document.documentElement);
      } catch (e) {
        // No ResizeObserver: fall back to a few re-measures while MathJax works.
        [400, 1200, 3000].forEach(function (t) {
          setTimeout(sync, t);
        });
      }
    });
  })();
</script>

---

## Run it yourself

<p class="tgv-links">
  <a href="https://molab.marimo.io/notebooks/nb_ftnExDz99uEZH7NpY9WsQv/app" target="_blank" rel="noopener noreferrer" class="btn btn-sm z-depth-0" role="button">Interactive demo (molab)</a>
  <a href="https://github.com/khilavm/taylor_green_vortex" target="_blank" rel="noopener noreferrer" class="btn btn-sm z-depth-0" role="button">Source on GitHub</a>
</p>

<p class="tgv-demo__hint">The GitHub link points at the repo containing (1) the JAX implementation, which runs at $N = 64$ on a free Colab T4, and (2) the $N = 512$ run presented above, using an H200.</p>
