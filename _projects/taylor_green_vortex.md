---
layout: page
title: Taylor-Green Vortex
description: Simulation of the Taylor-Green vortex using JAX.
importance: 2
category: work
github: https://github.com/khilavm/JAX-taylor_green_vortex
---

A simulation of the Taylor-Green vortex, a canonical benchmark problem in fluid dynamics, implemented in JAX using GPUs.

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
</style>

<a class="tgv-demo" href="https://molab.marimo.io/notebooks/nb_ftnExDz99uEZH7NpY9WsQv/app" target="_blank" rel="noopener noreferrer" aria-label="Open the interactive Taylor-Green vortex demo on molab (opens in a new tab)">
  <span class="tgv-demo__play" aria-hidden="true">&#9654;</span>
  <span class="tgv-demo__label">Open interactive demo &#8599;</span>
  <span class="tgv-demo__sublabel">JAX pseudospectral solver &middot; runs live on a molab GPU</span>
</a>

<p class="tgv-demo__hint">Opens in a new tab. Adjust the sliders, then click <strong>Run simulation</strong>; the GPU kernel takes a few seconds to warm up on first run.</p>

<a href="https://github.com/khilavm/JAX-taylor_green_vortex" target="_blank" rel="noopener noreferrer" class="btn btn-sm z-depth-0" role="button">View Jupyter code
