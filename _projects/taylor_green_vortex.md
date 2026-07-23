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
  /* Resolution-comparison figure grid. This was an HTML table of inline base64
     images in the notebook; the images are now real files, so the theme styles
     the captions and dark mode works without any per-element colours. */
  .tgv-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
    margin: 1.25rem 0;
  }
  .tgv-grid figure {
    margin: 0;
  }
  .tgv-grid img {
    width: 100%;
    height: auto;
    display: block;
    border-radius: 4px;
  }
  .tgv-grid figcaption {
    text-align: center;
    font-size: 0.9rem;
    margin-top: 0.35rem;
  }
  .tgv-grid figcaption span {
    font-size: 0.85em;
    opacity: 0.75;
  }
  @media (max-width: 576px) {
    .tgv-grid {
      grid-template-columns: 1fr;
    }
  }

  /* Markdown emits bare <table> elements, and the theme's only table rule sets
     font-size and padding with no borders at all, so they render as loose
     columns of text. Give them real rules here. Colours come from the theme
     variable / neutral alphas so both colour schemes work. */
  table {
    display: block;
    width: fit-content;
    max-width: 100%;
    overflow-x: auto;
    border-collapse: collapse;
    margin: 1.5rem 0;
  }
  table th,
  table td {
    border: 1px solid var(--global-divider-color);
    padding: 0.45rem 0.8rem;
    text-align: left;
    vertical-align: top;
  }
  table thead th {
    background-color: rgba(128, 128, 128, 0.12);
    font-weight: 700;
    border-bottom-width: 2px;
  }
  table tbody tr:hover {
    background-color: rgba(128, 128, 128, 0.06);
  }

  /* Notebook figures are exported at full plot resolution, so unconstrained
     they span the whole column. Cap them to a readable width and centre them.
     Scoped to p > img so the demo card and the .tgv-grid figures are untouched. */
  p > img {
    display: block;
    max-width: min(100%, 700px);
    height: auto;
    margin: 1.75rem auto;
    border-radius: 4px;
  }

  /* Long-form reading rhythm. The theme's defaults are tuned for short project
     blurbs; this page is a full article, so give it more leading and clearer
     separation between sections. */
  p,
  li {
    line-height: 1.75;
  }
  p {
    margin-bottom: 1.15rem;
  }
  li {
    margin-bottom: 0.35rem;
  }
  h2 {
    margin-top: 3rem;
    margin-bottom: 1rem;
    line-height: 1.3;
  }
  h3 {
    margin-top: 2.25rem;
    margin-bottom: 0.75rem;
    line-height: 1.35;
  }
  h4 {
    margin-top: 1.75rem;
    margin-bottom: 0.6rem;
  }
  /* Code blocks and their adjacent output need breathing room from the prose. */
  div.highlighter-rouge,
  figure.highlight,
  pre {
    margin-top: 1.25rem;
    margin-bottom: 1.25rem;
  }
  /* Display equations sit too tight against surrounding paragraphs. */
  mjx-container[display="true"] {
    margin: 1.5rem 0 !important;
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

A 3-D incompressible Navier-Stokes **pseudospectral** direct numerical simulation of the
Taylor-Green vortex on the triply-periodic domain $[0, 2\pi]^3$, run at $512^3 =
1.34\times10^8$ grid points on an **NVIDIA H200** with **nvmath-python + CuPy**.

| Component | Choice |
|-----------|--------|
| Spatial | Fourier pseudospectral (FFT) |
| Transforms | **R2C / C2R** real transforms (cuFFT via nvmath-python), fp32 |
| Dealiasing | 2/3 rule, i.e., zero modes with $\lvert k_i \rvert > N/3$ |
| Incompressibility | Leray projection in Fourier space |
| Time stepping | $2^{nd}$-order Runge-Kutta with Heun predictor-corrector |
| Resolution | $N = 512$, $Re = 1600$, $\Delta t = 1.25\times10^{-3}$, $T = 20$ |

**Initial condition** (Taylor-Green vortex):

$$u = \sin x \cos y \cos z, \quad v = -\cos x \sin y \cos z, \quad w = 0$$

**This run:** 16 000 RK2 steps at **114 ms/step**, peak dissipation
$\varepsilon = 1.286\times10^{-2}$ at $t = 9.0$ — within ~2% of the reference value
(~0.0126) of Brachet et al. (1983).

---

**Nondimensionalisation**

Starting from the dimensional incompressible Navier-Stokes equations:

$$\frac{\partial \mathbf{u}^*}{\partial t^*} + (\mathbf{u}^* \cdot \nabla)\mathbf{u}^* = -\frac{1}{\rho}\nabla p^* + \nu_1 \nabla^2 \mathbf{u}^*$$

Here, $\mathbf{u}$ is the velocity and $p$ the pressure. The viscosity is defined here as $\nu_1$ as we will use $\nu$ for the final nondimensional form later. We introduce a characteristic velocity $U$ and length scale $L$ and define nondimensional variables $\mathbf{u} = \mathbf{u}^*/U$, $t = t^* U/L$, $p = p^*/\rho U^2$. The equations become:

$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\nabla p + \frac{1}{Re} \nabla^2 \mathbf{u}, \qquad Re = \frac{UL}{\nu_1}$$

For the Taylor-Green vortex the natural scales are $U = 1$ (peak velocity of the initial condition) and $L = 1$ (set by the initial condition), giving $\nu = 1/Re$.

---

**On the numerics.** The physical fields $u, v, w$ are *real*, so their Fourier transforms are Hermitian-symmetric: $\hat u(-\mathbf{k}) = \overline{\hat u(\mathbf{k})}$. Storing and transforming only the non-redundant half spectrum (`rfft`/`irfft`, shape $(N, N, N/2{+}1)$) halves both the memory and the transform work relative to a full complex-to-complex path. This makes $N = 512$ fit and run at this speed. The R2C path was validated mode-by-mode against a full complex-to-complex reference solver and agrees with it to ~$10^{-5}$ relative, and the optimization campaign that led here is summarised at the end of this notebook.

```python
# Requires an NVIDIA GPU + CUDA 12 (Linux). R2C uses cuFFT, no device callbacks needed.
import sys
!{sys.executable} -m ensurepip --upgrade
!{sys.executable} -m pip install -q "nvmath-python[cu12]" cupy-cuda12x matplotlib
```

    Looking in links: /tmp/tmprjsqe6b3
    Processing /tmp/tmprjsqe6b3/pip-25.0.1-py3-none-any.whl
    Installing collected packages: pip
    Successfully installed pip-25.0.1
    
    [notice] A new release of pip is available: 25.0.1 -> 26.1.2
    [notice] To update, run: pip3 install --upgrade pip
    

```python
import numpy as np
import cupy as cp
import nvmath
import nvmath.fft as nvfft
import matplotlib.pyplot as plt
import time

REAL, COMPLEX = cp.float32, cp.complex64          # single precision (fp32)
FWD, INV      = nvfft.FFTDirection.FORWARD, nvfft.FFTDirection.INVERSE
_NAT          = {"result_layout": "natural"}
_name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode()
print("nvmath-python:", getattr(nvmath, "__version__", "?"), "| device:", _name)
```

    nvmath-python: 1.0.0 | device: NVIDIA H200
    

### GPU memory report — check free memory before pushing N

```python
def gpu_mem(label=""):
    free, total = cp.cuda.runtime.memGetInfo()
    used = total - free
    pool = cp.get_default_memory_pool()
    GB = 1024**3
    print(f"[GPU memory] {label}")
    print(f"  device    : {used/GB:6.2f} / {total/GB:6.2f} GB used ({100*used/total:4.1f}%) | {free/GB:6.2f} GB free")
    print(f"  CuPy pool : {pool.used_bytes()/GB:6.2f} GB live, {pool.total_bytes()/GB:6.2f} GB reserved")
    print(f"  ~not in this process's pool (other procs / workspaces / context): {(used - pool.total_bytes())/GB:6.2f} GB")
    return free, total

def gpu_free():
    cp.get_default_memory_pool().free_all_blocks()
    print("freed CuPy pool cached blocks")

gpu_mem("after imports")
```

    [GPU memory] after imports
      device    :   0.51 / 139.80 GB used ( 0.4%) | 139.29 GB free
      CuPy pool :   0.00 GB live,   0.00 GB reserved
      ~not in this process's pool (other procs / workspaces / context):   0.51 GB
    

    (149557477376, 150109880320)

### Parameters

$Re = 1600$ is the standard Taylor-Green benchmark value, chosen so the dissipation peak
near $t \approx 9$ can be compared against reference DNS.

The time step is set by the **advective CFL condition**: an explicit scheme is stable only
if a fluid parcel moves less than one grid cell per step, $\Delta t \lesssim \Delta x /
u_{max}$. Since $\Delta x = 2\pi/N$ and $u_{max} = O(1)$, the stable step scales as
$\Delta t \propto 1/N$ — hence `dt = 0.01 * (64 / N)`, calibrated at $N = 64$. Holding
$\Delta t$ fixed while refining the grid resulted in `NaN`s for high-$Re$ cases in the sweep (Re = 3200).

```python
N        = 512           # push for a finer vorticity figure; R2C halves memory vs C2C
Re       = 1600
nu       = 1.0 / Re
L        = 2.0 * np.pi
dt       = 0.01 * (64 / N)   # CFL
T        = 12.0              # enough to pass the enstrophy peak (~t=9)
n_steps  = int(T / dt)
log_freq = 50

print(f"Grid: {N}^3 = {N**3:,} points   Steps: {n_steps}   Re={Re}, dt={dt:.2e}, T={T}")
```

    Grid: 512^3 = 134,217,728 points   Steps: 9600   Re=1600, dt=1.25e-03, T=12.0
    

### Grid, half-spectrum wavenumbers, dealias mask, Hermitian weights

The wavenumber grid uses `fftfreq` on the two full axes, which returns integers
$0, 1, \ldots, N/2-1, -N/2, \ldots, -1$, and `rfftfreq` on the reduced last axis, which
returns only the non-negative half $0, 1, \ldots, N/2$. The array `RK2` $= k_x^2 + k_y^2 +
k_z^2$ is the spectral Laplacian. `RK2s` is the same with the $\mathbf{k} = 0$ entry set to
1 so the Leray projector can divide by it safely (the mean mode is handled separately).

**Dealiasing.** A pseudospectral code evaluates the quadratic term $\mathbf{u}\cdot\nabla
\mathbf{u}$ in physical space, where it is a pointwise product. A product of two modes $k_1$
and $k_2$ generates the mode $k_1 + k_2$, which can exceed the maximum resolvable wavenumber
$k_{max} = N/2$ and get folded back onto a low wavenumber contaminating the
resolved scales and in a nonlinear cascade, blowing up. The **2/3 rule** zeroes all modes
with $\lvert k_i \rvert > (2/3)k_{max} = N/3$ before every physical-space multiplication, so the
generated sum modes land in the emptied band instead of aliasing back.

**Hermitian Parseval weights.** This is bookkeeping.

The velocity fields are real, and the Fourier transform of a real field obeys
$\hat{u}(-\mathbf{k}) = \overline{\hat{u}(\mathbf{k})}$ so the mode at $-\mathbf{k}$ carries no information that the mode at $+\mathbf{k}$ does not already have. An `rfft` exploits exactly by storing only $k_z \geq 0$ and it discards the $k_z < 0$ half of the cube as redundant. So we get a factor-of-two saving in memory and transform work.

The catch is that Parseval sums such as $E = \frac{1}{2N^6}\sum_{\mathbf{k}} \lvert \hat{\mathbf{u}} \rvert^2$ are defined over the **full** spectrum, so summing only the stored half would give roughly half the right answer. The fix is to count each stored mode with the multiplicity of the full-spectrum modes it represents:

* A mode with $k_z > 0$ represents **two** full-spectrum modes: itself, and its unstored conjugate partner at $-\mathbf{k}$, which has identical $\lvert \hat u \rvert^2$ and identical $k^2$. Weight **2**.
* A mode on the $k_z = 0$ plane has its partner at $(-k_x, -k_y, 0)$, which lies *inside* the stored slab and is therefore already counted on its own. Weight **1**, since doubling it would count it twice.
* For even $N$ the same is true of the $k_z = N/2$ **Nyquist** plane, because $-N/2$ and $+N/2$ are the same wavenumber on a discrete grid. That plane is self-paired too, so weight **1**.

The `herm` array takes care of this. 2 everywhere and overwritten with 1 on the first and last $k_z$ slices. With it the half-spectrum sums below exactly reproduce the full-spectrum ones.

```python
x = cp.linspace(0.0, L, N, endpoint=False).astype(REAL)
X, Y, Z = cp.meshgrid(x, x, x, indexing='ij')

M   = N // 2 + 1                                  # rfft last-axis length
_kf = cp.fft.fftfreq(N, d=1.0 / N).astype(REAL)   # full frequencies (axes 0,1)
_kr = cp.fft.rfftfreq(N, d=1.0 / N).astype(REAL)  # non-negative frequencies (reduced axis 2)
RKX, RKY, RKZ = cp.meshgrid(_kf, _kf, _kr, indexing='ij')   # (N, N, M)
RK2  = RKX**2 + RKY**2 + RKZ**2
RK2s = cp.where(RK2 == 0, 1.0, RK2)
kd   = N // 3
rdeal = ((cp.abs(RKX) <= kd) & (cp.abs(RKY) <= kd) & (cp.abs(RKZ) <= kd)).astype(REAL)

# Hermitian Parseval weights: 1 on k_z=0 and (even N) the Nyquist plane, else 2
herm = cp.full((N, N, M), 2.0, dtype=cp.float64)
herm[:, :, 0] = 1.0
if N % 2 == 0:
    herm[:, :, -1] = 1.0

_INV_NORM = 1.0 / (N ** 3)
print(f"half-spectrum {(N, N, M)};  dealias |k_i| <= {kd};  active {int(rdeal.sum())}/{N*N*M}")
```

    half-spectrum (512, 512, 257);  dealias |k_i| <= 170;  active 19884050/67371008
    

### R2C / C2R plans (reused every step)

One `R2C` plan (real -> Hermitian half) and one `C2R` plan (half -> real), reused via
`reset_operand`. `irfftn` folds in the `1/N**3` that cuFFT's C2R omits.

```python
_r2c = nvfft.FFT(cp.empty((N, N, N), REAL), axes=(0, 1, 2),
                 options={"fft_type": "R2C", "result_layout": "natural"}); _r2c.plan()
_c2r = nvfft.FFT(cp.empty((N, N, M), COMPLEX), axes=(0, 1, 2),
                 options={"fft_type": "C2R", "last_axis_parity": "even", "result_layout": "natural"}); _c2r.plan()

def rfftn(a):
    _r2c.reset_operand(cp.ascontiguousarray(a, dtype=REAL))
    return _r2c.execute(direction=FWD)

def irfftn(a):
    _c2r.reset_operand(cp.ascontiguousarray(a, dtype=COMPLEX))
    return _c2r.execute(direction=INV) * _INV_NORM

_tr = cp.random.random((N, N, N)).astype(REAL)
print("R2C round-trip error:", float(cp.max(cp.abs(irfftn(rfftn(_tr)) - _tr))))
```

    R2C round-trip error: 1.0728836059570312e-06
    

### Initial conditions

The Taylor-Green vortex has initial kinetic energy
$E_0 = \tfrac{1}{2}\langle u^2 + v^2 + w^2 \rangle = \tfrac{1}{8} = 0.125$.

It is analytically divergence-free as $\partial_x u + \partial_y v + \partial_z w =
 0$. The printed $E_0$ is therefore both a physics check and a check that the R2C normalization and Hermitian weights are computed
correctly.

```python
u0 =  cp.sin(X) * cp.cos(Y) * cp.cos(Z)
v0 = -cp.cos(X) * cp.sin(Y) * cp.cos(Z)
w0 =  cp.zeros_like(X)
uh0_r, vh0_r, wh0_r = rfftn(u0), rfftn(v0), rfftn(w0)
print(f"initial KE = {0.5*float(cp.mean(u0**2 + v0**2 + w0**2)):.6f}  (exact 0.125)")
```

    initial KE = 0.125000  (exact 0.125)
    

### Right-hand side — the projected Navier-Stokes equations

Defining the nonlinear advection term $\mathbf{N} = -(\mathbf{u}\cdot\nabla)\mathbf{u}$, the projected Navier-Stokes equations in Fourier space are:

$$\frac{\partial \hat{\mathbf{u}}}{\partial t} = \mathbf{P}(\mathbf{k})\,\hat{\mathbf{N}} - \nu k^2 \hat{\mathbf{u}}$$

where $\mathbf{P}(\mathbf{k}) = \mathbf{I} - \mathbf{k}\mathbf{k}^\top / k^2$ is the **Leray projector**, which removes the pressure gradient and enforces $\nabla \cdot \mathbf{u} = 0$.

$\mathbf{N}$ is computed in physical space to avoid $O(N^6)$ convolutions, with dealiasing applied before and after the transform.

**Why the transform path is worth optimizing.** Each RHS evaluation costs 12 inverse transforms (3 velocities + 9 velocity gradients) and 3 forward transforms. With two RHS calls per Heun step that is **30 three-dimensional FFTs per time step**, and at $N = 512$ each one moves a 537 MB array. Everything else in the step, meaning the products, the projection and the Heun combination, is elementwise arithmetic on arrays of that same size.

The reason this makes the solver *bandwidth*-bound rather than *compute*-bound is a matter of arithmetic intensity, the number of flops performed per byte moved:

* An FFT does $O(\log N)$ flops per element. At $N = 512$ that is a few dozen flops against the 8 bytes of a `complex64`, so a handful of flops per byte.
* The elementwise operations are far worse. An expression like `1j*RKX*uhd` reads two arrays, writes one, and does a single multiply, which is well under **1 flop per byte**.

An H200 delivers roughly 60 TFLOP/s of fp32 against 4.8 TB/s of memory bandwidth, a machine balance of about **13 flops per byte**. Any kernel below that ratio finishes its arithmetic while still waiting on memory, so its runtime is set purely by how many bytes it touches. Both categories above sit one to two orders of magnitude below it.

Two consequences follow, and together they are the whole optimization campaign:

1. **Move fewer bytes per element.** fp32 instead of fp64 halves every transfer, and R2C instead of C2C halves the number of elements. Neither changes the flop count meaningfully, and both roughly double the speed.
2. **Eliminate the elementwise passes entirely.** Each `dealias *` or `1j*RKX*` is a separate kernel that reads and rewrites half a gigabyte in order to do almost no arithmetic. Folding those products *into* the FFT kernel as a load or store callback makes them nearly free because the data was already in registers. That is the callback fusion described at the end of this notebook.

---

**Why the Leray projector?**

*Where the Poisson equation for pressure comes from.*

Start from the momentum equation (dropping the viscous term for clarity):

$$\frac{\partial \mathbf{u}}{\partial t} = \underbrace{-(\mathbf{u} \cdot \nabla)\mathbf{u}}_{\mathbf{N}} - \nabla p$$

Take the divergence of both sides:

$$\frac{\partial (\nabla \cdot \mathbf{u})}{\partial t} = \nabla \cdot \mathbf{N} - \nabla^2 p$$

We want $\nabla \cdot \mathbf{u} = 0$ to hold for all time, so the left side must be zero. This gives the **Poisson equation for pressure**:

$$\nabla^2 p = \nabla \cdot \mathbf{N} = -\nabla \cdot \left[(\mathbf{u} \cdot \nabla)\mathbf{u}\right]$$

Pressure has no independent dynamics. It is slaved to the velocity field, adjusting itself to keep $\nabla \cdot \mathbf{u} = 0$.

*Why does $\mathbf{N}$ have non-zero divergence?*

Using index notation with the incompressibility condition $\partial_i u_i = 0$:

$$\nabla \cdot \mathbf{N} = -\partial_i(u_j \, \partial_j u_i) = -(\partial_i u_j)(\partial_j u_i) - u_j \underbrace{\partial_j(\partial_i u_i)}_{=\,0} = -(\partial_i u_j)(\partial_j u_i)$$

This last term vanishes only for very special (e.g. constant) velocity fields. So every time-step produces a velocity increment that would violate incompressibility, and the pressure corrects it.

Once $p$ is found, the corrected (divergence-free) nonlinear term is $\mathbf{N}^\perp = \mathbf{N} - \nabla p$. In Fourier space the Poisson equation $-k^2\hat{p} = i\mathbf{k} \cdot \hat{\mathbf{N}}$ gives:

$$\hat{p} = \frac{-i\mathbf{k} \cdot \hat{\mathbf{N}}}{k^2}, \qquad \hat{\mathbf{N}}^{\perp} = \hat{\mathbf{N}} - i\mathbf{k}\,\hat{p} = \hat{\mathbf{N}} - \frac{\mathbf{k}(\mathbf{k} \cdot \hat{\mathbf{N}})}{k^2} = \underbrace{\left(\mathbf{I} - \frac{\mathbf{k}\mathbf{k}^\top}{k^2}\right)}_{\mathbf{P}(\mathbf{k})}\hat{\mathbf{N}}$$

This is the Leray projector: a pointwise operation in Fourier space that replaces what would otherwise be a global Poisson solver in physical space. In the code it is the three lines built from `kdotN / RK2s`.

```python
def compute_rhs_r2c(state):
    uh, vh, wh = state
    uhd, vhd, whd = uh*rdeal, vh*rdeal, wh*rdeal

    u = irfftn(uhd); v = irfftn(vhd); w = irfftn(whd)

    dudx = irfftn(1j*RKX*uhd); dudy = irfftn(1j*RKY*uhd); dudz = irfftn(1j*RKZ*uhd)
    dvdx = irfftn(1j*RKX*vhd); dvdy = irfftn(1j*RKY*vhd); dvdz = irfftn(1j*RKZ*vhd)
    dwdx = irfftn(1j*RKX*whd); dwdy = irfftn(1j*RKY*whd); dwdz = irfftn(1j*RKZ*whd)

    Nuh = rfftn(-(u*dudx + v*dudy + w*dudz)) * rdeal
    Nvh = rfftn(-(u*dvdx + v*dvdy + w*dvdz)) * rdeal
    Nwh = rfftn(-(u*dwdx + v*dwdy + w*dwdz)) * rdeal

    kdotN = RKX*Nuh + RKY*Nvh + RKZ*Nwh
    Nuh = Nuh - kdotN / RK2s * RKX
    Nvh = Nvh - kdotN / RK2s * RKY
    Nwh = Nwh - kdotN / RK2s * RKZ
    return (Nuh - nu*RK2*uh, Nvh - nu*RK2*vh, Nwh - nu*RK2*wh)
```

### Energy diagnostics

Two scalar quantities are tracked over time.

**Kinetic energy** $E(t)$ is the volume-averaged energy per unit mass:

$$E = \frac{1}{2}\langle \lvert \mathbf{u} \rvert^2 \rangle = \frac{1}{2N^6} \sum_\mathbf{k} \left( \lvert \hat u \rvert^2 + \lvert \hat v \rvert^2 + \lvert \hat w \rvert^2 \right)$$

It measures how much energy the flow contains. For the Taylor-Green vortex it starts at $E_0 = 1/8$ and decays monotonically once turbulence develops.

**Dissipation rate** $\varepsilon(t)$ is the rate at which kinetic energy is converted to heat by viscosity:

$$\varepsilon = \nu\langle \lvert \nabla\mathbf{u} \rvert^2 \rangle = \frac{\nu}{N^6} \sum_\mathbf{k} k^2 \left( \lvert \hat u \rvert^2 + \lvert \hat v \rvert^2 + \lvert \hat w \rvert^2 \right)$$

The $k^2$ factor comes from Parseval's theorem. Spectral differentiation multiplies each mode by $ik$, so $\lvert \nabla u \rvert^2$ in physical space becomes $k^2\lvert \hat{u} \rvert^2$ in Fourier space. $\varepsilon$ peaks near $t \approx 9$ for $Re = 1600$, marking the transition to turbulence.

Since $E$ is the only energy in the system and viscosity the only sink, $\dfrac{dE}{dt} = -\varepsilon$, so the energy curve is the integral of the dissipation curve. This is why $E$ falls fastest right where $\varepsilon$ peaks.

---

**Enstrophy and why its peak is important.**

The **enstrophy** is the mean-square vorticity,

$$\Omega(t) = \tfrac{1}{2}\langle \lvert \boldsymbol{\omega} \rvert^2 \rangle, \qquad \boldsymbol{\omega} = \nabla \times \mathbf{u}$$

For an incompressible field on a periodic domain the two are equal, $\langle\lvert \nabla \mathbf{u} \rvert^2\rangle = \langle\lvert \boldsymbol{\omega} \rvert^2\rangle$.

*In Fourier space*, which is the natural setting here since it is how the code actually computes $\varepsilon$, the argument is simple. $\hat{\boldsymbol{\omega}} = i\mathbf{k} \times \hat{\mathbf{u}}$ gives

$$\lvert \hat{\boldsymbol{\omega}} \rvert^2 = k^2\lvert \hat{\mathbf{u}} \rvert^2 - \lvert \mathbf{k}\cdot\hat{\mathbf{u}} \rvert^2$$

Incompressibility in Fourier space is $\mathbf{k}\cdot\hat{\mathbf{u}} = 0$, which is the condition the Leray projector enforces every step, so the second term vanishes mode by mode and $\lvert \hat{\boldsymbol{\omega}} \rvert^2 = k^2\lvert \hat{\mathbf{u}} \rvert^2$. Summing over $\mathbf{k}$ with Parseval gives the identity.

*In physical space* the same statement required an integration by parts. Using index notation with $\varepsilon_{ijk}\varepsilon_{ilm} = \delta_{jl}\delta_{km} - \delta_{jm}\delta_{kl}$, we get 

$$\lvert \boldsymbol{\omega} \rvert^2 = (\varepsilon_{ijk}\partial_j u_k)(\varepsilon_{ilm}\partial_l u_m) = \partial_j u_k\, \partial_j u_k - \partial_j u_k\, \partial_k u_j = \lvert \nabla\mathbf{u} \rvert^2 - \partial_j u_k\, \partial_k u_j$$

Rewriting,

$$\partial_j u_k\, \partial_k u_j = \partial_j\left(u_k\, \partial_k u_j\right) - u_k\, \partial_k \underbrace{\left(\partial_j u_j\right)}_{\nabla\cdot\mathbf{u}\ =\ 0} = \nabla \cdot \left[(\mathbf{u}\cdot\nabla)\mathbf{u}\right]$$

So incompressibility leaves a pure divergence. Averaging over the box means integrating, and the volume integral of a divergence is a flux through the boundary. On a triply-periodic domain every face is identified with the face opposite it, so those fluxes cancel out and the integral is zero.

Both conditions are important. Drop incompressibility and the $u_k \partial_k(\nabla\cdot\mathbf{u})$ term survives. Drop periodicity, as in any wall-bounded flow, and the surface flux survives instead. In that second case dissipation and enstrophy differ by a boundary term, which is why the clean identity used here is a property of homogeneous turbulence rather than of turbulence in general.

Enstrophy and dissipation are therefore the same measurement up to a constant:

$$\varepsilon = \nu \langle \lvert \nabla\mathbf{u} \rvert^2 \rangle = \nu \langle \lvert \boldsymbol{\omega} \rvert^2 \rangle = 2\nu\,\Omega$$

This is why the code below tracks $\varepsilon$ while the time loop calls the snapshot it keeps the "peak enstrophy" state. At fixed $\nu$ the two peak at the same instant.

Enstrophy comes from **vortex stretching**. The vorticity equation for incompressible flow is

$$\frac{D\boldsymbol{\omega}}{Dt} = \underbrace{(\boldsymbol{\omega} \cdot \nabla)\mathbf{u}}_{\text{stretching}} + \nu\nabla^2\boldsymbol{\omega}$$

The first term has no counterpart in the energy budget and no 2-D analogue. When a vortex tube is stretched along its own axis, conservation of angular momentum spins it up, so $\lvert \boldsymbol{\omega} \rvert$ grows while the tube thins. Thinner structures mean larger wavenumbers, which the $k^2$ in $\varepsilon$ weights heavily. This is the energy cascade seen from the vorticity side, and it is the mechanism that carries the Taylor-Green vortex from a smooth, entirely laminar initial condition to fully developed turbulence with nothing forcing it.

The **peak** is therefore the moment the cascade completes. Stretching has driven energy all the way down to the scales where viscosity can act, and from then on viscous destruction of vorticity outruns its production. There are three reasons why it is the benchmark quantity for this flow:

* **It is a single scalar with a known answer.** Brachet et al. (1983) established $\varepsilon_{max} \approx 1.26\times10^{-2}$ near $t = 9$ at $Re = 1600$, and it has been reproduced by enough independent codes to serve as a de facto standard.
* **It is the most resolution-sensitive quantity in the run.** The peak is exactly when the spectrum extends furthest toward high $k$. An under-resolved grid cannot represent those scales, cannot dissipate what the cascade delivers, and reports a peak that is too low, at the wrong time, or diverges outright. Matching the peak is a far sharper test of a solver than matching $E(t)$, which is smooth and forgiving.
* **It is the most interesting instant physically.** Maximum small-scale activity means the richest structure, which is why the time loop below snapshots the state at the peak and uses that snapshot for the spectrum and vorticity figures rather than the final, decayed field.

Both are computed via Parseval's theorem (normalisation factor $N^6$ for an unnormalised `fftn`), sharing the intermediate array `amp2` $= \texttt{herm}\cdot(\lvert \hat{u} \rvert^2 + \lvert \hat{v} \rvert^2 + \lvert \hat{w} \rvert^2)$. The `herm` multiplicity weights turn the half-spectrum sum back into the full-spectrum one. The sums are accumulated in `float64` even though the state is `float32`: $N^3 = 1.3\times10^8$ terms of wildly different magnitude would otherwise lose the small-scale contribution to round-off.

---

The **energy spectrum** $E(k)$ shows how energy is distributed across length scales. Each 3-D Fourier mode $\mathbf{k} = (k_x, k_y, k_z)$ is assigned to the integer shell $k = \mathrm{round}(\lvert \mathbf{k} \rvert)$ and its energy is accumulated there:

$$E(k) = \sum_{\mathrm{round}(\lvert \mathbf{k} \rvert) = k} \frac{\lvert \hat{u} \rvert^2 + \lvert \hat{v} \rvert^2 + \lvert \hat{w} \rvert^2}{2N^6}$$

```python
def diag_r2c(state):
    uh, vh, wh = state
    norm = float(N) ** 6
    amp2 = herm * (cp.abs(uh)**2 + cp.abs(vh)**2 + cp.abs(wh)**2)
    E   = 0.5 * float(cp.sum(amp2, dtype=cp.float64)) / norm
    eps = nu  * float(cp.sum(RK2 * amp2, dtype=cp.float64)) / norm
    return E, eps

def energy_spectrum_r2c(state):
    uh, vh, wh = state
    norm = float(N) ** 6
    energy_k = 0.5 * herm * (cp.abs(uh)**2 + cp.abs(vh)**2 + cp.abs(wh)**2) / norm
    kmax = N // 2
    shells = cp.asnumpy(cp.clip(cp.round(cp.sqrt(RK2)).astype(cp.int32).ravel(), 0, kmax))
    vals   = cp.asnumpy(energy_k.ravel())
    spectrum = np.zeros(kmax + 1); np.add.at(spectrum, shells, vals)
    return spectrum
```

### RK2 time integration (Heun's method)

$$\mathbf{k}_1 = f(t_n,\, \mathbf{u}_n)$$

$$\mathbf{k}_2 = f(t_n + \Delta t,\, \mathbf{u}_n + \Delta t\,\mathbf{k}_1)$$

$$\mathbf{u}_{n+1} = \mathbf{u}_n + \frac{\Delta t}{2}(\mathbf{k}_1 + \mathbf{k}_2)$$

This is an explicit predictor-corrector method. It is second-order accurate, needs two RHS evaluations per step, and stores only the current state, which matters at $512^3$ where each additional saved field costs 537 MB. The viscous terms is also treated explicitly. The time step is set by a CFL condition.

```python
def rk2_r2c(state):
    k1 = compute_rhs_r2c(state)
    sp = tuple(s + dt*f for s, f in zip(state, k1))
    k2 = compute_rhs_r2c(sp)
    return tuple(s + 0.5*dt*(f1+f2) for s, f1, f2 in zip(state, k1, k2))
```

### Sanity check (Hermitian weights + wiring)

```python
E0, eps0 = diag_r2c((uh0_r, vh0_r, wh0_r))
print(f"initial diagnostics: E {E0:.6f} (exact 0.125),  eps {eps0:.4e}")
assert abs(E0 - 0.125) < 1e-3, "energy off -> check Hermitian weights / normalization"
print("R2C wiring OK.  ✓")
gpu_mem("after setup")
```

    initial diagnostics: E 0.125000 (exact 0.125),  eps 4.6875e-04
    R2C wiring OK.  ✓
    [GPU memory] after setup
      device    :  10.93 / 139.80 GB used ( 7.8%) | 128.87 GB free
      CuPy pool :   7.52 GB live,  10.40 GB reserved
      ~not in this process's pool (other procs / workspaces / context):   0.54 GB
    

    (138369171456, 150109880320)

### Time loop (R2C) with peak-enstrophy capture

```python
state = (uh0_r, vh0_r, wh0_r)
times, energies, dissipations = [0.0], [], []
E0, eps0 = diag_r2c(state); energies.append(E0); dissipations.append(eps0)
peak = {"eps": -1.0, "t": 0.0, "state": None}

print("Warming up R2C plans ...", end=" ", flush=True)
_ = rk2_r2c(state); cp.cuda.Stream.null.synchronize(); print("done.")

t_wall = time.perf_counter()
for step in range(1, n_steps + 1):
    state = rk2_r2c(state)
    if step % log_freq == 0:
        t = step * dt
        E, eps = diag_r2c(state)
        times.append(t); energies.append(E); dissipations.append(eps)
        if eps > peak["eps"]:
            peak = {"eps": eps, "t": t, "state": tuple(s.copy() for s in state)}
        if step % (log_freq * 10) == 0:
            print(f"  t={t:5.2f}  E={E:.5f}  eps={eps:.4e}")
cp.cuda.Stream.null.synchronize()
elapsed = time.perf_counter() - t_wall
print(f"\nCompleted {n_steps} steps in {elapsed:.1f} s ({1000*elapsed/n_steps:.2f} ms/step)")
print(f"peak enstrophy eps={peak['eps']:.4e} at t={peak['t']:.2f}")
```

    Warming up R2C plans ... done.
      t= 0.62  E=0.12470  eps=4.8715e-04
      t= 1.25  E=0.12438  eps=5.5021e-04
      t= 1.88  E=0.12400  eps=6.7367e-04
      t= 2.50  E=0.12352  eps=8.8089e-04
      t= 3.12  E=0.12288  eps=1.2019e-03
      t= 3.75  E=0.12198  eps=1.7260e-03
      t= 4.38  E=0.12061  eps=2.7841e-03
      t= 5.00  E=0.11844  eps=4.1271e-03
      t= 5.62  E=0.11555  eps=5.0189e-03
      t= 6.25  E=0.11214  eps=6.0156e-03
      t= 6.88  E=0.10794  eps=7.2252e-03
      t= 7.50  E=0.10304  eps=8.7301e-03
      t= 8.12  E=0.09696  eps=1.0823e-02
      t= 8.75  E=0.08961  eps=1.2722e-02
      t= 9.38  E=0.08171  eps=1.1967e-02
      t=10.00  E=0.07448  eps=1.1267e-02
      t=10.62  E=0.06757  eps=1.0690e-02
      t=11.25  E=0.06120  eps=9.6541e-03
      t=11.88  E=0.05550  eps=8.5908e-03
    
    Completed 9600 steps in 1098.4 s (114.41 ms/step)
    peak enstrophy eps=1.2855e-02 at t=9.00
    

### Results — kinetic energy and dissipation rate

$E(t)$ decays monotonically from $t = 0$ as energy is continuously drained by viscosity. The dissipation rate $\varepsilon(t)$ rises as vorticity intensifies through the turbulent cascade, peaks near $t \approx 9$, then falls as the flow runs out of energy. The peak of $\varepsilon$ marks the onset of fully developed turbulence and is the primary quantity benchmarked against reference DNS data.

At $N = 512$ this run gives **$\varepsilon_{max} = 1.286\times10^{-2}$ at $t = 9.0$**, against
the reference value of $\approx 1.26\times10^{-2}$ — about 2% high, the expected direction and
magnitude of error for a fp32 run at this resolution.

> **Reference:** Brachet, M. E., Meiron, D. I., Orszag, S. A., Nickel, B. G., Morf, R. H., & Frisch, U. (1983). Small-scale structure of the Taylor-Green vortex. *Journal of Fluid Mechanics*, **130**, 411-452.

```python
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(times, energies, lw=1.5); ax[0].set_title('Kinetic Energy $E(t)$')
ax[0].set_xlabel('$t$'); ax[0].grid(True, alpha=0.3)
ax[1].plot(times, dissipations, lw=1.5, color='C1'); ax[1].set_title(r'Dissipation $\varepsilon(t)$')
ax[1].set_xlabel('$t$'); ax[1].grid(True, alpha=0.3)
plt.suptitle(f'TGV (R2C, fp32)  |  N={N}, Re={Re}', y=1.02); plt.tight_layout(); plt.show()
```

    
![png](/assets/img/tgv_h200/taylor_green_vortex_25_0.png)
    

### Results — energy spectrum

At the dissipation peak the inertial sub-range shows a $k^{-5/3}$ Kolmogorov scaling.
The dealiasing cutoff at $k = N/3 = 170$ is visible as the drop-off on the right. The
spectrum is taken at the peak-enstrophy snapshot rather than at the final time, since that
is when the cascade is most fully developed and the inertial range widest.

```python
_ps = peak["state"] if peak["state"] is not None else state
spec = energy_spectrum_r2c(_ps); kk = np.arange(len(spec))
fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(kk[1:], spec[1:], lw=1.5, label=f'$t={peak["t"]:.1f}$ (peak)')
kref = kk[2:24]; ax.loglog(kref, 5e-4*kref**(-5/3), 'k--', lw=1, label=r'$k^{-5/3}$')
ax.axvline(N // 3, color='gray', lw=0.8, ls=':', label=f'dealias k={N//3}')
ax.set_xlabel('$k$'); ax.set_ylabel('$E(k)$'); ax.legend(); ax.grid(True, alpha=0.3, which='both')
ax.set_title(f'Energy spectrum  |  N={N}'); plt.tight_layout(); plt.show()
```

    
![png](/assets/img/tgv_h200/taylor_green_vortex_27_0.png)
    

### Results — vorticity cross-section

The $z$-component of vorticity $\omega_z = \partial v/\partial x - \partial u/\partial y$
at the $z=0$ plane, at the moment of peak enstrophy.

Note that $z = 0$ is a **symmetry plane** of the Taylor-Green vortex: the initial condition
forces $w = \omega_x = \omega_y = 0$ there, and the symmetry is preserved by the dynamics.
The flow in that plane is therefore effectively two-dimensional, which makes it a clean but
incomplete view — it cannot show the three-dimensional vortex sheets that carry most of the
small-scale structure. The next figure does.

```python
uh_f, vh_f, wh_f = peak["state"] if peak["state"] is not None else state
t_snap = peak["t"] if peak["state"] is not None else T
omega_z = cp.asnumpy(irfftn(1j*RKX*vh_f*rdeal) - irfftn(1j*RKY*uh_f*rdeal))

fig, ax = plt.subplots(figsize=(9, 9), dpi=160)
vmax = np.percentile(np.abs(omega_z), 99.5)
im = ax.imshow(omega_z[:, :, 0], origin='lower', extent=[0, 2*np.pi, 0, 2*np.pi],
               cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='bilinear')
plt.colorbar(im, ax=ax, label=r'$\omega_z$', shrink=0.82, pad=0.02)
ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.set_aspect('equal')
ax.set_title(rf'TGV vorticity $\omega_z$ at $z=0$,  $t={t_snap:.1f}$ (peak),  $N={N}$ (R2C fp32)')
plt.tight_layout()
plt.savefig(f'tgv_vorticity_R2C_N{N}.png', dpi=160, bbox_inches='tight')
plt.show()
print(f"saved tgv_vorticity_R2C_N{N}.png")
```

    
![png](/assets/img/tgv_h200/taylor_green_vortex_29_0.png)
    

    saved tgv_vorticity_R2C_N512.png
    

### Results — the three-dimensional structure

Taking the magnitude $\lvert \boldsymbol{\omega} \rvert = \lvert \nabla \times \mathbf{u} \rvert$ over the whole volume and projecting its maximum along $z$ collects every vortex sheet in the domain into one image. The $z = 0$ slice cannot show this structure. At peak enstrophy the flow
has organised into thin, folded sheets of intense vorticity, which is the
small-scale geometry that makes $\varepsilon$ peak. The dissipation rate $\nu\langle\lvert \nabla\mathbf{u} \rvert^2\rangle$ is concentrated where the velocity gradients are steepest.

```python
uh_f, vh_f, wh_f = peak["state"]
def d(f, k): return irfftn(1j*k*f*rdeal)          # ∂/∂(dir) of a half-spectrum field
wx = d(wh_f, RKY) - d(vh_f, RKZ)
wy = d(uh_f, RKZ) - d(wh_f, RKX)
wz = d(vh_f, RKX) - d(uh_f, RKY)
wmag = cp.sqrt(wx**2 + wy**2 + wz**2)             # |ω|, full 3D

# (A) max-intensity projection along z
proj = cp.asnumpy(wmag.max(axis=2))
# (B) or an off-symmetry slice, e.g. z = π/2  (index N//4)
slab = cp.asnumpy(wmag[:, :, N//4])

img = proj    # try both
plt.figure(figsize=(9,9), dpi=160)
plt.imshow(img, origin='lower', extent=[0,2*np.pi,0,2*np.pi],
           cmap='inferno', vmax=np.percentile(img, 99.5), interpolation='bilinear')
plt.colorbar(label=r'$|\omega|$'); plt.title(rf'TGV $|\omega|$ (max-proj), $t={peak["t"]:.1f}$, $N={N}$')
plt.tight_layout(); plt.savefig(f'tgv_wmag_N{N}.png', dpi=160, bbox_inches='tight'); plt.show()

```

    
![png](/assets/img/tgv_h200/taylor_green_vortex_31_0.png)
    

### Continuing to $t = 20$ — the decayed field

The run above stopped at $t = 12$, just past the dissipation peak. Continuing to $t = 20$
costs another 6 400 steps and shows the other end of the story. With the cascade finished
and most of the energy dissipated ($E$ falls from $0.125$ to about $0.022$ so more than 80% of the
initial kinetic energy gone to heat), the remaining motion is large-scale and smooth again.

```python
# continue from the current state (t≈12) to t=20 to see the decayed field
t_now = times[-1]; T_end = 20.0
extra = int(round((T_end - t_now) / dt))
print(f"continuing from t={t_now:.2f} for {extra} steps ...")
_t0 = time.perf_counter()
for step in range(1, extra + 1):
    state = rk2_r2c(state)
    if step % log_freq == 0:
        t = t_now + step*dt
        E, eps = diag_r2c(state)
        times.append(t); energies.append(E); dissipations.append(eps)
        if step % (log_freq*10) == 0:
            print(f"  t={t:5.2f}  E={E:.5f}  eps={eps:.4e}")
cp.cuda.Stream.null.synchronize()
print(f"reached t={times[-1]:.2f} in {time.perf_counter()-_t0:.1f} s")

```

    continuing from t=12.00 for 6400 steps ...
      t=12.62  E=0.04967  eps=6.8975e-03
      t=13.25  E=0.04564  eps=6.1015e-03
      t=13.88  E=0.04201  eps=5.5424e-03
      t=14.50  E=0.03874  eps=4.8692e-03
      t=15.12  E=0.03591  eps=4.2049e-03
      t=15.75  E=0.03344  eps=3.7395e-03
      t=16.38  E=0.03119  eps=3.4787e-03
      t=17.00  E=0.02910  eps=3.2263e-03
      t=17.62  E=0.02717  eps=2.9340e-03
      t=18.25  E=0.02543  eps=2.6156e-03
      t=18.88  E=0.02390  eps=2.3055e-03
      t=19.50  E=0.02254  eps=2.0424e-03
    reached t=20.00 in 732.5 s
    

#### $\omega_z(z{=}0)$ at $t = 20$

Late in the decay the $z = 0$ plane is at its most legible. The fine turbulent structure has
been dissipated away and what remains is the large-scale spiral of the surviving
two-dimensional vortices. The figure to compare against is the $\omega_z(z{=}0)$ image at $t = 9$, two figures above: same component, same plane, same colormap. The $\lvert \boldsymbol{\omega} \rvert$ image immediately preceding this one is a *different* quantity, an unsigned magnitude projected through the whole volume rather than a signed component in a single plane, so it is not the right basis for a before-and-after. Between the two $\omega_z$ snapshots the dissipation falls from $1.29\times10^{-2}$ to $2.0\times10^{-3}$, a factor of roughly six.

```python
uh_f, vh_f, wh_f = state                      # t=20 field
omega_z = cp.asnumpy(irfftn(1j*RKX*vh_f*rdeal) - irfftn(1j*RKY*uh_f*rdeal))
plt.figure(figsize=(9,9), dpi=160)
vmax = np.percentile(np.abs(omega_z), 99.5)
plt.imshow(omega_z[:,:,0], origin='lower', extent=[0,2*np.pi,0,2*np.pi],
           cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='bilinear')
plt.colorbar(label=r'$\omega_z$'); plt.gca().set_aspect('equal')
plt.title(rf'TGV $\omega_z$ at $z=0$, $t={times[-1]:.0f}$ (decayed), $N={N}$')
plt.xlabel('$x$'); plt.ylabel('$y$'); plt.tight_layout()
plt.savefig(f'tgv_wz_t20_N{N}.png', dpi=160, bbox_inches='tight'); plt.show()

```

    
![png](/assets/img/tgv_h200/taylor_green_vortex_35_0.png)
    

#### The same slice at lower resolution

Every panel below is the identical diagnostic: $\omega_z$ on the $z = 0$ plane at $t = 20$, produced by the same plotting code as the $N = 512$ figure above. Only the grid changes.

<div class="tgv-grid">
  <figure>
    <img src="/assets/img/tgv_h200/resolution_N64.png" alt="Vorticity cross-section at N = 64" loading="lazy">
    <figcaption><strong>N = 64</strong><br><span>262 144 points</span></figcaption>
  </figure>
  <figure>
    <img src="/assets/img/tgv_h200/resolution_N128.png" alt="Vorticity cross-section at N = 128" loading="lazy">
    <figcaption><strong>N = 128</strong><br><span>2 097 152 points</span></figcaption>
  </figure>
  <figure>
    <img src="/assets/img/tgv_h200/resolution_N256.png" alt="Vorticity cross-section at N = 256" loading="lazy">
    <figcaption><strong>N = 256</strong><br><span>16 777 216 points</span></figcaption>
  </figure>
</div>

After dealiasing, the largest wavenumber each grid actually carries is $k_{max} = N/3$, so these four runs resolve up to $k = 21$, $42$, $85$ and $170$ respectively, with grid spacings $\Delta x = 2\pi/N$ of $0.098$, $0.049$, $0.025$ and $0.012$. This can be seen in the thinness of the filaments. The width of the sharpest gradient a run can represent is set by $\Delta x$.

**A caveat on this comparison** $t = 20$ is the most forgiving moment to test resolution. By then viscosity has removed most of the small-scale content, the spectrum has collapsed back toward low $k$, and even $N = 64$ produces a plausible-looking picture. The moment resolution really matters is the dissipation peak at $t \approx 9$, when the cascade has pushed energy as far toward high $k$ as it will go. A grid that cannot represent those scales cannot dissipate what the cascade delivers, and the failure shows up not as a blurry figure but as a wrong $\varepsilon_{max}$, at the wrong time, or as an outright blow-up. 

One rendering caveat too: the $N = 512$ figure clips its colour scale at the 99.5th percentile of $\lvert \omega_z \rvert$ while these three clip at the 99th, so the low-resolution panels are very slightly more saturated. The structural difference is larger than that offset.

---

### Why nvmath-python + CuPy?

The original version of this solver was written in JAX and run at $N = 64$ on a Colab T4.
Getting to $N = 512$ took three changes, each measured on the same GPU at the same $N$:

| Lever | Speedup | What it does |
|---|---|---|
| **cuFFT callback fusion** (LTO-IR) | 1.22× | folds the dealias mask, the $ik$ derivative factors, and the $1/N^3$ normalization into the FFT kernels, so 12 of the 15 transforms per RHS carry their pre/post-multiplies for free instead of as separate memory-bound passes |
| **fp32 instead of fp64** | 3.13× | halves the bytes moved per transform; on consumer/workstation Blackwell, FP64 throughput is also suppressed to ~1/64 of FP32 |
| **R2C real transforms** | 1.45× at $N{=}128$, more at higher $N$ | exploits Hermitian symmetry: half the spectrum stored, half the transform work |

Together with an H200 (~4.8 TB/s High Bandwidth Memory vs ~1.8 TB/s) this is what brings a $512^3$ step from
**344 ms** (C2C-fused, fp32, RTX PRO 6000 Blackwell) down to **114 ms**. Thus, a full $T = 20$
run in about half an hour instead of most of a day.

#### The jargon 

**HBM**, or *High Bandwidth Memory*. Ordinary GPUs use GDDR, meaning memory chips sitting on the board next to the die and talking over a relatively narrow bus. HBM instead stacks DRAM dies vertically and mounts them on the same package as the GPU, connected by a bus thousands of bits wide. Same DRAM technology, vastly more wires. The H200's HBM3e runs at ~4.8 TB/s and the RTX PRO 6000's GDDR7 at ~1.8 TB/s. Since this solver is bandwidth-bound, as shown in the right-hand-side section above, that ~2.7× ratio is most of the explanation for the 3× wall-clock difference. The H200 is not doing more arithmetic, it is waiting on memory less.

**Callback fusion.** A normal FFT kernel loads each input element, transforms, and stores each output element. cuFFT lets you supply two small device functions that it splices into those load and store instructions: a **prolog** that runs on every element as it is read, and an **epilog** that runs on every element as it is written. So instead of

```
tmp = 1j * RKX * uhd * dealias / N**3   # separate kernel: read 537 MB, write 537 MB
out = ifft(tmp)                          # FFT kernel:      read 537 MB, write 537 MB
```

the products happen inside the FFT's own load on data already sitting in registers, and the intermediate array never exists. That is one kernel launch instead of two and half the memory traffic. For a workload whose elementwise passes do under 1 flop per byte, deleting them is nearly free speed.

**LTO-IR**, or *Link-Time Optimization Intermediate Representation*. The difficulty with the above is that the callback is *your* code while the FFT is NVIDIA's precompiled library, and ordinarily the two cannot be merged. We would need a function-pointer call per element, which on a GPU is slow. LTO-IR solves this by shipping the callback not as finished machine code but as a kind of intermediate representation. At plan time cuFFT links that IR into its own kernel and optimizes the combined result. Here `numba-cuda` compiles the Python callback to LTO-IR and nvmath-python hands it to cuFFT. 

#### Why nvmath-python rather than a generic array framework

All three levers above are **transform-level decisions**, and a framework that exposes only `fftn(x)` gives you nowhere to express them. There is no argument to `fftn` that means "fuse this mask into the load", and no way to say "plan this as R2C with an even last-axis parity, then reuse that plan 16 000 times".

Creating an FFT plan is expensive: cuFFT inspects the shape, chooses a factorization and algorithm, possibly runs heuristics, and allocates a scratch workspace. Doing that once per call, which is what a stateless `fftn` API must either do or hide behind a cache you do not control, would cost more than the transform itself at this step count. nvmath-python instead exposes the plan as a **first-class object** that you hold onto. You call `nvfft.FFT(...)` and `.plan()` once at setup, then `reset_operand()` each step to rebind that plan to a new data pointer, then `.execute()`. The same plan and workspace use new data. That how the `rfftn` and `irfftn` helpers near the top of this notebook are structured.

CuPy handles the ordinary array arithmetic in between, so the code still reads like NumPy.

---

## Run it yourself

<p class="tgv-links">
  <a href="https://molab.marimo.io/notebooks/nb_ftnExDz99uEZH7NpY9WsQv/app" target="_blank" rel="noopener noreferrer" class="btn btn-sm z-depth-0" role="button">Interactive demo (molab)</a>
  <a href="https://github.com/khilavm/taylor_green_vortex" target="_blank" rel="noopener noreferrer" class="btn btn-sm z-depth-0" role="button">Source on GitHub</a>
</p>

<p class="tgv-demo__hint">The GitHub link points at the repo containing (1) the JAX implementation, which runs at $N = 64$ on a free Colab T4, and (2) the $N = 512$ run presented above, using an H200.</p>
