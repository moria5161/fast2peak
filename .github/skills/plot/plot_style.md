# Fast2Peak Plotting Style Guide

This skill summarizes the standard, publication-ready plotting style across all `fast2peak_v2` notebooks. **Important:** While these are the baseline guidelines, always remain flexible. If the user has special requirements or specific tweaks, adjust and adapt based on this skill, rather than rigidly enforcing this style.

## 1. Global Setup (Matplotlib `rcParams`)
Always insert the following basic Matplotlib configurations to ensure font and line style consistency. **Note: Unify to normal font weight (do not enforce bold fonts globally unless specifically requested by the user)**:

```python
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
from pathlib import Path
import os

# Define font paths
FONT_PATH = Path("/home/slurmusers/moria/fonts/Arial.ttf") # Adjust if necessary

# Define standard font sizes
FONT_SIZE_GLOBAL = 16
FONT_SIZE_TITLE = 18
FONT_SIZE_PANEL = 20
FONT_SIZE_LEGEND = 12

if os.path.exists(FONT_PATH):
    font_manager.fontManager.addfont(FONT_PATH)
    prop = font_manager.FontProperties(fname=FONT_PATH)
    plt.rcParams['font.family'] = prop.get_name()
else:
    plt.rcParams['font.family'] = 'Arial' # Fallback

plt.rcParams.update({
    'font.size': FONT_SIZE_GLOBAL,
    'font.weight': 'normal', # Unified to normal text, remove bold by default
    'axes.linewidth': 2.0,
    'xtick.major.width': 2.0,
    'ytick.major.width': 2.0,
    'xtick.major.size': 6.0,
    'ytick.major.size': 6.0,
    'axes.unicode_minus': False,
    'savefig.format': 'png',
    'figure.dpi': 100
})
```

## 2. Standard Colors and Linewidths (Colormap)
Use these configurations consistently when comparing different experimental methods. Do NOT use bold lineweights unless specified:
- **RAW (Noisy Spectrum):** `color='#969899'` (Gray), `alpha=1.0`, `linewidth=1.5`
- **GT (Ground Truth / Clean):** `color='#333333'` (Dark Charcoal / Black) or similar high-contrast dark color, `linewidth=1.5`
- **F2P (Proposed Method):** `color='#E64B35'` (Red), `alpha=1.0`, `linewidth=1.5` to `2.0`
- **SMAE (Baseline):** `color='#7fbcc1'` (Cyan/Teal), `linewidth=1.5` to `2.0`
- **WTD/PEER (Other Baselines):** Assign distinctive, color-blind friendly hex colors (e.g., `#4DBBD5`, `#00A087`) 
- **Markers:** 
  - `marker='o'` (F2P/Other elements) 
  - `marker='^'` (SMAE/Alternative Baselines)
  - `markersize=6` to `10` depending on density.

## 3. Subplot Layouts (`gridspec`)
Do not use basic `plt.subplots` for complex arrangements. Prefer `matplotlib.gridspec.GridSpec` to manually control sizes and spacing. 

Example for a 1x5 layout or 2x5 grid:
```python
fig = plt.figure(figsize=(14, 5))
gs = gridspec.GridSpec(1, 5, figure=fig, wspace=0.05, hspace=0.35)
ax = fig.add_subplot(gs[0, col_idx])
```

## 4. Signal Offsetting (Stacked Spectra Layout)
When visualizing multiple Raman spectra, apply calculated vertical offsets instead of plotting them atop each other.
- **Compute the global offset:** `master_y_offset = (noisy_spec_ref.max() - noisy_spec_ref.min()) * 0.8` (or factor ~ 0.75-0.8)
- **Shift subsequent inputs:** `ins_f2p_shifted = ins_f2p_raw - master_y_offset`
- Allow upper/lower space adjustments via margins:
  `limit_bottom = y_data_bottom - (y_span * 0.15)`
  `limit_top = y_data_top + (y_span * 0.05)`

## 5. Axis Decoration & Formatting
- **Axis hiding:** Mask internal/vertical y-ticks entirely for stacked offset plots: `ax.set_yticks([])`
- **Inward ticks:** `ax.tick_params(axis='x', direction='in', labelsize=FONT_SIZE_GLOBAL)`
- **Labeling conventions:** 
  - Standard X-label: `r'Raman Shift (cm$^{-1}$)'`
  - Titles for conditions (Normal weight): `ax.set_title(r'SNR: 5.0 dB', fontsize=FONT_SIZE_TITLE)`
  - Subfigure Panel Annotations ((a), (b), etc.): 
    ```python
    ax.text(-0.15, 1.05, '(a)', transform=ax.transAxes, fontsize=FONT_SIZE_PANEL, va='top', ha='right')
    ```
- **Legends:** 
  - Standardized legend style: `ax.legend(fontsize=FONT_SIZE_LEGEND, ncol=1, frameon=True, facecolor='white', edgecolor='gray')`

## 6. Title Label Formatting (Specific Formularies)
For PSNR titles and values, use standard RF-string notations:
```python
ax.set_title(rf"PSNR$_{{\mathrm{{RAW}}}}$={snr}", fontsize=FONT_SIZE_TITLE, pad=10)
```