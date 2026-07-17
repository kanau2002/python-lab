import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

plt.rcParams['font.family'] = 'Hiragino Sans'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(BASE_DIR, 'input_parking/area_by_region_all.csv'))
y = df['駐車場総面積_km2'].values
names = df['地域名'].values

configs = [
    {
        'x_col': '市町村総面積_km2',
        'title': '市町村総面積',
        'x_label': '市町村総面積 (km²)',
        'x_tick': 100,
        'x_fmt': None,
        'outlier_sigma': 1.25,
    },
    {
        'x_col': '駐車場数',
        'title': '駐車場数',
        'x_label': '駐車場数',
        'x_tick': 10000,
        'x_fmt': lambda x, _: f'{x/10000:.0f}万' if x > 0 else '0',
        'outlier_sigma': 2.0,
    },
    {
        'x_col': '人口',
        'title': '人口',
        'x_label': '人口',
        'x_tick': 200000,
        'x_fmt': lambda x, _: f'{x/10000:.0f}万' if x > 0 else '0',
        'outlier_sigma': 2.0,
    },
    {
        'x_col': '乗用車数',
        'title': '乗用車数',
        'x_label': '乗用車数',
        'x_tick': 50000,
        'x_fmt': lambda x, _: f'{x/10000:.0f}万' if x > 0 else '0',
        'outlier_sigma': 2.0,
    },
]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax, cfg in zip(axes, configs):
    x = df[cfg['x_col']].values

    slope, intercept, r_val, _, _ = stats.linregress(x, y)
    y_pred = slope * x + intercept
    residuals = y - y_pred
    rmse = np.sqrt(np.mean(residuals ** 2))
    n = len(x)

    res_std = np.std(residuals)
    outlier_mask = np.abs(residuals) > cfg['outlier_sigma'] * res_std

    ax.scatter(x, y, s=20, zorder=3)

    x_line = np.linspace(0, x.max() * 1.05, 300)
    ax.plot(x_line, slope * x_line + intercept, color='gray', linewidth=1.5,
            linestyle='--', zorder=2)

    # Annotate outliers
    for i in np.where(outlier_mask)[0]:
        xi, yi = x[i], y[i]
        # Offset direction based on position in data range
        xoff = -5 if xi > x.max() * 0.55 else 5
        yoff = -12 if yi > y.max() * 0.55 else 5
        ha = 'right' if xi > x.max() * 0.55 else 'left'
        ax.annotate(names[i], (xi, yi), textcoords='offset points',
                    xytext=(xoff, yoff), fontsize=9, ha=ha, color='dimgray')

    stats_text = f'$n$ = {n}\n$r$ = {r_val:.3f}\nRMSE = {rmse:.3f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='lightgray'))

    ax.set_title(cfg['title'], fontsize=12, fontweight='bold')
    ax.set_xlabel(cfg['x_label'], fontsize=10)
    if ax is axes[0]:
        ax.set_ylabel('駐車場総面積 (km²)', fontsize=10)
    else:
        ax.set_ylabel('')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    if cfg['x_tick']:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(cfg['x_tick']))
    if cfg['x_fmt']:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(cfg['x_fmt']))

    ax.tick_params(axis='x', labelrotation=0)
    ax.set_box_aspect(1)

plt.tight_layout()
plt.subplots_adjust(wspace=0.08)
output_path = os.path.join(BASE_DIR, 'output/scatter_plots.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f'Saved: {output_path}')
