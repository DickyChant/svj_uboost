"""
Signal systematic templates after smoothing, drawn as templates: for one merged signal histogram file, the raw
up and down variations and the smoothed ones of one or more smoothed files (e.g. smooth_shapes without and with
--ratio-span / --ratio-span-file), on top of the raw and smoothed central. Bottom panel: each divided by the smoothed
central of the first smoothed file, plus the raw variation / raw central (what --ratio-span smooths).

Usage:
    python plot_ratio_smoothing.py merged_<date>/SVJ_..._mt.json \\
        --smooth "per variation=smooth_A/SVJ_..._mt_smooth.json" "ratio=smooth_B/SVJ_..._mt_smooth.json" \\
        [--systematics jes scale ...] [--mtmin 180] [--mtmax 650] [--outdir plots_ratio_smoothing]
One PNG per systematic (up in red, down in blue): <outdir>/<input name>_<systematic>.png
"""
import os, os.path as osp, json, argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep

plt.style.use(hep.style.CMS)   # as apply_DDT.py, without LaTeX
plt.rcParams.update({'text.usetex': False, 'font.size': 15, 'axes.labelsize': 15, 'legend.fontsize': 10,
                     'xtick.labelsize': 12, 'ytick.labelsize': 12})

STYLES = ['--', '-', ':', '-.']


def load(path):
    with open(path) as f:
        d = json.load(f)
    return {k: (np.array(v['binning'], float), np.array(v['vals'], float), np.array(v['errs'], float),
                v.get('metadata', {}).get('ratio_span'))
            for k, v in d.items() if isinstance(v, dict) and 'vals' in v}


def cut(h, mtmin, mtmax):
    b, v, e = h[:3]
    x = (b[1:] + b[:-1]) / 2
    w = (x > mtmin) & (x < mtmax)
    return x[w], v[w], e[w]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('raw', help='merged (unsmoothed) signal histogram file')
    ap.add_argument('--smooth', nargs='+', required=True, help='"label=file" of smoothed files (smooth_shapes output)')
    ap.add_argument('--systematics', nargs='+', default=None,
                    help='e.g. jes jec2018 scale (default: all with _up and _down in the first smoothed file)')
    ap.add_argument('--mtmin', type=float, default=180.)
    ap.add_argument('--mtmax', type=float, default=650.)
    ap.add_argument('--outdir', default='plots_ratio_smoothing')
    args = ap.parse_args()

    raw = load(args.raw)
    smooth = [(s.split('=', 1)[0], load(s.split('=', 1)[1])) for s in args.smooth]
    systs = args.systematics or sorted({k[:-3] for k in smooth[0][1] if k.endswith('_up')
                                        and k[:-3] + '_down' in smooth[0][1] and k in raw})
    os.makedirs(args.outdir, exist_ok=True)
    x, c_raw, c_err = cut(raw['central'], args.mtmin, args.mtmax)
    c_sm = cut(smooth[0][1]['central'], args.mtmin, args.mtmax)[1]
    ok = c_sm > 0
    okr = c_raw > 0
    tag = osp.basename(args.raw).replace('.json', '')
    for syst in systs:
        fig, (ax, rx) = plt.subplots(2, 1, figsize=(8, 9), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.06))
        ax.errorbar(x, c_raw, c_err, fmt='o', color='black', ms=3, lw=1, label='central, raw')
        ax.plot(x, c_sm, color='black', lw=2, label=f'central, smoothed ({smooth[0][0]})')
        rx.axhline(1, color='black', lw=1)
        spread, top = 0.03, c_raw.max()
        for d, col, dx in (('up', '#c0392b', 1.5), ('down', '#1f4e9c', -1.5)):
            name = f'{syst}_{d}'
            _, v_raw, v_err = cut(raw[name], args.mtmin, args.mtmax)
            top = max(top, v_raw.max())
            ax.errorbar(x + dx, v_raw, v_err, fmt='s', color=col, ms=3, lw=1, alpha=0.5, label=f'{name}, raw')
            rx.errorbar(x + dx, np.where(ok, v_raw / np.where(ok, c_sm, 1), np.nan),
                        np.where(ok, v_err / np.where(ok, c_sm, 1), np.nan), fmt='s', color=col, ms=3, lw=1, alpha=0.4)
            rx.plot(x + dx, np.where(okr, v_raw / np.where(okr, c_raw, 1), np.nan), 'o', mfc='white', mec=col, ms=4,
                    label=f'raw {name} / raw central')
            for (lab, h), ls in zip(smooth, STYLES):
                v_sm = cut(h[name], args.mtmin, args.mtmax)[1]
                span = h[name][3]
                lw = 2.2 if ls == '-' else 1.6
                ax.plot(x, v_sm, color=col, lw=lw, ls=ls,
                        label=f'{name}, smoothed ({lab}' + (f', ratio span {span:g})' if span else ')'))
                r = np.where(ok, v_sm / np.where(ok, c_sm, 1), np.nan)
                rx.plot(x, r, color=col, lw=lw, ls=ls)
                spread = max(spread, 1.4 * np.nanmax(np.abs(r - 1)))
        rx.set_ylim(1 - spread, 1 + spread)
        ax.set_ylim(0, 1.25 * top)
        ax.set_ylabel('events / bin')
        ax.text(0.97, 0.95, f'{tag[:60]}\n{tag[60:]}\n{syst} up / down', transform=ax.transAxes, ha='right', va='top',
                fontsize=9)
        hep.cms.label('Preliminary', data=False, com=13, ax=ax, fontsize=12)
        rx.set_ylabel('/ smoothed central')
        rx.set_xlabel(r'$m_\mathrm{T}$ [GeV]')
        rx.set_xlim(args.mtmin, args.mtmax)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = rx.get_legend_handles_labels()
        rx.legend(h1 + h2, l1 + l2, loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=2, frameon=False, fontsize=8)
        fig.savefig(osp.join(args.outdir, f'{tag}_{syst}.png'), bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    main()
