r"""
Cross-validation of the span of the signal-systematic ratio smoothing (build_datacard.py smooth_shapes --ratio-span).

The per-year signal histograms written by build_datacard.py (<process>_sel-<selection>_year-<year>_mt.json) are
independent MC samples with the same binning. For each signal point, systematic and direction, each sample is held out
in turn: the central and the variation of the other samples are summed, their ratio is smoothed with
build_datacard.smooth_ratio_to_central (the --ratio-span code itself) at span S, and the prediction is compared with
the raw ratio of the held-out sample in the fit range:

    score(S) = sum_folds sum_bins (r_heldout - pred(S))^2 / e_heldout^2,   e = relative MC error of the central

The noise of the held-out sample enters every candidate equally, so the candidate with the lowest score summed over
the signal points is the one closest to the true ratio. Two references are scored the same way: "none" (the raw
training ratio, i.e. each variation used unsmoothed) and "flat" (the training yield ratio in the fit range, i.e. a
normalisation-only systematic). e is a per-bin weight, not the error of the ratio (variation and central share their
events), so only the ranking of the candidates is meaningful, not the absolute value.

The samples must be statistically independent. build_histograms processes each 2018 MC skim for both 2018PRE and
2018POST (same events, different luminosity weight and HEM veto), so the two are not independent: by default 2018POST
is not used and 2018PRE stands for the 2018 MC. Use --years only with samples made from disjoint events.
Year-specific systematics (e.g. jec2016) are scored under their generic name (jec), each sample with its own variation.

Usage:
    python ratio_span_cv.py hists/SVJ_*_year-*_mt.json [--spans 0.1 0.2 0.3 0.4 0.5 0.7 1.0] [--mtmin 180]
        [--mtmax 650] [--years 2016 2017 2018PRE] [--tolerance 0] [--span-max 0.5] [--out ratio_span_cv.json]
        [--plot plotdir] [--plot-heldout dir --plot-match 'mMed-350.*\|jes_up']
Prints, per selection and systematic, the summed score of each candidate relative to the best one, and the chosen
span: the smallest span whose summed score reaches the best span's (within --tolerance, default 0: the best span
itself), capped at --span-max: wider spans than the cap are not used even when they score best.
"""
import os, os.path as osp, sys, json, re, argparse
from collections import defaultdict

import numpy as np

import common
from build_datacard import smooth_ratio_to_central, get_ratio_span

FNAME = re.compile(r'^(?P<process>.*)_sel-(?P<sel>.*)_year-(?P<year>\w+)_mt\.json$')
YEARLY = re.compile(r'^(?P<name>[a-z]+)(?P<year>\d{4})_(?P<dir>up|down)$')
NOT_INDEPENDENT = {'2018POST'}   # same MC events as 2018PRE


def variation_name(key, year):
    """generic name of a variation in a sample of this year, or None if not used here"""
    if not (key.endswith('_up') or key.endswith('_down')) or key.startswith('mcstat'):
        return None
    m = YEARLY.match(key)
    if m is None:
        return key
    return f"{m['name']}_{m['dir']}" if year.startswith(m['year']) else None


def load(files, years):
    """{(process, sel): {year: {name: Histogram}}} for the signal samples"""
    out = defaultdict(dict)
    for f in files:
        m = FNAME.match(osp.basename(f))
        if m is None or m['year'] not in years:
            continue
        with open(f) as fp:
            d = json.load(fp)
        if d['central']['metadata'].get('sample_type') != 'sig':
            continue
        hists = {'central': common.Histogram.from_dict(d['central'])}
        for k, v in d.items():
            name = variation_name(k, m['year'])
            if name is not None:
                hists[name] = common.Histogram.from_dict(v)
        out[(m['process'], m['sel'])][m['year']] = hists
    return out


def raw_ratio(var, central):
    """ratio and its weight error as smoothed by smooth_ratio_to_central / do_loess (error 1 where the ratio is 0)"""
    ok = central.vals > 0
    r = np.where(ok, var.vals / np.where(ok, central.vals, 1.), 1.)
    e = np.where(ok, central.errs / np.where(ok, central.vals, 1.), 1.)
    return r, np.where(r == 0, 1., e)


def unit(hist):
    """histogram of ones: smooth_ratio_to_central(..., unit, span) returns the smoothed ratio itself"""
    out = hist.copy()
    out.vals = np.ones_like(hist.vals, dtype=float)
    out.errs = np.zeros_like(hist.vals, dtype=float)
    return out


def score_template(samples, name, spans, mtmin, mtmax):
    """summed held-out scores of one signal point and variation; samples = {year: {name: Histogram}}"""
    years = sorted(samples)
    x = samples[years[0]]['central'].binning
    x = (x[:-1] + x[1:]) / 2
    window = (x > mtmin) & (x < mtmax)
    scores = {**{str(s): 0. for s in spans}, 'none': 0., 'flat': 0.}
    nbins = 0
    for held in years:
        train = [y for y in years if y != held]
        c = sum((samples[y]['central'] for y in train[1:]), samples[train[0]]['central'])
        v = sum((samples[y][name] for y in train[1:]), samples[train[0]][name])
        r_o, e_o = raw_ratio(samples[held][name], samples[held]['central'])
        m = window & (c.vals > 0) & (samples[held]['central'].vals > 0)
        nbins += int(m.sum())
        score = lambda pred: float(np.sum((r_o[m] - pred[m])**2 / e_o[m]**2))
        scores['none'] += score(raw_ratio(v, c)[0])
        scores['flat'] += score(np.full(len(x), v.vals[window].sum() / c.vals[window].sum()))
        for s in spans:
            try:
                pred = smooth_ratio_to_central(v, c, unit(c), s).vals
            except Exception as e:   # e.g. too few bins in the window for the local fit
                common.logger.warning(f'{name} span {s}: {e}')
                pred = np.full(len(x), np.nan)
            scores[str(s)] += score(pred)
    return scores, nbins


def plot_heldout(key, name, samples, spans, mtmin, mtmax, outdir, highlight=None):
    """one plot per fold: the training samples' raw ratio, the held-out sample's raw ratio, the smoothed ratio at each
    span (smooth_ratio_to_central) and the flat ratio, with each entry's held-out score relative to the best entry"""
    plt, hep = cms_style()
    os.makedirs(outdir, exist_ok=True)
    years = sorted(samples)
    x = samples[years[0]]['central'].binning
    x = (x[:-1] + x[1:]) / 2
    window = (x > mtmin) & (x < mtmax)
    cmap = plt.get_cmap('Blues')
    for held in years:
        train = [y for y in years if y != held]
        c = sum((samples[y]['central'] for y in train[1:]), samples[train[0]]['central'])
        v = sum((samples[y][name] for y in train[1:]), samples[train[0]][name])
        rt, et = raw_ratio(v, c)
        r_o, e_o = raw_ratio(samples[held][name], samples[held]['central'])
        m = window & (c.vals > 0) & (samples[held]['central'].vals > 0)
        score = lambda pred: float(np.sum((r_o[m] - pred[m])**2 / e_o[m]**2))
        flat = v.vals[window].sum() / c.vals[window].sum()
        preds = [(sp, smooth_ratio_to_central(v, c, unit(c), sp).vals) for sp in spans]
        sc = {**{str(sp): score(p) for sp, p in preds}, 'flat': score(np.full(len(x), flat)), 'none': score(rt)}
        best = min(sc.values())
        fig, ax = plt.subplots(figsize=(8, 7.5))
        ax.errorbar(x[m] + 1.5, r_o[m], e_o[m], fmt='s', color='#c0392b', ms=4, lw=1, alpha=0.8,
                    label=f'held out ({held}) raw ratio')
        ax.errorbar(x[m] - 1.5, rt[m], et[m], fmt='o', color='black', ms=3, lw=1, alpha=0.6,
                    label=f'training ({" + ".join(train)}) raw ratio, score {sc["none"] / best:.3f}')
        ax.axhline(flat, color='grey', ls='--', lw=1.5, label=f'flat, score {sc["flat"] / best:.3f}')
        for i, (sp, p) in enumerate(preds):
            lw = 3 if sp == highlight else 1.6
            ax.plot(x[window], p[window], color=cmap(0.35 + 0.65 * i / max(1, len(preds) - 1)), lw=lw,
                    label=f'span {sp:g}, score {sc[str(sp)] / best:.3f}')
        lo = min(np.percentile(r_o[m], 5), min(p[window].min() for _, p in preds))
        hi = max(np.percentile(r_o[m], 95), max(p[window].max() for _, p in preds))
        pad = max(0.01, 0.3 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(mtmin, mtmax)
        ax.set_xlabel(r'$m_\mathrm{T}$ [GeV]')
        ax.set_ylabel(f'{name} / central')
        ax.text(0.03, 0.96, f'{key[0][:60]}\nsel-{key[1]}, {name}, held out {held}', transform=ax.transAxes,
                ha='left', va='top', fontsize=9, bbox=dict(facecolor='white', edgecolor='none', alpha=1))
        cms_label(hep, ax)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=9)
        fig.savefig(osp.join(outdir, f'heldout_{key[0]}_sel-{key[1]}_{name}_ho{held}.png'), bbox_inches='tight')
        plt.close(fig)


def _worker(key, name, samples, spans, mtmin, mtmax):
    return key, name, score_template(samples, name, spans, mtmin, mtmax)


def summarise(templates, candidates):
    """per selection|systematic: summed score of each candidate relative to the best"""
    summed = defaultdict(lambda: defaultdict(float))
    for (process, sel, name), (scores, nbins) in templates.items():
        syst = name.rsplit('_', 1)[0]
        for k in (f'{sel}|{syst}', f'{sel}|all'):
            for c in candidates:
                summed[k][c] += scores[c]
    out = {}
    for k, sc in summed.items():
        best = min(v for v in sc.values() if np.isfinite(v))
        out[k] = {c: sc[c] / best for c in candidates}
    return out


def cms_style():
    """CMS style (mplhep, as apply_DDT.py) without LaTeX; returns pyplot and mplhep"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import mplhep as hep
    plt.style.use(hep.style.CMS)
    plt.rcParams.update({'text.usetex': False, 'font.size': 15, 'axes.labelsize': 15, 'legend.fontsize': 10,
                         'xtick.labelsize': 12, 'ytick.labelsize': 12})
    return plt, hep


def cms_label(hep, ax):
    hep.cms.label('Preliminary', data=False, com=13, ax=ax, fontsize=12)


def choose_span(rel, spans, tolerance, span_max):
    """smallest span whose score is within (1 + tolerance) of the best span's, capped at span_max"""
    best = min(rel[str(s)] for s in spans)
    reached = [s for s in sorted(spans) if rel[str(s)] <= best * (1 + tolerance)]
    return min(reached[0], span_max)


def plot(summary, spans, outdir):
    plt, hep = cms_style()
    os.makedirs(outdir, exist_ok=True)
    for sel in sorted({k.split('|')[0] for k in summary}):
        fig, ax = plt.subplots(figsize=(8, 6))
        cols = iter(plt.get_cmap('tab10').colors)   # one distinct colour per systematic (the CMS cycle has fewer)
        for k, rel in sorted(summary.items()):
            s, syst = k.split('|')
            if s != sel:
                continue
            ax.plot(spans, [rel[str(x)] for x in spans], marker='o', lw=2.5 if syst == 'all' else 1.2,
                    color='black' if syst == 'all' else next(cols),
                    label=f"{syst} (none {rel['none']:.2f}, flat {rel['flat']:.2f})")
        ax.set_xlabel('ratio span')
        ax.set_ylabel('held-out score / best')
        ax.text(0.97, 0.95, f'sel-{sel}', transform=ax.transAxes, ha='right', va='top', fontsize=12)
        cms_label(hep, ax)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=9)
        fig.savefig(osp.join(outdir, f'ratio_span_cv_{sel}.png'), bbox_inches='tight')
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('jsonfiles', nargs='+', help='per-year histogram files (*_year-<year>_mt.json)')
    ap.add_argument('--spans', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
    ap.add_argument('--mtmin', type=float, default=180., help='lower edge of the scored range (fit range)')
    ap.add_argument('--mtmax', type=float, default=650., help='upper edge of the scored range (fit range)')
    ap.add_argument('--years', nargs='+', default=None,
                    help='independent samples to use (default: all found except 2018POST, see above)')
    ap.add_argument('--tolerance', type=float, default=0.,
                    help='relative score difference to the best span that still counts as optimal')
    ap.add_argument('--span-max', type=float, default=0.5, help='upper limit of the chosen span')
    ap.add_argument('--out', default='ratio_span_cv.json')
    ap.add_argument('--plot', default=None, help='directory for a summary plot per selection')
    ap.add_argument('--plot-heldout', default=None,
                    help='directory for per-fold plots of single templates (selected with --plot-match)')
    ap.add_argument('--plot-match', default=None,
                    help='regular expression on "<process>|<selection>|<variation>" for --plot-heldout')
    args = ap.parse_args()

    found = {m['year'] for m in (FNAME.match(osp.basename(f)) for f in args.jsonfiles) if m}
    years = set(args.years) if args.years else found - NOT_INDEPENDENT
    if not args.years and found & NOT_INDEPENDENT:
        common.logger.info(f'not using {sorted(found & NOT_INDEPENDENT)}: same MC events as 2018PRE')
    if len(years) < 2:
        raise ValueError(f'need at least two independent samples, got {sorted(years)}')
    common.logger.info(f'samples: {sorted(years)}')

    tasks = []
    for key, samples in sorted(load(args.jsonfiles, years).items()):
        if set(samples) != years:
            common.logger.warning(f'skip {key}: samples {sorted(samples)}')
            continue
        names = set.intersection(*[set(h) for h in samples.values()]) - {'central'}
        tasks += [(key, n, samples, args.spans, args.mtmin, args.mtmax) for n in sorted(names)]
    common.logger.info(f'{len(tasks)} templates')

    with common.mp_pool() as p:
        results = p.starmap(_worker, tasks)
    templates = {(key[0], key[1], name): res for key, name, res in results}

    candidates = [str(s) for s in args.spans] + ['none', 'flat']
    summary = summarise(templates, candidates)
    chosen = {k: choose_span(rel, args.spans, args.tolerance, args.span_max) for k, rel in summary.items()}
    with open(args.out, 'w') as f:
        json.dump({'meta': {'samples': sorted(years), 'spans': args.spans, 'range': [args.mtmin, args.mtmax],
                            'tolerance': args.tolerance, 'span_max': args.span_max},
                   'summary': summary, 'chosen_span': chosen,
                   'templates': {'|'.join(k): {'scores': s, 'nbins': n} for k, (s, n) in templates.items()}},
                  f, indent=1)

    print(f"{'selection|systematic':40s}" + ''.join(f'{c:>8s}' for c in candidates) + f"{'chosen':>8s}")
    for k in sorted(summary):
        print(f'{k:40s}' + ''.join(f'{summary[k][c]:8.3f}' for c in candidates) + f'{chosen[k]:8g}')
    if args.plot:
        plot(summary, args.spans, args.plot)
    if args.plot_heldout:
        pat = re.compile(args.plot_match or '$^')
        for key, name, samples, *_ in tasks:
            if pat.search(f'{key[0]}|{key[1]}|{name}'):
                plot_heldout(key, name, samples, args.spans, args.mtmin, args.mtmax, args.plot_heldout,
                             highlight=get_ratio_span(chosen, key[1], name, args.span_max))


if __name__ == '__main__':
    main()
