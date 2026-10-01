import os, os.path as osp, sys, json, re, math
from time import strftime
from collections import defaultdict

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

import svj_ntuple_processing as svj
import common
from hadd_skims import expand_wildcards

THIS_DIR = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(THIS_DIR, 'systematics'))

scripter = common.Scripter()

def change_bin_width(hist_var):
    """
    Changes MT binning based on command line options
    """
    VarHistogram = common.registered_varhists[hist_var]
    binw = common.pull_arg(f'--{hist_var}binw', type=float).__dict__[f"{hist_var}binw"]
    binmin = common.pull_arg(f'--{hist_var}min', type=float, default=VarHistogram.default_binmin()).__dict__[f"{hist_var}min"]
    binmax = common.pull_arg(f'--{hist_var}max', type=float, default=VarHistogram.default_binmax()).__dict__[f"{hist_var}max"]
    if binw is not None:
        # Testing different bin widths
        VarHistogram.bins = VarHistogram.create_binning(binw, binmin, binmax)
        VarHistogram.non_standard_binning = True
        common.logger.warning(f'Changing bin width to {binw} ({binmin}, {binmax}); new binning: {VarHistogram.bins}')

def check_rebin(hist,name, hist_var):
    VarHistogram = common.registered_varhists[hist_var]
    if not VarHistogram.non_standard_binning:
        return hist
    msg = []
    if hist.binning[0]>VarHistogram.bins[0]:
        msg.append(f'left {hist.binning[0]}>{VarHistogram.bins[0]}')
    if hist.binning[-1]<VarHistogram.bins[-1]:
        msg.append(f'right {hist.binning[-1]}>{VarHistogram.bins[-1]}')
    orig_width = hist.binning[1] - hist.binning[0]
    new_width = VarHistogram.bins[1] - VarHistogram.bins[0]
    rebin_factor = int(new_width/orig_width)
    rebin_mod = new_width % orig_width
    if rebin_mod!=0:
        msg.append(f'rebin {orig_width} % {new_width} = {rebin_mod}')
    if len(msg)>0:
        msg = ', '.join(msg)
        common.logger.warning(f'Hist {name} inconsistent with requested binning ({msg})')
    return hist.rebin(rebin_factor).cut(VarHistogram.bins[0],VarHistogram.bins[-1])

def rebin_dict(hists, hist_var):
    for key in hists:
        if isinstance(hists[key],list):
            for i,entry in enumerate(hists[key]):
                hists[key][i] = check_rebin(entry,f'{key}[{i}]', hist_var)
        elif isinstance(hists[key],dict):
            for k,v in hists[key].items():
                hists[key][k] = check_rebin(v,f'{key}[{k}]', hist_var)
        else:
            hists[key] = check_rebin(hists[key],key, hist_var)
    return hists

def rebin_name(outfile, hist_var):
    VarHistogram = common.registered_varhists[hist_var]
    if VarHistogram.non_standard_binning:
        binw = VarHistogram.bins[1] - VarHistogram.bins[0]
        left = VarHistogram.bins[0]
        right = VarHistogram.bins[-1]
        outfile = outfile.replace(
            '.json',
            f'_binw{binw:0.2f}_range{left:.1f}-{right:.1f}.json'
        )
    return outfile

def rebin_file(file, hist_var):
    file2 = rebin_name(file, hist_var)
    if file2==file:
        return file
    with open(file,'r') as f:
        mths = json.load(f, cls=common.Decoder)
    mths = rebin_dict(mths, hist_var)
    with open(file2,'w') as f:
        json.dump(mths, f, cls=common.Encoder, indent=4)
    return file2

def basename(meta):
    """
    Generates a name based on signal metadata.
    """
    return (
        f'mz{meta["mz"]:.0f}_rinv{meta["rinv"]:.1f}_mdark{meta["mdark"]:.0f}'
        )

@scripter
def build_histogram(args=None):
    if args is None:
        # Read from sys.argv
        selection = common.pull_arg('selection', type=str).selection
        hist_var = common.pull_arg("hist_var", type=str, default='mt').hist_var
        lumi = common.pull_arg('--lumi', type=float, default=None, help='Luminosity in pb-1 (overrides defaults)').lumi
        year = common.pull_arg('--year', type=str, default=None, help='year (overrides metadata)').year
        fullyear = common.pull_arg('--fullyear', action="store_true", help='treat 2018 as one year instead of splitting into pre and post').fullyear
        skimfile = common.pull_arg('skimfile', type=str).skimfile
        change_bin_width(hist_var)
    else:
        # Use passed input
        selection, hist_var, lumi, year, fullyear, skimfile = args

    def get_variation(var):
        return skimfile.replace(".npz",f"_{var}.npz")

    # apply final selections
    def apply_selection(cols,year):
        metadata = cols.metadata
        # HEM veto
        startHEM = 319077
        if year=="2018PRE":
            if metadata["sample_type"]=="data":
                cols = cols.select(cols.arrays['run']<startHEM)
                cols.cutflow['preHEM'] = len(cols)
        elif year=="2018POST":
            if metadata["sample_type"]=="data":
                cols = cols.select(cols.arrays['run']>=startHEM)
                cols.cutflow['postHEM'] = len(cols)
            cols = common.apply_hemveto(cols)

        # signal region
        wp = common.split_bdt(selection) if "=" in selection else None
        if selection=='cutbased':
            cols = common.apply_cutbased(cols)
        elif selection.startswith('cutbased_ddt='):
            cols = common.apply_cutbased_ddt(cols, lumi, cut_val=wp)
        elif selection.startswith('rtcutbased_ddt='):
            cols = common.apply_rtcutbased_ddt(cols, lumi, cut_val=wp)
        elif selection.startswith('bdt='):
            cols = common.apply_bdtbased(cols, wp, lumi, ddt_map_file=common.DDT_FILE_BDTBASED)
        elif selection.startswith('rtbdt='):
            cols = common.apply_bdtbased(cols, wp, lumi, ddt_map_file=common.DDT_FILE_BDTBASED_RT_DDT)
        # control regions
        elif selection=='cutbasedCR':
            cols = common.apply_cutbasedCR(cols)
        elif selection=='cutbasedCRloose':
            cols = common.apply_cutbasedCRloose(cols)
        elif selection=='anticutbased':
            cols = common.apply_anticutbased(cols)
        elif selection.startswith('anticutbased_ddt='):
            cols = common.apply_anticutbased_ddt(cols,lumi,cut_val=wp)
        elif selection.startswith('antirtcutbased_ddt='):
            cols = common.apply_antirtcutbased_ddt(cols,lumi,cut_val=wp)
        elif selection=='antiloosecutbased':
            cols = common.apply_antiloosecutbased(cols)
        elif selection.startswith('antiloosecutbased_ddt='):
            cols = common.apply_antiloosecutbased_ddt(cols,lumi, cut_val=wp)
        elif selection.startswith('antiloosertcutbased_ddt='):
            cols = common.apply_antiloosertcutbased_ddt(cols,lumi, cut_val=wp)
        elif selection.startswith('antibdt='):
            cols = common.apply_bdtbased(cols,wp,lumi,anti=True)
        elif selection.startswith('antirtbdt='):
            cols = common.apply_bdtbased(cols,wp,lumi,anti=True,ddt_map_file=common.DDT_FILE_BDTBASED_RT_DDT)
        elif selection.startswith('antiloosebdt='):
            cols = common.apply_antiloosebdt(cols,wp,lumi,rt_ddt_file=None,ddt_map_file=common.DDT_FILE_BDTBASED)
        elif selection.startswith('antiloosertbdt='):
            cols = common.apply_antiloosebdt(cols,wp,lumi,rt_ddt_file=common.RT_DDT_FILE,ddt_map_file=common.DDT_FILE_BDTBASED_RT_DDT)
        elif selection=='preselection':
            pass
        elif selection=="preselection_minus":
            pass
        else:
            raise common.InvalidSelectionException(sel=selection)
        return cols

    hist_variants = {}
    cen_columns = svj.Columns.load(skimfile)
    metadata = cen_columns.metadata
    if year is None: year = str(metadata["year"])
    else:
        metadata["year"] = year
        lumi = common.lumis[year]

    # process 2018 samples twice as PRE and POST
    if year=="2018" and not fullyear:
        outfiles1 = build_histogram((selection,hist_var,None,"2018PRE",False,skimfile))
        outfiles2 = build_histogram((selection,hist_var,None,"2018POST",False,skimfile))
        return outfiles1+outfiles2

    # extract options from selection
    do_syst = True
    if selection=="preselection_only":
        selection = "preselection"
        do_syst = False

    common.logger.info(f'Selection: {selection}')
    cen_columns = apply_selection(cen_columns,year)
    w = None
    if metadata["sample_type"] != "data":
        w = common.get_event_weight(cen_columns,lumi)
        if metadata['sample_type'] == 'sig':
            w *= common.get_model_sf(cen_columns, var="cen")
        metadata['event_weight'] = common.get_single_event_weight(w)

    # Defining the histogram type to use
    VarHistogram = common.registered_varhists[hist_var]

    hist_central = VarHistogram(cen_columns, w)
    hist_central.metadata.update(metadata)
    # approach to large weight removal: remove isolated bins from histogram
    if metadata["sample_type"]=="bkg":
        hist_central = hist_central.mask(common.mask_isolated_bins(hist_central.vals))
    hist_variants['central'] = hist_central

    if metadata["sample_type"]=="sig" and do_syst and selection!="preselection_minus":
        # Scale
        good_scales = np.array([0,1,2,3,4,6,8])
        scale_weight = cen_columns.to_numpy(['scaleweights'])
        scale_weight = scale_weight[ak.num(scale_weight,axis=1)>np.max(good_scales)]
        scale_weight = scale_weight[:, good_scales]
        if len(scale_weight):
            weight_up = w * np.max(scale_weight, axis=-1) * cen_columns.metadata['scale_factor_up']
            weight_down = w * np.min(scale_weight, axis=-1) * cen_columns.metadata['scale_factor_down']
            hist_variants['scale_up'] = VarHistogram(cen_columns, weight_up)
            hist_variants['scale_down'] = VarHistogram(cen_columns, weight_down)

        # JEC/JER/JES
        def mth_jerjecjes(tag):
            col = svj.Columns.load(get_variation(tag))
            col = apply_selection(col,year)
            event_weight = common.get_event_weight(col,lumi) * common.get_model_sf(col, var="cen")
            return VarHistogram(col, event_weight)
        # JER, JEC treated as uncorrelated between years (but 2018PRE, 2018POST always correlated)
        sysyear = get_sysyear(year)
        hist_variants[f'jer{sysyear}_up'] = mth_jerjecjes('jer_up')
        hist_variants[f'jer{sysyear}_down'] = mth_jerjecjes('jer_down')
        hist_variants[f'jec{sysyear}_up'] = mth_jerjecjes('jec_up')
        hist_variants[f'jec{sysyear}_down'] = mth_jerjecjes('jec_down')
        hist_variants['jes_up'] = mth_jerjecjes('jesup_both')
        hist_variants['jes_down'] = mth_jerjecjes('jesdown_both')

        # PS
        ps_weights = w[:,None] * cen_columns.to_numpy(['ps_isr_up', 'ps_isr_down', 'ps_fsr_up', 'ps_fsr_down'])
        hist_variants['isr_up']   = VarHistogram(cen_columns, ps_weights[:,0])
        hist_variants['isr_down'] = VarHistogram(cen_columns, ps_weights[:,1])
        hist_variants['fsr_up']   = VarHistogram(cen_columns, ps_weights[:,2])
        hist_variants['fsr_down'] = VarHistogram(cen_columns, ps_weights[:,3])

        # PU
        # also uncorrelated between years
        pu_weights = cen_columns.to_numpy(['puweight', 'pu_sys_up', 'pu_sys_down'])
        hist_variants[f'pu{sysyear}_up'] = VarHistogram(cen_columns, w / pu_weights[:,0] * pu_weights[:,1])
        hist_variants[f'pu{sysyear}_down'] = VarHistogram(cen_columns, w / pu_weights[:,0] * pu_weights[:,2])

        # PDF
        pdf_weights = cen_columns.to_numpy(['pdf_weights'])
        # set massive unphysical weights to physical max
        pdf_max = np.max(pdf_weights, where=pdf_weights<100, initial=1)
        pdf_weights = np.clip(pdf_weights,a_min=None,a_max=pdf_max)
        pdf_weights /= pdf_weights[:,:1] # Divide by first pdf
        mu_pdf = np.mean(pdf_weights, axis=1)
        sigma_pdf = np.std(pdf_weights, axis=1)
        pdfw_up = (mu_pdf+sigma_pdf) / cen_columns.metadata['pdfw_norm_up']
        pdfw_down = (mu_pdf-sigma_pdf) / cen_columns.metadata['pdfw_norm_down']
        pdfw_down = np.clip(pdfw_down, a_min=0, a_max=None)
        hist_variants['pdf_up'] = VarHistogram(cen_columns, w*pdfw_up)
        hist_variants['pdf_down'] = VarHistogram(cen_columns, w*pdfw_down)

        # Jet var modelling
        jet_sf = common.get_model_sf(cen_columns, var="cen")
        jet_sf_up = common.get_model_sf(cen_columns, var="up")
        jet_sf_down = common.get_model_sf(cen_columns, var="down")
        hist_variants["jetvar_up"] = VarHistogram(cen_columns, w * jet_sf_up / jet_sf)
        hist_variants["jetvar_down"] = VarHistogram(cen_columns, w * jet_sf_down / jet_sf)
        hist_variants["jetvar_none"] = VarHistogram(cen_columns, w / jet_sf)

        # MC stats
        mc_stat_err = np.sqrt(np.histogram(VarHistogram._create_var_array(cen_columns), bins=hist_central.binning, weights=w**2)[0])
        for i in range(hist_central.nbins):
            cen_copy = hist_central.copy()
            cen_copy.vals[i] += mc_stat_err[i]
            hist_variants[f'mcstat{i}_{sysyear}_up'] = cen_copy
            cen_copy = hist_central.copy()
            cen_copy.vals[i] -= mc_stat_err[i]
            hist_variants[f'mcstat{i}_{sysyear}_down'] = cen_copy

    # save cutflow after applying final selection & after doing any copying (to avoid duplication)
    hist_variants['central'].cutflow = cen_columns.cutflow.copy()
    outdir = f'hists_{strftime("%Y%m%d")}'
    os.makedirs(outdir, exist_ok=True)
    process = osp.basename(skimfile).replace(".npz","")
    if metadata["sample_type"]=="data":
        # keep data era info to avoid overwriting
        process += '_'+osp.basename(osp.dirname(skimfile))
    if selection == "preselection_minus":
        process, skip_cut = process.split("_skip_cut-")
        selection += "-" + skip_cut
    outfile = f'{outdir}/{process}_sel-{selection}_year-{year}_{hist_var}.json'
    common.logger.info(f'Dumping histograms to {outfile}')
    with open(outfile, 'w') as f:
        json.dump(hist_variants, f, cls=common.Encoder, indent=4)
    return [outfile]

@scripter
def build_all_histograms():
    # Read from sys.argv
    selection = common.pull_arg('selection', type=str).selection
    hist_var_list = common.pull_arg("--hist_var_list", type=str, nargs='*', default=['mt']).hist_var_list
    fullyear = common.pull_arg('--fullyear', action="store_true", help='treat 2018 as one year instead of splitting into pre and post').fullyear
    threads = common.pull_arg('--threads', type=int, default=None).threads
    skimdir = common.pull_arg('skimdir', type=str).skimdir

    skims = expand_wildcards(skimdir)
    for hist_var in hist_var_list:
        change_bin_width(hist_var)
        args = [(selection, hist_var, None, None, fullyear, skim) for skim in skims]
        if threads==0:
            for arg in args:
                build_histogram(arg)
        else:
            with common.mp_pool(max_thread=threads) as p:
                p.map(build_histogram, args)

@scripter
def merge_histograms():
    selection = common.pull_arg('selection', type=str).selection
    histdir = common.pull_arg('histdir', type=str).histdir
    hist_var = common.pull_arg("--hist_var", type=str,default='mt').hist_var
    cat = common.pull_arg('--cat', type=str, required=True, choices=['sig','bkg','data']).cat
    years = common.pull_arg('--years', type=str, default=["2016","2017","2018PRE","2018POST"], nargs='*').years
    if histdir[-1]!='/': histdir += '/'

    change_bin_width(hist_var)
    def get_files(samples,years):
        files = []
        for year in years:
            for sample in samples:
                files += expand_wildcards(histdir+f'{sample}*_sel-{selection}_year-{year}_{hist_var}.json')
        return files

    def get_hists(file):
        with open(file,'r') as f:
            return json.load(f, cls=common.Decoder)

    def add_hists(hist,htmp):
        if hist is None:
            hist = htmp
        else:
            # once two hists (& their cutflows) are added, event_weight no longer needed -> set to 1
            hist += htmp
            if 'event_weight' in hist.metadata:
                hist.metadata['event_weight'] = 1.0
        return hist

    outdir = histdir.replace("hists","merged")
    os.makedirs(outdir, exist_ok=True)
    def write(hists,proc):
        if selection in proc:
            outfile = f'{outdir}/{proc}_{hist_var}.json'
        else:
            outfile = f'{outdir}/{proc}_sel-{selection}_{hist_var}.json'
        hists = rebin_dict(hists, hist_var)
        outfile = rebin_name(outfile, hist_var)
        common.logger.info(f'Dumping merged histograms to {outfile}')
        with open(outfile, 'w') as f:
            json.dump(hists, f, cls=common.Encoder, indent=4)

    lumi_total = sum(common.lumis[year] for year in years)
    def assign_metadata(hist):
        hist.metadata['selection'] = selection
        hist.metadata['year'] = years
        hist.metadata['lumi'] = lumi_total

    samples = {
        "data": ["JetHT", "HTMHT"],
        "bkg": ['QCD', 'TTJets', 'ZJets', 'WJets'],
        "sig": ['SVJ'],
    }
    files = get_files(samples[cat],years)
    default = 'central'
    if cat=="data":
        # just add them all up
        hist_variants = {
            cat : None,
        }
        for file in files:
            hist_variants[cat] = add_hists(hist_variants[cat],get_hists(file)[default])
        assign_metadata(hist_variants[cat])
        write(hist_variants,cat)

    elif cat=="sig":
        # just add years
        signals = defaultdict(list)
        for file in files:
            signals['_'.join(file.split('/')[1].split('_')[:-2])].append(file)
        for signal,sigfiles in signals.items():
            common.logger.info(f"Running merge for {signal}")
            for year in years:
              if len([f for f in sigfiles if year in f]) == 0:
                  common.logger.warning(f"Missing files for signal/year: {signal}/{year}")
            sighists = {year: get_hists([f for f in sigfiles if year in f][0]) for year in years}
            keys = list(sorted(set([key for y,h in sighists.items() for key in h])))
            hist_variants = {}
            for key in keys:
                hist_variants[key] = None
                # handle uncorrelated systematics (vary one year at a time)
                if '20' in key:
                    getter = lambda h: h.get(key,h[default])
                # correlated systematics must be present in all years
                else:
                    getter = lambda h: h[key]
                for year,sighist in sighists.items():
                    hist_variants[key] = add_hists(hist_variants[key],getter(sighist))
            assign_metadata(hist_variants[default])
            write(hist_variants,signal)
    elif cat=="bkg":
        # add up but keep components
        hist_variants = {
            cat : None
        }
        for b in samples["bkg"]:
            b = b.lower()
            hist_variants[b+'_individual'] = []
            hist_variants[b] = None
        for file in files:
            tmp = get_hists(file)[default]
            bkg = next((b for b in samples["bkg"] if b in file)).lower()
            hist_variants[bkg+'_individual'].append(tmp) # Save individual histogram
            hist_variants[bkg] = add_hists(hist_variants[bkg],tmp) # Add up per background category
            hist_variants[cat] = add_hists(hist_variants[cat],tmp) # Add up all
        assign_metadata(hist_variants[cat])
        write(hist_variants,cat)


# __________________________________________
# Plotting

def reorderLegend(ax,order,title):
    handles, labels = ax.get_legend_handles_labels()
    lhzip = list(zip(labels,handles))
    mapping = [labels.index(o) for o in order]
    labels, handles = zip(*[lhzip[i] for i in mapping])
    ax.legend(handles, labels, title=title)

class Plot:
    def __init__(self, meta):
        self.selection = meta['selection']
        self.fig, (self.top, self.bot) = plt.subplots(2,1, height_ratios=[3,1], figsize=(10,13))
        self.top.set_yscale('log')
        common.put_on_cmslabel(self.top, year = meta['lumi'] if 'lumi' in meta else meta['year'])
        self.top.set_ylabel('Event count')
        self.bot.set_ylabel('Ratio to central')
        self.bot.set_xlabel(r'$m_{T}$ (GeV)')
        self.legend_title = None

    def plot_hist(self, hist, denominator=None, label=None):
        ratio = np.ones(hist.nbins) if denominator is None else hist.vals / denominator.vals
        l = self.top.step(hist.binning[:-1], hist.vals, where='post', label=label)[0]
        self.bot.step(hist.binning[:-1], ratio, where='post', label=label, color=l.get_color())

    def set_xlim(self, *args, **kwargs):
        self.bot.set_xlim(*args, **kwargs)
        self.top.set_xlim(*args, **kwargs)

    def save(self, outfile='tmp.png', pdf=True, legend_order=None):
        self.top.text(
            0.02, 0.02,
            self.selection,
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=self.top.transAxes,
            usetex=True,
            fontsize=25
            )
        if legend_order is None:
            self.top.legend(title=self.legend_title)
        else:
            reorderLegend(self.top,legend_order,title=self.legend_title)
        outdir = osp.dirname(osp.abspath(outfile))
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(outfile, bbox_inches="tight")
        plt.savefig(outfile.replace('.png', '.pdf'), bbox_inches="tight")
        common.imgcat(outfile)

def get_sysyear(year):
    return year[:4]

def get_systs_uncorrelated():
    uncorrelated = [
        'jer',
        'jec',
        'pu',
        'stat',
    ]
    return uncorrelated

# years = None -> generic case (uncorrelated systs not expanded)
def get_systs(names=False,years=["2016","2017","2018"],smooth=False):
    uncorrelated = get_systs_uncorrelated()
    syst_names = {
        'scale': "Scales",
        'jer': "JER",
        'jec': "JEC",
        'jes': "JES",
        'isr': "ISR (parton shower)",
        'fsr': "FSR (parton shower)",
        'pu': "Pileup reweighting",
        'pdf': "PDF",
        'jetvar': "Model"
    }
    if smooth:
        syst_names.update({
            'stat': "MC statistical (fit)",
        })
        uncorrelated.remove('stat')
    else:
        syst_names.update({
            'stat': "MC statistical",
        })
    # expand uncorrelated systs
    if years is not None:
        if not isinstance(years,list): years = [years]
        # convert to sysyears
        years = [get_sysyear(year) for year in years]
        syst_names2 = {k:v for k,v in syst_names.items() if k not in uncorrelated}
        for unc in uncorrelated:
            for year in years:
                syst_names2[unc+year] = syst_names[unc]+f' ({year})'
        syst_names = syst_names2
    if names: return syst_names
    else: return list(syst_names.keys())

# should always be run before any rebinning
def make_stat_combined(mths,sysyear):
    if f'stat{sysyear}_up' not in mths.keys():
        stat_up = mths['central'].copy()
        stat_down = mths['central'].copy()
        i = 0
        while f'mcstat{i}_{sysyear}_up' in mths.keys():
            stat_up.vals[i] = mths[f'mcstat{i}_{sysyear}_up'].vals[i]
            stat_down.vals[i] = mths[f'mcstat{i}_{sysyear}_down'].vals[i]
            i += 1

        mths[f'stat{sysyear}_up'] = stat_up
        mths[f'stat{sysyear}_down'] = stat_down
    return mths

@scripter
def plot_systematics():
    hist_var = common.pull_arg('--hist_var', type=str, default='mt').hist_var
    yrange = common.pull_arg('--yrange', type=float, nargs=2, default=None).yrange
    json_file = common.pull_arg('jsonfile', type=str).jsonfile
    change_bin_width(hist_var)
    with open(json_file) as f:
        mths = json.load(f, cls=common.Decoder)
    meta = mths['central'].metadata
    years = meta['year']
    if not isinstance(years,list): years = [years]
    systs = get_systs(years=years,smooth="smooth" in json_file)
    for year in years:
        sysyear = get_sysyear(year)
        mths = make_stat_combined(mths,sysyear)

    mths = rebin_dict(mths, hist_var)
    n = mths['central'].vals.sum()
    common.logger.info(f'central integral: {n}')
    common.logger.info(f'central metadata:\n{mths["central"].metadata}')

    model_str = osp.basename(json_file).replace(".json","")
    outdir = f'plots_{strftime("%Y%m%d")}_{model_str}'
    os.makedirs(outdir, exist_ok=True)

    for syst in systs:
        plot = Plot(meta)
        plot.plot_hist(mths['central'], label='Central')
        plot.plot_hist(mths[f'{syst}_up'], mths['central'], f'{syst} up')
        plot.plot_hist(mths[f'{syst}_down'], mths['central'], f'{syst} down')
        if syst == "jetvar":
            plot.plot_hist(mths[f'{syst}_none'], mths['central'], f'{syst} none')
        if yrange is not None:
            plot.bot.set_ylim(yrange[0],yrange[1])
        plot.save(f'{outdir}/{syst}.png')

@scripter
def plot_bkg():
    mtmin = common.pull_arg('--mtmin', type=float, default=180.).mtmin
    mtmax = common.pull_arg('--mtmax', type=float, default=650.).mtmax
    rebin = common.pull_arg('--rebin', type=int, default=1).rebin
    json_file = common.pull_arg('jsonfile', type=str).jsonfile
    with open(json_file) as f:
        mths = json.load(f, cls=common.Decoder)

    sig_json_file = common.pull_arg('sigjsonfile', type=str, nargs='*').sigjsonfile
    do_signal = len(sig_json_file) > 0

    h = mths['bkg'].rebin(rebin).cut(mtmin,mtmax)
    h_sum = np.sum(h.vals)
    binning = h.binning
    nbins = h.nbins
    zero = np.zeros(nbins)

    with common.quick_ax(outfile=f"bkg_sel-{h.metadata['selection']}.pdf") as ax:
        # for bkg in ['zjets', 'wjets', 'ttjets', 'qcd']:
        for label, bkg in [("QCD", 'qcd'), ("TT+Jets", 'ttjets'), ("W+Jets", 'wjets'), ("Z+Jets", 'zjets')]:
            fraction = np.sum(mths[bkg].rebin(rebin).cut(mtmin,mtmax).vals) / h_sum * 100.
            ax.fill_between(h.binning[:-1], zero, h.vals, step='post', label=label+ f" ({fraction:.2f}\\%)")
            h.vals -= mths[bkg].rebin(rebin).cut(mtmin,mtmax).vals

        if do_signal:
            with open(sig_json_file[0]) as f:
                sig = json.load(f, cls=common.Decoder)['central']
                sig = sig.rebin(rebin).cut(mtmin,mtmax)
                ax.step(
                    sig.binning[:-1], sig.vals, '--k',
                    where='post', label=basename(sig.metadata)
                    )

        ax.set_yscale('log')
        ax.legend()
        ax.set_ylabel('Event count')
        ax.set_xlabel(r'$m_{T}$ (GeV)')
        common.put_on_cmslabel(ax)
        ax.text(
            0.02, 0.02,
            h.metadata['selection'],
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=ax.transAxes,
            usetex=True,
            fontsize=25
            )

def get_xye(hist):
    x = hist.binning
    x = (x[:-1]+x[1:])/2 # bin centers
    y = hist.vals
    errs = hist.errs
    return x,y,errs

def do_loess(hist,span,do_gcv=False):
    from uloess import loess
    x, y, errs = get_xye(hist)
    # safety check for empty bins w/ empty errs
    for i,yy in enumerate(y):
        if yy==0:
            errs[i] = 1
    # 1sigma interval
    try:
        pred, conf_int, gcv, q_rough = loess(x, y, errs, deg=2, alpha=0.683, span=span)
    except np.linalg.LinAlgError as err:
        gcv = 1e10
        print(err)
    if do_gcv:
        return gcv, q_rough
    else:
        return pred, conf_int

@scripter
def smooth_shapes():
    span_val = common.pull_arg('--span', type=float, default=0.25, help="span value").span
    span_min = common.pull_arg('--spanmin', type=float, default=0.05, help="minimum span value").spanmin # if span is too small, no points are included
    span_max = common.pull_arg('--spanmax', type=float, default=0.5, help="maximum span value").spanmax
    simple = common.pull_arg('--simple', default=False, action="store_true", help="use simple GCV-based optimization, ignoring roughness").simple
    leak_tol = common.pull_arg('--leaktol', type=float, default=1.0, help="relative tolerance for roughness leakage knee").leaktol
    min_run = common.pull_arg('--minrun', type=float, default=2, help="min consecutive spans with roughness leakage < max").minrun
    gcv_tol = common.pull_arg('--gcvtol', type=float, default=0.0, help="relative GCV tolerance").gcvtol
    do_opt = common.pull_arg('--optimize', type=int, default=0, help="optimize span value using n values").optimize
    default = common.pull_arg('--default', type=str, default='central', help="default histogram for metadata").default
    target = common.pull_arg('--target', type=str, default=default, help="optimize only based on target hist").target
    debug = common.pull_arg('--debug', default=False, action="store_true", help="debug optimization").debug
    var = common.pull_arg('--variation', type=str, default=None, help="MT variation to debug").variation
    save = common.pull_arg('--save', default=False, action="store_true", help="save debug variation output").save
    mtmin = common.pull_arg('--mtmin', type=float, default=None).mtmin
    mtmax = common.pull_arg('--mtmax', type=float, default=None).mtmax
    norm = not common.pull_arg('--unnorm', default=False, action="store_true", help="fit unnormalized shape").unnorm
    ratio_span = common.pull_arg('--ratio-span', type=float, default=0.,
        help="signal systematics: smooth the ratio variation/central (LOESS with this span) and apply it to the smoothed "
             "central, instead of smoothing each variation separately (0: off)").ratio_span
    json_files = common.pull_arg('jsonfiles', nargs='+', type=str).jsonfiles

    with common.mp_pool() as p:
        args = [(span_val, span_min, span_max, simple, leak_tol, min_run, gcv_tol, do_opt, default, target, debug, var, save, mtmin, mtmax, norm, f, ratio_span) for f in json_files]
        p.starmap(smooth_shape_single, args)


def smooth_ratio_to_central(hist, raw_central, central_smooth, span):
    """Variation = smoothed central x LOESS-smoothed ratio of the raw variation to the raw central.
    Variations are made from the same MC events as the central template, so their bin-to-bin differences are
    migrations with the MC statistical noise of the sample; smoothing each variation separately (with the central's
    span) keeps that noise in the ratio where the MC statistics are low, and the fit then sees a noisy shape
    systematic scaled by r. The ratio is smoothed with weights from the central's relative MC precision per bin."""
    ok = raw_central.vals > 0
    ratio = hist.copy()
    ratio.vals = np.where(ok, hist.vals / np.where(ok, raw_central.vals, 1.), 1.)
    ratio.errs = np.where(ok, raw_central.errs / np.where(ok, raw_central.vals, 1.), 1.)
    pred, _ = do_loess(ratio, span=span)
    out = central_smooth.copy()
    out.vals = central_smooth.vals * pred
    out.errs = central_smooth.errs * pred
    return out


def smooth_shape_single(span_val, span_min, span_max, simple, leak_tol, min_run, gcv_tol, do_opt, default, target, debug, var, save , mtmin, mtmax, norm, json_file, ratio_span=0.):
    # keep only regions with at least min_run consecutive True values
    # (avoid one-point outliers)
    def consec_true_mask(mask, min_run):
        mask = np.asarray(mask, dtype=bool)
        if min_run <= 1:
            return mask
        out = np.zeros_like(mask)
        n = len(mask)
        i = 0
        while i < n:
            if not mask[i]:
                i += 1
                continue
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_run:
                out[i:j] = True
            i = j
        return out

    with open(json_file) as f:
        mths = json.load(f, cls=common.Decoder)
    h_default = mths[default]
    common.logger.info(f'{default} metadata:\n{h_default.metadata}')

    # loop over central and systematics
    if h_default.metadata["sample_type"]=="sig":
        year = h_default.metadata["year"]
        if not isinstance(year,str) and not isinstance(year,list): year = str(int(year))
        variations = get_systs(years=year)
        variations = [v for v in variations if not v.startswith('stat')]
        variations = [var+'_up' for var in variations]+[var+'_down' for var in variations]+["jetvar_none"]
        variations = [default]+variations
    else:
        variations = [default]

    # find optimization target
    if len(target)>0:
        if target not in variations:
            raise ValueError("Unknown target {} (known: {})".format(target, ', '.join(variations)))
        # put target first
        variations = [target]+[v for v in variations if v!=target]

    cut_args = {}
    if mtmin is not None: cut_args["xmin"] = mtmin
    if mtmax is not None: cut_args["xmax"] = mtmax
    mths_new = {}
    save_all = True
    if var is not None:
        target = var
        variations = [var]
        save_all = False
    central_uncut = None     # smoothed central before the mT cut (for --ratio-span)
    for var in variations:
        hist = mths[var]
        hyield = hist.vals.sum()
        common.logger.info(f'{var} integral: {hyield}')
        meta = hist.metadata

        if ratio_span > 0 and var != default and h_default.metadata["sample_type"] == "sig":
            if central_uncut is None:
                raise ValueError("--ratio-span needs the normalised central template smoothed before the variations "
                                 "(not available with --unnorm, --var, or a --target other than the central)")
            hsmooth = smooth_ratio_to_central(hist, mths[default], central_uncut, ratio_span).cut(**cut_args)
            meta["ratio_span"] = ratio_span
            hsmooth.metadata = meta
            mths_new[var] = hsmooth
            continue

        # normalize shape to unit area, then scale prediction by original yield
        if norm: hist = hist*(1./hyield)

        if do_opt>0 and (var==target or len(target)==0):
            spans = np.linspace(span_min,span_max,do_opt,endpoint=False) # skip 1
            scans = [do_loess(hist, span, do_gcv=True) for span in spans]
            # list of pairs -> pair of lists
            gcvs, q_rough = [np.array(qty) for qty in zip(*scans)]
            if simple:
                span_val = spans[np.argmin(gcvs)]
                if debug: print(f"Chosen span = {span_val}")
            else:
                # optimization procedure:
                # 1. find "knee" where roughness_leakage stops decreasing rapidly
                # 2. mask out fluctuations using min_run
                # 3. pick smallest of these spans w/ gcv in tolerance
                from uloess import knee
                knee_index = knee(spans, q_rough, leak_tol)
                span_knee = spans[knee_index]
                q_knee = q_rough[knee_index]
                # this is equivalent to applying the tolerance to log(q_rough)
                q_thresh = q_knee**leak_tol
                if debug: print("q_thresh",q_thresh)
                # only allow tolerance to include smaller spans in feasible range, not larger ones
                feasible_raw = (q_rough <= q_knee) | ((q_rough <= q_thresh) & (spans < span_knee))
                if min_run>1:
                    feasible = consec_true_mask(feasible_raw, min_run)
                # fallback: ignore consecutive run requirement
                if not np.any(feasible):
                    if debug: print("Warning: ignoring consecutive run requirement")
                    feasible = feasible_raw
                if debug: print(f"Feasible spans: {np.min(spans[feasible])}, {np.max(spans[feasible])}")
                best_feasible_gcv = np.min(gcvs[feasible])
                feasible_indices = np.where(feasible)[0]
                best_feasible_idx = feasible_indices[np.argmin(gcvs[feasible])]
                best_feasible_gcv_span = spans[best_feasible_idx]
                if debug: print(f"Best feasible span = {best_feasible_gcv_span} with gcv = {best_feasible_gcv}")
                plateau_gcv = best_feasible_gcv*(1.0+gcv_tol)
                if gcv_tol>0:
                    plateau_mask = feasible & (gcvs < plateau_gcv)
                    chosen_idx = np.flatnonzero(plateau_mask)[0]
                else:
                    chosen_idx = best_feasible_idx
                if debug: print(f"Lowest feasible span = {spans[chosen_idx]} with gcv = {gcvs[chosen_idx]}")
                span_val = spans[chosen_idx]
            if debug: print('\n'.join(['{} {} {}'.format(span,gcv,q) for span,gcv,q in zip(spans,gcvs,q_rough)]))
            meta["span"] = span_val
            meta["gcvs"] = list(gcvs)
            if not simple:
                meta["q_rough"] = list(q_rough)
                meta["q_knee"] = q_knee
                meta["q_thresh"] = q_thresh
                meta["feasible_spans"] = [np.min(spans[feasible]), np.max(spans[feasible])]
                meta["best_feasible_gcv"] = best_feasible_gcv
                meta["plateau_gcv"] = plateau_gcv
            meta["span_min"] = span_min
            meta["span_max"] = span_max
            meta["span_pts"] = do_opt

        pred, conf = do_loess(hist,span=span_val)

        hsmooth = hist.copy()
        hsmooth.vals = np.where((hist.vals == 0) & (pred < 0), 0.0, pred)
        hsmooth.errs = conf[1] - pred

        # cuts applied *after* interpolation
        if norm:
            hsmooth = hsmooth*hyield
            if var == default:
                central_uncut = hsmooth.copy()
            hsmooth = hsmooth.cut(**cut_args)
        mths_new[var] = hsmooth
        if var=='central':
            inames = ['down','up']
            # computed after normalization to original yield (above)
            ivals = [hsmooth.vals-hsmooth.errs, hsmooth.vals+hsmooth.errs]
            for iname,ival in zip(inames,ivals):
                hstat = hsmooth.copy()
                # avoid errors going negative
                hstat.vals = np.clip(ival,0,None)
                hstat.errs = np.zeros_like(hstat.vals)
                mths_new[f'stat_{iname}'] = hstat

        # store updated metadata to avoid duplicating for stat variations
        hsmooth.metadata = meta

    outdir = os.path.dirname(json_file).replace("hists","smooth").replace("merged","smooth")
    os.makedirs(outdir, exist_ok=True)
    if save_all:
        # copy any other contents from original input
        # omitting mcstat uncertainties, which are replaced by overall confidence interval
        # and applying cuts
        for key in mths.keys():
            if key not in mths_new.keys() and 'mcstat' not in key:
                if isinstance(mths[key],list):
                    mths_new[key] = []
                    for entry in mths[key]:
                        mths_new[key].append(entry.cut(**cut_args))
                else:
                    mths_new[key] = mths[key].cut(**cut_args)
        outfile = outdir+'/'+osp.basename(json_file).replace(".json","_smooth.json")
    elif save:
        outfile = outdir+'/'+osp.basename(json_file).replace(".json","_smooth_{}.json".format(var))
    if save_all or save:
        with open(outfile, 'w') as f:
            json.dump(mths_new, f, indent=4, cls=common.Encoder)


@scripter
def plot_smooth():
    mtmin = common.pull_arg('--mtmin', type=float, default=180.).mtmin
    mtmax = common.pull_arg('--mtmax', type=float, default=650.).mtmax
    default = common.pull_arg('--default', type=str, default='central', help="default histogram for metadata").default
    var = common.pull_arg('--variation', type=str, default=default, help="MT variation to plot (or 'all')").variation
    noratio = common.pull_arg('--no-ratio', default=False, action="store_true", help="skip ratio").no_ratio
    names = common.pull_arg('--names', type=str, nargs='*', default=[], help="legend names for files").names
    json_files = common.pull_arg('jsonfiles', type=str, nargs='+').jsonfiles

    if len(names)==0:
        names = ["sample {}".format(i+1) for i in range(len(json_files))]
    elif len(names)>0 and len(names)!=len(json_files):
        raise ValueError("Mismatch between length of names ({}) and length of files ({})".format(len(names),len(files)))

    vars = [var]
    if var=='all':
        vars = get_systs()
        vars = [var+'_up' for var in vars]+[var+'_down' for var in vars] + ["jetvar_none"]
        vars = [default]+vars

    mths = []
    for json_file in json_files:
        with open(json_file) as f:
            mths.append(json.load(f, cls=common.Decoder))
            omit = [var for var in vars if var not in mths[-1].keys()]
            vars = [var for var in vars if var in mths[-1].keys()]
            if len(omit)>0: print("Omitting keys missing in {}: {}".format(json_file,', '.join(omit)))

    model_str = osp.basename(json_file).replace(".json","")
    outdir = f'plot_smooth_{strftime("%Y%m%d")}'
    os.makedirs(outdir, exist_ok=True)

    for var in vars:
        meta = mths[0][default].metadata
        plot = Plot(meta)
        legend_order = []
        h_denom = None

        for i,(mth,name) in enumerate(zip(mths,names)):
            hist = mth[var].cut(mtmin,mtmax)

            x, y, e = get_xye(hist)
            if 'smooth' in name:
                ys_up = y+e
                ys_dn = y-e
                e = None
            legend_order.append(name)
            line = plot.top.errorbar(x,y,yerr=e,label=legend_order[-1])
            line = line[0]

            if 'smooth' in name:
                plot.top.fill_between(x,ys_dn,ys_up,alpha=0.33,color=line.get_color())

            if not noratio:
                if h_denom is None:
                    h_denom = y
                    plot.bot.set_ylabel('Ratio to {}'.format(name))
                else:
                    plot.bot.plot(x,y/h_denom,color=line.get_color())

        plot.save(f'{outdir}/{model_str}_{var}.png',legend_order=legend_order)


@scripter
def plot_span_opt_diagnostic():
    default = common.pull_arg('--default', type=str, default='central', help="default histogram for metadata").default
    json_file = common.pull_arg('jsonfile', type=str).jsonfile

    with open(json_file) as f:
        mths = json.load(f, cls=common.Decoder)
    h_default = mths[default]
    meta = h_default.metadata

    spans = np.linspace(meta["span_min"],meta["span_max"],int(meta["span_pts"]),endpoint=False)
    feasible_spans = meta["feasible_spans"]
    gcvs = meta["gcvs"]
    best_gcv = meta["best_feasible_gcv"]
    plateau_gcv = meta["plateau_gcv"]
    q_rough = meta["q_rough"]
    q_knee = meta["q_knee"]
    q_thresh = meta["q_thresh"]

    chosen = meta["span"]

    fig, ax1 = plt.subplots()

    ax1.plot(spans, gcvs, marker="o", color="blue", label="GCV")
    ax1.axvline(chosen, linestyle='-', color="red", label="chosen span")
    ax1.axhline(best_gcv, linestyle="--", color="blue", label="best GCV")
    if plateau_gcv != best_gcv: ax1.axhline(plateau_gcv, linestyle=":", color="blue", label="plateau GCV")
    ax1.axvline(feasible_spans[0], linestyle="-.", color="black", label="feasible span range")
    ax1.axvline(feasible_spans[1], linestyle="-.", color="black")
    ax1.set_xlabel("span")
    ax1.set_ylabel("GCV", color="blue")
    ax1.set_yscale("log")
    ax1.tick_params(axis='y', which='both', right=False, labelright=False)
    ax1.tick_params(axis="y", which='both', labelcolor="blue", color="blue")
    ax1.spines["left"].set_color("blue")
    ax1.spines['left'].set_edgecolor("blue")

    ax2 = ax1.twinx()
    ax2.plot(spans, q_rough, marker="s", color="orange", label="roughness")
    ax2.axhline(q_knee, linestyle="--", color="orange", label="knee")
    if q_thresh != q_knee: ax2.axhline(q_thresh, linestyle=":", color="orange", label="relaxed")
    ax2.set_yscale("log")
    ax2.set_ylabel("roughness", color="orange")
    ax2.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax2.tick_params(axis="y", which='both', labelcolor="orange", color="orange")
    ax2.spines["right"].set_color("orange")
    ax2.spines['left'].set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, frameon=True, framealpha=0.8, facecolor="white", edgecolor="black", loc="upper right")

    model_str = osp.basename(json_file).replace(".json","")
    outdir = f'plot_span_opt_diagnostic_{strftime("%Y%m%d")}'
    os.makedirs(outdir, exist_ok=True)
    outfile = f'{outdir}/{model_str}_{default}.png'
    plt.savefig(outfile)


def get_yield(hist):
    return hist.vals.sum()

def pct_diff(central,syst):
    return abs(1-syst/central)*100

def printSigFigs(num,fig,maxdec):
    # use scientific notation to get sig figs, then convert back to regular notation
    sci = "{0:>.{1}e}".format(num,fig-1)
    val = sci.split('e')[0]
    expo = int(sci.split('e')[1])
    pindex = val.find('.')
    if expo==0:
        result = val
    elif expo>0:
        if pindex==-1:
            result = val
        else:
            # move decimal point
            rval = val.replace('.','')
            pindex2 = pindex+expo
            # trailing zeroes
            if pindex2 >= len(rval): result = rval+'0'*(pindex2-len(rval))
            else: result = rval[0:pindex2]+'.'+rval[pindex2:]
    else: # expo<0
        rval = val if pindex==-1 else val.replace('.','')
        # add leading zeroes
        result = "0."+'0'*(abs(expo)-1)+rval
    # recursive truncation
    if '.' in result and len(result.split('.')[1])>maxdec:
        if fig==1:
            rnum = round(num,abs(expo)-1)
            if rnum==0.0: return "0.0"
            else: return printSigFigs(rnum,fig,maxdec)
        else: return printSigFigs(num,fig-1,maxdec)
    else:
        if set(result.replace('.',''))==set([0]): return "0.0"
        else: return result

@scripter
def systematics_table():
    hist_var = common.pull_arg('--hist_var', type=str, default='mt').hist_var
    qtyrange = common.pull_arg('--qtyrange', metavar=("qty min max"), default=[], type=str, action='append', nargs=3).qtyrange
    minimum = common.pull_arg('--minimum', type=float, default=0.01, help="minimum value to display, smaller values rounded to 0").minimum
    skimdir = common.pull_arg('skimdir', type=str).skimdir
    skims = expand_wildcards(skimdir)
    change_bin_width(hist_var)

    # set up qty range limitations
    qtyfilters = []
    for iq,qr in enumerate(qtyrange):
        qtyfilters.append(lambda md: float(qtyrange[iq][1])<=md[qtyrange[iq][0]] and md[qtyrange[iq][0]]<=float(qtyrange[iq][2]))

    # needs to be kept in sync w/ boostedsvj/svj_limits/boosted_fits.py:gen_datacard()
    flat_systs = {
        'lumi': 0.73,
        'trigger_cr': 2.0,
        'trigger_sim': 2.1,
    }

    unc_systs = get_systs_uncorrelated()
    systs = get_systs(names=True,years=None)
    systs.update({
        'lumi': "Luminosity",
        'trigger_cr': "Trigger (CR)",
        'trigger_sim': "Trigger (MC)",
    })

    # indexing: [year][syst]
    syst_effects = defaultdict(lambda: defaultdict(lambda: (1e10, 0.0)))
    def update_effect(year,syst,effect):
        syst_effects[year][syst] = (min(effect, syst_effects[year][syst][0]), max(effect, syst_effects[year][syst][1]))
    for skim in skims:
        with open(skim) as f:
            mths = json.load(f, cls=common.Decoder)
        meta = mths['central'].metadata
        year = meta['year']
        if isinstance(year, list): year = "merged"
        if not isinstance(year,str): year = str(int(year))
        mths = make_stat_combined(mths,get_sysyear(year))
        mths = rebin_dict(mths, hist_var )
        central = mths['central']
        central_yield = get_yield(central)

        passed = True
        for qf in qtyfilters:
            if not qf(meta):
                passed = False
                break
        if not passed:
            continue

        total = 0
        for syst in sorted(systs.keys()):
            syst_effect = 0
            asyst_list = []
            # uncorrelated systs stored with sysyear naming (for datacard creation)
            if syst in unc_systs:
                if year == "merged":
                    if syst != "stat":
                        asyst_list += [syst + y for y in ["2016", "2017", "2018"]]
                    else:
                        asyst_list += [syst]
                else:
                    asyst_list.append(syst + get_sysyear(year))
            else:
                asyst_list.append(syst)

            for asyst in asyst_list:
                if f'{asyst}_up' in mths:
                    syst_up_yield = get_yield(mths[f'{asyst}_up'])
                    syst_dn_yield = get_yield(mths[f'{asyst}_down'])
                    syst_effect = max(pct_diff(central_yield,syst_up_yield),pct_diff(central_yield,syst_dn_yield))
                elif syst in flat_systs:
                    syst_effect = flat_systs[syst]
                else:
                    common.logger.warning(f'could not find systematic: {syst}')
                    continue
                update_effect(year,syst,syst_effect)
                total += syst_effect**2
        total = np.sqrt(total)
        update_effect(year,"total",total)

    # keep correct order
    all_years = ["2016","2017","2018","2018PRE","2018POST"]
    years = [y for y in all_years if y in list(syst_effects.keys())]
    # add overall
    if len(years)>1:
        years.append("Overall")
        for syst in list(systs.keys())+["total"]:
            syst_effects["Overall"][syst] = (min([syst_effects[year][syst][0] for year in years]), max([syst_effects[year][syst][1] for year in years]))

    # print settings
    sigfig = 2
    maxdec = int(abs(np.log10(minimum)))

    print(" & ".join(["Systematic"]+list(syst_effects.keys()))+r" \\")
    print(r"\hline")
    def print_syst_row(syst):
        cols = [systs[syst]]
        for era in syst_effects.keys():
            tmin = syst_effects[era][syst][0]
            tmax = syst_effects[era][syst][1]
            smin = printSigFigs(tmin,sigfig,maxdec)
            smax = printSigFigs(tmax,sigfig,maxdec)
            # don't bother to display a range if values are equal within precision
            if abs(tmax-tmin)>minimum and smin != smax:
                trange = smin+"--"+smax
            else:
                trange = smax
            cols.append(trange)
        print(" & ".join(cols)+r" \\")
    for syst in sorted(systs):
        print_syst_row(syst)
    print(r"\hline")
    systs["total"] = "total"
    print_syst_row("total")

@scripter
def acc():
    qty = common.pull_arg('--qty', type=str, required=True, help="signal qty").qty
    skimdir = common.pull_arg('skimdir', type=str).skimdir
    skims = expand_wildcards(skimdir)
    def get_from_cutflow(cutflow,index):
        return float(list(cutflow.items())[index][1])
    for skim in skims:
        with open(skim) as f:
            mths = json.load(f, cls=common.Decoder)
        meta = mths['central'].metadata
        cutflow = mths['central'].cutflow
        print(meta[qty],get_from_cutflow(cutflow,-1)/get_from_cutflow(cutflow,0)*100)

@scripter
def merge(args=None):
    if args is None:
        outfile = common.pull_arg('-o', '--outfile', type=str).outfile
        json_files = common.pull_arg('jsonfiles', type=str, nargs='+').jsonfiles
    else:
        outfile, json_files = args

    d = {}
    for json_file in json_files:
        common.logger.info(f'Merging {json_file}')
        with open(json_file) as f:
            mths = json.load(f, cls=common.Decoder)
            d.update(mths)

    if not outfile:
        for f in json_files:
            f = osp.abspath(f)
            if 'mz' not in f: continue
            outfile = osp.basename(osp.abspath(f)) + '_withbkg.json'
            break
        else:
            outfile = 'out.npz'

    common.logger.info(f'Dumping to {outfile}')
    with open(outfile, 'w') as f:
        json.dump(d, f, indent=4, cls=common.Encoder)


@scripter
def ls():
    infile = common.pull_arg('infile', type=str).infile
    if infile.endswith('.json'):
        with open(infile, 'r') as f:
            d = json.load(f, cls=common.Decoder)
        from pprint import pprint
        pprint(d)
    elif infile.endswith('.root'):
        array = svj.open_root(infile, load_gen=True, load_jerjec=True)
        print(f'Found {len(array)} in {infile}')
        print(f'Metadata:\n{array.metadata}')
    else:
        print(f'Unknown file format: {infile}')


if __name__ == '__main__':
    scripter.run()
