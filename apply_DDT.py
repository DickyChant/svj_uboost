#==============================================================================
# apply_DDT.py ----------------------------------------------------------------
#------------------------------------------------------------------------------
# Author(s): Brendan Regnery, Yi-Mu Chen, Sara Nabili -------------------------
#------------------------------------------------------------------------------
# Applies a DDT to a trained BDT model ----------------------------------------
#    (Designed Decorrelated Tagger, https://arxiv.org/pdf/1603.00027.pdf) -----
#------------------------------------------------------------------------------

import os, os.path as osp, glob, pickle, logging, warnings, json, math, re
from time import strftime
from collections import OrderedDict
from common import create_DDT_map_dict, calculate_varDDT, add_ddt_smoothing_info
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg') # prevents opening displays (fast), must use before pyplot
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc
from scipy.ndimage import gaussian_filter
import pandas as pd
import xgboost as xgb
import mplhep as hep
import tqdm
hep.style.use("CMS") # CMS plot style
import svj_ntuple_processing as svj
import seutils as se

np.random.seed(1001)

from common import read_training_features, logger, lumis, DATADIR, Columns, time_and_log, imgcat, set_mpl_fontsize, columns_to_numpy, apply_rt_signalregion, calc_bdt_scores, expand_wildcards, signal_xsecs, MTHistogram, get_event_weight, compute_bkg_isolatedevt_mask, SELECTION_RT_SIGNAL_REGION

#------------------------------------------------------------------------------
# Global variables and user input arguments -----------------------------------
#------------------------------------------------------------------------------

training_features = []

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some inputs.')

    # Which analysis to run (BDT or cut-based)
    parser.add_argument('--analysis_type', default='BDT-based', choices=['RT', 'BDT-based', 'cut-based'], help='Computation of the DDT targetting the RT, BDT or cut-based analysis')

    parser.add_argument('--bkg_files', default='data/bkg_20241030/Summer20UL*/QCD*.npz', help='Background data files (default is QCD only DDT)')
    parser.add_argument('--sig_files', default='data/sig_20241030/sig_*/*.npz', help='Signal data files')

    # BDT and ddt model
    parser.add_argument('--bdt_file', default='models/svjbdt_obj_rev_version.json', help='BDT model file')
    parser.add_argument('--ddt_map_file', default='models/bdt_ddt_AN_v6.json', help='DDT map file')

    parser.add_argument('--rt_sel', type=float, default=1.18, help='RT selection defining the signal region') # TODO: update to optimal value
    parser.add_argument('--rt_ddt_file', help='DDT map file for RT selection, leave none to use non DDT RT selection')
    parser.add_argument('--smooth_sigmas', nargs='*', type=float, default=[], help='Store with the DDT map the information for N_eff-weighted on-the-fly smoothing (calculate_varDDT smooth_sigma) of these widths in bins; the stored thresholds are unchanged. If the map file exists, the information is added to it (same inputs and --rt_sel as when it was made). With --rt_ddt_file, that RT map must carry the same widths')

    # The default value of 0.65 was the optimal cut value determined. If the training or selection changes,
    # the value should be adapted accordingly
    parser.add_argument('--sig_bdt_cut', type=float, default=0.67, help='BDT cut for signal plotting (current optimal cut is 0.67)')

    # Choose the variable cut values that you want to make for the DDT
    parser.add_argument('--var_cuts', nargs='+', type=float,  help='List of variable cuts values')

    # Allowed plots: 2D DDT maps, Background Scores vs MT, FOM significance,
    #                Signal mt spectrums for one BDT working point,  one signal mt spectrum for many BDT working points
    allowed_plots = ['2D_DDT_map', 'bkg_scores_mt', 'sig_scores_mt', 'fom_significance', 'fom_mt_scan', 'sig_mt_single_BDT', 'one_sig_mt_many_bdt']
    parser.add_argument('--plot', nargs='*', type=str, default=[], choices=allowed_plots, help='Plots to make')

    # Add verbosity level argument
    parser.add_argument('--verbosity', type=int, choices=[0, 1, 2], default=1, help="Set verbosity level: 0 (silent), 1 (default), 2 (detailed)")

    return parser.parse_args()

#------------------------------------------------------------------------------
# User defined functions ------------------------------------------------------
#------------------------------------------------------------------------------

def FOM(s, b):
    '''
    The significance for signal vs. background derived from the CLs method
    '''
    FOM = np.sqrt(2*((s+b)*np.log(1+s/b)-s))
    return FOM

def simple_fig():
    fig, ax = plt.subplots(figsize=(10, 8))
    hep.cms.label(rlabel="(13 TeV)")
    return fig, ax


def save_plot(plt, plt_name, flag_tight_layout=True, **kwargs):
    '''
    Save the plots, save some lines of code
    '''
    if flag_tight_layout == True: plt.tight_layout()
    print(f"Saving plot {plt_name}")
    plt.savefig(f'plots/{plt_name}.png', **kwargs)
    plt.savefig(f'plots/{plt_name}.pdf', **kwargs)


def make_ddt_inputs(input_files: list[str], all_features: list[str]):
    X_list, W_list = [], []
    for input_file in input_files:
        col = Columns.load(input_file)
        x = col.to_numpy(all_features)
        w = get_event_weight(col, lumis[str(col.metadata["year"])])
        if len(x) == 0: # Skipping length 0 arrays, as this messes up the masking creating routine
            continue
        # Only construct mask for background sample
        mask = compute_bkg_isolatedevt_mask(x[:,-3]) if col.metadata['sample_type'] == 'bkg' else np.ones_like(w, dtype=bool)
        X_list.append(x[mask])
        W_list.append(w[mask])

    X = np.concatenate(X_list)
    weight = np.concatenate(W_list)

    # Grab tail variables
    rT, rho, mT, pT = X[:, -1], X[:, -2], X[:, -3], X[:, -4]
    X = X[:, :-4]

    return X, pT, mT, rT, weight

def make_RT_selection(mT, pT, rT, rt_sel: float | None, rt_ddt_file: str | None, smooth_sigma: float = 0.0):
    """Given the rT continious variable, create the boolean array for RT selection"""
    if rt_sel is None: # Ignore RT selection
        return np.ones_like(rT, dtype=bool)
    if rt_ddt_file is None: # Performing non-DDT selection
        return (rT > rt_sel)
    mask = np.ones_like(mT, dtype=bool)
    return calculate_varDDT(mT, pT, mask, rT, rt_sel, rt_ddt_file, smear=0.2, smooth_sigma=smooth_sigma) > 0.0


#------------------------------------------------------------------------------
# The Main Function -----------------------------------------------------------
#------------------------------------------------------------------------------

def main():
    # Parse arguments and take the results
    args = parse_arguments()

    ana_type = args.analysis_type
    bkg_files = args.bkg_files
    sig_files = args.sig_files
    model_file = args.bdt_file
    ddt_map_file = args.ddt_map_file
    rt_sel = args.rt_sel
    rt_ddt_file = args.rt_ddt_file
    smooth_sigmas = args.smooth_sigmas
    sig_bdt_cut = args.sig_bdt_cut
    var_cuts = args.var_cuts
    plots = args.plot
    verbosity = args.verbosity

    set_mpl_fontsize(18, 22, 26)

    features_common = ["pt", "mt", "rho", "rt"]
    ana_variant_dict = {
        "RT": {
            "features": ["rt"] + features_common,
            "inputs_to_primary": lambda x: x[:, 0],
            "primary_var_label": "$R_T$",
            "default_cut_vals": [np.round(x,4) for x in np.linspace(1.14, 1.22, 17)],
            "smear": 0.2,
        },
        "cut-based": {
            "features": ["ecfm2b1"] + features_common,
            "inputs_to_primary": lambda x: x[:, 0],
            "primary_var_label": "$M_2^{(1)}$",
            "default_cut_vals": [np.round(x,4) for x in np.linspace(0.07 , 0.17, 21)],
            "smear": 0.2,
        },
        "BDT-based": {
            "features": read_training_features(model_file) + features_common,
            "inputs_to_primary": lambda x:  calc_bdt_scores(x, model_file=model_file),
            "primary_var_label": "BDT",
            "default_cut_vals": [0.1, 0.2, 0.3, 0.32, 0.35, 0.37, 0.4, 0.42, 0.45, 0.47, 0.5, 0.52, 0.55, 0.57, 0.6, 0.62, 0.65, 0.67, 0.7, 0.72, 0.75, 0.77, 0.8, 0.9],
            "smear": 0.2,
        }
    }
    # Additional parsing based on analysis method
    ana_variant = ana_variant_dict[ana_type]
    var_cuts = var_cuts if var_cuts is not None else ana_variant['default_cut_vals']
    rt_sel = None if ana_type == "RT" else rt_sel # Explicitly disabling RT selection if computing RT DDT
    smear = ana_variant["smear"]

    # Extracting the arrays of interest
    X, pT, mT, rT, bkg_weight = make_ddt_inputs(expand_wildcards([bkg_files]), ana_variant["features"])
    primary_var = ana_variant["inputs_to_primary"](X)
    rt_mask = make_RT_selection(mT, pT, rT, rt_sel, rt_ddt_file) # This is an all true array for RT DDT computation

    # For on-the-fly smoothing with an RT-DDT map: the shifts that keep the working points are computed with the RT
    # selection smoothed the same way, as it will be applied
    recalibration_rt_sel = None
    if smooth_sigmas and rt_ddt_file is not None:
        recalibration_rt_sel = {s: make_RT_selection(mT, pT, rT, rt_sel, rt_ddt_file, smooth_sigma=s) for s in smooth_sigmas}

    # Only make the 2D DDT map if it doesn't exist.
    # Target efficiency is computed based on RT selection, while the DDT computation includes everything
    if not osp.exists(ddt_map_file):
        bkg_percents = []
        bkg_sum = np.sum(bkg_weight[(primary_var > 0.0) & rt_mask])
        for cut_val in var_cuts:
            bkg_percents.append((np.sum(bkg_weight[(primary_var > cut_val) & rt_mask]) / bkg_sum)*100)
        create_DDT_map_dict(mT, pT, rt_mask, primary_var, bkg_weight, bkg_percents, var_cuts, ddt_map_file,
                            smooth_sigmas=smooth_sigmas, recalibration_rt_sel=recalibration_rt_sel)
    elif smooth_sigmas:
        print("Adding the on-the-fly smoothing information to the existing DDT map (thresholds unchanged)")
        add_ddt_smoothing_info(ddt_map_file, mT, pT, rt_mask, primary_var, bkg_weight, smooth_sigmas, cut_vals=var_cuts,
                               recalibration_rt_sel=recalibration_rt_sel)
    else: print("The DDT has already been calculated, please change the name if you want to remake the ddt map")

    # Load the dictionary from the json file
    with open(ddt_map_file, 'r') as f:
        var_dict = json.load(f)

    if '2D_DDT_map' in plots:
        if verbosity > 0 : print("Making variable DDT plots")
        ana_label = ana_type
        if rt_ddt_file is not None:
            ana_label += 'withRTDDT'

        def plot_single(img_array, MT_PT_edges, PT_edges, plt_name):
            MT_PT_edges = np.array(MT_PT_edges)
            PT_edges = np.array(PT_edges)
            fig, ax = simple_fig()
            im_show_kwargs = {
                'extent': [MT_PT_edges[0], MT_PT_edges[-1], PT_edges[0], PT_edges[-1]],
                'aspect':'auto',
                'origin':'lower',
                'cmap': 'viridis',
            }
            im = ax.imshow(img_array, **im_show_kwargs)
            ax.figure.colorbar(im, label='DDT Map value')
            ax.set_xlabel('$\\rho_{\\mathrm{T}}$')
            ax.set_ylabel('$p_{\\mathrm{T}}$ [GeV]')
            save_plot(plt, plt_name)
            plt.close()

        for cut_val, arrays in var_dict.items():
            if cut_val == "metadata": continue
            var_map, MT_PT_edges, PT_edges, RT_edges = arrays
            var_map =  np.array(var_map)
            plot_single(var_map[:,:,1].T, MT_PT_edges, PT_edges, f'2D_map_{ana_label}_{cut_val}_nosmear')
            plot_single(gaussian_filter(var_map[:,:,1], smear).T, MT_PT_edges, PT_edges, f'2D_map_{ana_label}_{cut_val}')
            if ana_type != 'RT':
                plot_single(var_map[:,:,0].T, MT_PT_edges, PT_edges, f'2D_map_{ana_label}_{cut_val}_antiRT_nosmear')
                plot_single(gaussian_filter(var_map[:,:,0], smear).T, MT_PT_edges, PT_edges, f'2D_map_{ana_label}_{cut_val}_antiRT')
            plt.close()

    if 'bkg_scores_mt' in plots :
        if verbosity > 0 : print("Applying the DDT background")
        # Common label
        var_label = ana_variant["primary_var_label"]
        ana_label = ana_type
        if rt_ddt_file is not None:
            ana_label += 'withRTDDT'
        print("Computing var DDTs")
        primary_var_ddt = { cut_val: calculate_varDDT(mT, pT, rt_mask, primary_var, cut_val, ddt_map_file, smear=smear) for cut_val in tqdm.tqdm(var_cuts)}

        # Plot histograms for the DDT scores for different BDT cuts
        if verbosity > 0 : print("Making background plots")
        fig, ax = simple_fig()
        for cuts, scores in primary_var_ddt.items():
            if cuts != 0.6: alpha = 0.3
            else: alpha = 1.0
            ax.hist(scores, bins=60, range=(-1.0, 1.0), alpha=alpha, histtype='step', label=f'BDT Cut {cuts}')
        ax.set_xlabel('BKG_score_ddt')
        ax.set_ylabel('Events')
        ax.legend()
        save_plot(plt,'DDT_score')
        plt.close()

        def save_mt_fig(ax, name, save_log=False):
            # Common method for fixing files for MT, related items
            ax.set_xlabel('$m_{\\mathrm{T}}$ [GeV]')
            ax.legend()
            save_plot(plt, name)
            if save_log:
                ax.set_yscale('log') # Saving a log version of the plot
                save_plot(plt,f'log_{name}')
            plt.close()


        # Apply DDT > 0.0 for the different BDT score transformations
        fig, ax = simple_fig()
        for cuts, scores in primary_var_ddt.items():
            SR_mask = (scores > 0.0) & rt_mask
            ax.hist(mT[SR_mask], bins=47, range=(180,650), histtype='step', label=f'DDT({var_label} $>$ {cuts})', weights=bkg_weight[SR_mask])
        save_mt_fig(ax, f'bkg_events_vs_mT_{ana_label}', save_log=True)

        # Do it again normalized to unit area
        fig, ax = simple_fig()
        for cuts, scores in primary_var_ddt.items():
            SR_mask = (scores > 0.0) & rt_mask
            ax.hist(mT[SR_mask], bins=47, range=(180,650), histtype='step', label=f'DDT({var_label} $>$ {cuts})', weights=bkg_weight[SR_mask], density=True)
        ax.set_ylabel('Events')
        save_mt_fig(ax, f'norm_bkg_events_vs_mT_{ana_label}', save_log=True)

        # Apply DDT > 0.0 for the different BDT score transformations
        fig,ax = simple_fig()
        mT_bins = np.linspace(180, 650, 48)
        bin_centers = 0.5 * (mT_bins[:-1] + mT_bins[1:])
        bin_widths = np.diff(mT_bins)
        for cuts, scores in primary_var_ddt.items():
            SR_mask = (scores > 0.0) & rt_mask
            mT_before, _ = np.histogram(mT[rt_mask], bins=mT_bins, weights=bkg_weight[rt_mask])
            mT_after, _ = np.histogram(mT[SR_mask], bins=mT_bins, weights=bkg_weight[SR_mask])
            with np.errstate(divide='ignore', invalid='ignore') :
                mT_eff = mT_after / mT_before
                mT_eff[mT_after == 0] = np.nan
            ax.plot(bin_centers, mT_eff, drawstyle='steps-mid', label=f'DDT({var_label} $>$ {cuts})')
        ax.set_ylabel('Bkg efficiency')
        save_mt_fig(ax, f'bkg_eff_vs_mT_{ana_label}')

        # Apply DDT > 0.0 for the different BDT score transformations
        fig, ax = simple_fig()
        for cuts, scores in primary_var_ddt.items():
            SR_mask = (scores > 0.0) & rt_mask # DDT score > 0.0 is equivalent to BDT score about BDT cut value
            mT_before, _ = np.histogram(mT[rt_mask], bins=mT_bins, weights=bkg_weight[rt_mask])
            mT_after, _ = np.histogram(mT[SR_mask], bins=mT_bins, weights=bkg_weight[SR_mask])
            with np.errstate(divide='ignore', invalid='ignore') :
                mT_eff = mT_after / mT_before
                mT_eff[mT_after == 0] = np.nan
            mT_eff_area =  np.nansum(mT_eff * bin_widths)
            mT_norm_eff = mT_eff / mT_eff_area
            ax.plot(bin_centers, mT_norm_eff, drawstyle='steps-mid', label=f'DDT({var_label} $>$ {cuts})')
        ax.set_ylabel('norm bkg efficiency')
        save_mt_fig(ax,f'norm_bkg_eff_vs_mT_{ana_label}')

        # Plot ratio of events above and below DDT > 0 in mT bins as step histograms
        fig, ax = simple_fig()
        all_ratios = []
        for cuts, scores in primary_var_ddt.items():
            mask_above = (scores > 0) & rt_mask
            mask_below = (scores < 0) & rt_mask
            num_above, _ = np.histogram(mT[mask_above], bins=mT_bins, weights=bkg_weight[mask_above])
            num_below, _ = np.histogram(mT[mask_below], bins=mT_bins, weights=bkg_weight[mask_below])
            ratio = np.divide(num_above, num_below, out=np.zeros_like(num_above, dtype=float), where=num_below > 0)
            all_ratios.append(ratio)
            ax.step(bin_centers, ratio, where='mid', label=f'DDT({var_label} $>$ {cuts})')
        ax.set_ylabel(r'Ratio: $\mathrm{DDT} > 0 \,/\, \mathrm{DDT} < 0$')
        save_mt_fig(ax, f'bkg_ddt_ratio_vs_mT_{ana_label}')

        # Plot normalized ratio: divide each ratio curve by its average to see shape only
        fig, ax = simple_fig()
        for cuts, ratio in zip(primary_var_ddt.keys(), all_ratios):
            # Compute average, ignoring empty bins
            avg = np.nanmean(ratio[ratio > 0])  # or use np.mean with a mask
            normalized_ratio = np.divide(ratio, avg, out=np.zeros_like(ratio), where=avg > 0)
            ax.step(bin_centers, normalized_ratio, where='mid', label=f'DDT({var_label} $>$ {cuts})')
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
        ax.set_ylim(0.0, 2.0)
        ax.set_ylabel(r'$(\mathrm{DDT}>0/\mathrm{DDT}<0) \,/\, \langle\mathrm{DDT}>0/\mathrm{DDT}<0\rangle$')
        save_mt_fig(ax, f'bkg_ddt_ratio_vs_mT_normalized_{ana_label}')

        # Plot ratio of events above and below DDT > 0 compared with the control region that relaxes the RT selection
        if ana_type != 'RT':
            fig, ax = simple_fig()
            all_ratios = []
            all_uncs = []
            for cuts, scores in primary_var_ddt.items():
                mask_above = (scores > 0) & rt_mask
                mask_below = (scores < 0)
                num_above, _ = np.histogram(mT[mask_above], bins=mT_bins, weights=bkg_weight[mask_above])
                num_below, _ = np.histogram(mT[mask_below], bins=mT_bins, weights=bkg_weight[mask_below])
                ratio = np.divide(num_above, num_below, out=np.zeros_like(num_above, dtype=float), where=num_below > 0)
                unc = np.sqrt(1/num_above + 1/num_below)
                all_ratios.append(ratio)
                all_uncs.append(unc)
                ax.fill_between(bin_centers, ratio + unc*ratio, ratio - unc*ratio, step="mid", alpha=0.2)
                ax.step(bin_centers, ratio, where='mid', label=f'DDT({var_label} $>$ {cuts})')
            ax.set_ylabel(r'Ratio: $\mathrm{DDT} > 0 \,/\, \mathrm{DDT} < 0$')
            save_mt_fig(ax, f'bkg_ddt_ratio_loose_vs_mT_{ana_label}')

            fig, ax = simple_fig()
            for cuts, ratio, unc in zip(primary_var_ddt.keys(), all_ratios, all_uncs):
                # Compute average, ignoring empty bins
                avg = np.nanmean(ratio[ratio > 0])  # or use np.mean with a mask
                normalized_ratio = np.divide(ratio, avg, out=np.zeros_like(ratio), where=avg > 0)
                print(unc, normalized_ratio, bin_centers)
                ax.fill_between(bin_centers, normalized_ratio * (1+unc), normalized_ratio* (1 - unc), step='mid', alpha=0.1)
                ax.step(bin_centers, normalized_ratio, where='mid', label=f'DDT({var_label} $>$ {cuts})')
            ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
            ax.set_ylim(0.0, 2.0)
            ax.set_ylabel(r'$(\mathrm{DDT}>0/\mathrm{DDT}<0) \,/\, \langle\mathrm{DDT}>0/\mathrm{DDT}<0\rangle$')
            save_mt_fig(ax, f'bkg_ddt_ratio_loose_vs_mT_normalized_{ana_label}')

        if verbosity > 1 : print(primary_var_ddt)

    if 'sig_scores_mt' in plots :
        if verbosity > 0 : print("Applying the DDT to the signal sample")
        # Common label
        var_label = ana_variant["primary_var_label"]
        ana_label = ana_type
        if rt_ddt_file is not None:
            ana_label += 'withRTDDT'

        sig_files =  [
            f for f in expand_wildcards([sig_files])
            if f'mMed-300' in f and 'mDark-10' in f and 'rinv-0p3' in f
        ]
        sig_X, sig_pT, sig_mT, sig_rT, sig_weight = make_ddt_inputs(sig_files, ana_variant['features'])
        sig_primary_var = ana_variant["inputs_to_primary"](sig_X)
        sig_rt_mask = make_RT_selection(sig_mT, sig_pT, sig_rT, rt_sel, rt_ddt_file)
        primary_var_ddt = { cut_val: calculate_varDDT(sig_mT, sig_pT, sig_rt_mask, sig_primary_var, cut_val, ddt_map_file, smear=smear) for cut_val in tqdm.tqdm(var_cuts)}

        def save_mt_fig(ax, name, save_log=False):
            # Common method for fixing files for MT, related items
            ax.set_xlabel('$m_{\\mathrm{T}}$ [GeV]')
            ax.legend()
            save_plot(plt, name)
            if save_log:
                ax.set_yscale('log') # Saving a log version of the plot
                save_plot(plt,f'log_{name}')
            plt.close()

        # Apply DDT > 0.0 for the different BDT score transformations
        fig, ax = simple_fig()
        for cuts, scores in primary_var_ddt.items():
            SR_mask = (scores > 0.0) & sig_rt_mask
            ax.hist(sig_mT[SR_mask], bins=47, range=(180,650), histtype='step', label=f'DDT({var_label} $>$ {cuts})', weights=sig_weight[SR_mask])
        save_mt_fig(ax, f'sig_events_vs_mT_{ana_label}', save_log=True)

        # Do it again normalized to unit area
        fig, ax = simple_fig()
        for cuts, scores in primary_var_ddt.items():
            SR_mask = (scores > 0.0) & sig_rt_mask
            ax.hist(sig_mT[SR_mask], bins=47, range=(180,650), histtype='step', label=f'DDT({var_label} $>$ {cuts})', weights=sig_weight[SR_mask], density=True)
        ax.set_ylabel('Events')
        save_mt_fig(ax, f'norm_sig_events_vs_mT_{ana_label}', save_log=True)
    # _____________________________________________
    # Create Significance Plots
    if 'fom_significance' in plots :
        files_by_mass = { # Group files by mass point
            mass: [
                f for f in expand_wildcards([sig_files])
                if f'mMed-{mass}' in f and 'mDark-10' in f and 'rinv-0p3' in f
            ]
            for mass in signal_xsecs.keys()
        }
        # Prepare a figure
        ana_label = ana_type
        if rt_ddt_file is not None:
            ana_label += 'withRTDDT'

        # Storing the results of interest from the cut scanning
        best_bdt_cuts = [] #store the best cut values
        best_fom_vals = [] #store the optimal FOM for each mass point
        # Storing the FOM array and the input components of the array
        FOM_lists = []
        S_count_lists = []
        B_count_lists = []

        # Iterate over the variables in the 'con' dictionary
        for mz, mz_files in files_by_mass.items():
            s = f'bsvj_{mz:d}_10_0.3'

            # Grab the input features and weights
            sig_X, sig_pT, sig_mT, sig_rT, sig_weight = make_ddt_inputs(mz_files, ana_variant['features'])
            if verbosity > 0 : print("M(Z') = ", mz, " Events: ", len(sig_X), " weights: ", sig_weight)
            sig_primary_var = ana_variant["inputs_to_primary"](sig_X)
            bkg_score_ddt = []
            if verbosity > 0 : print("Applying the DDT and calculate FOM")

            # Iterate over the cut values
            fom, s_count, b_count = [],[],[] # to store the figure of merit values
            for cut_val in var_cuts:
                def _get_ddt_yield(mT, pT, rT, var, weight):
                    rt_mask = make_RT_selection(mT, pT, rT, rt_sel, rt_ddt_file)
                    score_ddt = calculate_varDDT(mT, pT, rt_mask, var, cut_val, ddt_map_file, smear=smear)
                    SR_mask = (score_ddt > 0) & rt_mask
                    mt_mask = (mT > (mz - 100)) & (mT < (mz + 100))
                    mt_fill = mT[SR_mask & mt_mask]
                    weight_fill = weight[SR_mask & mt_mask]
                    return sum(np.histogram(mt_fill, bins=50, weights=weight_fill)[0])

                # Calculate the figure of merit values for this bdt cut
                S = _get_ddt_yield(sig_mT, sig_pT, sig_rT, sig_primary_var, sig_weight)
                B = _get_ddt_yield(mT, pT, rT, primary_var, bkg_weight)
                F = FOM(S,B)
                if verbosity > 0 : print("mZ': ", mz, "cut:" , cut_val, " S: ", S, "B: ", B, "FOM:" , F)
                fom.append(F)
                s_count.append(S)
                b_count.append(B)

            # Find the cut value corresponding to the maximum figure of merit
            best_bdt_cuts.append([mz, var_cuts[np.argmax(fom)]])
            best_fom_vals.append([mz, np.max(fom)])
            FOM_lists.append([mz, fom])
            S_count_lists.append([mz, s_count])
            B_count_lists.append([mz, b_count])
            if verbosity > 1 : print(best_bdt_cuts)

        # Plotting the cut-value scan maps
        for container, plotName, ylabel, yscale in [
            (FOM_lists, "FOM", "FoM", 'linear'),
            (S_count_lists, "SCount", "Num. of Signal", 'log'),
            (B_count_lists, "BCount", "Num. of Background", 'log')
        ]:
            fig = plt.figure(figsize=(10, 7))
            hep.cms.label(rlabel="(13 TeV)") # full run 2
            ax = fig.gca()

            for mz, var in container:
                alpha = 1.0 if mz == 300 else 0.3
                ax.plot(var_cuts, var, marker='o', label=f"m(Z') {mz}", alpha=alpha)

            # grab labels that will go into the legend
            handles, labels = ax.get_legend_handles_labels()

            # Convert last part of labels to integers, handle errors if conversion is not possible
            masses = []
            for label in labels:
                try:
                    mass = int(label.split()[-1])  # Assuming mass is the last word in the label
                except ValueError:
                    mass = float('inf')  # If conversion fails, assign a large number
                masses.append(mass)

            # Sort legend items by mass (in descending order)
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: masses[labels.index(t[0])], reverse=True))

            ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))
            ax.ticklabel_format(style='sci', axis='x')
            ax.set_ylabel(ylabel)
            ax.set_yscale(yscale)
            ax.set_xlabel(ana_variant_dict[ana_type]["primary_var_label"])
            save_plot(plt, f'metrics_{ana_label}_{plotName}', flag_tight_layout=False, bbox_inches='tight')
            plt.close()

        # sort the best bdt cuts
        for container, plotName, ylabel in [
            (best_bdt_cuts, "cuts", "Best cut value"),
            (best_fom_vals, "fom", "Maximum FOM value")
        ]:
            best_bdt_cuts = np.array(container)
            sort_indices = np.argsort(best_bdt_cuts[:,0]) #sort array to ascend by mz
            best_bdt_cuts = best_bdt_cuts[sort_indices]

            # Find optimal cut
            mask = (best_bdt_cuts[:,0] >= 200) & (best_bdt_cuts[:,0] < 400) if ana_type == "BDT-based" else np.ones_like(best_bdt_cuts[:,0], dtype=bool)
            selected_values = best_bdt_cuts[mask, 1]
            average = np.mean(selected_values)
            optimal_bdt_cut = min(var_cuts, key=lambda x: abs(x - average)) # Finding the nearest value in ana_variant['cut_values'] to the calculated average

            # plot the best bdt cuts
            fig = plt.figure(figsize=(10, 7))
            hep.cms.label(rlabel="(13 TeV)") # full run 2
            ax = fig.gca()
            ax.plot(best_bdt_cuts[:,0], best_bdt_cuts[:,1], marker='o')
            ax.text(0.05, 0.10, f'Optimal Cut: {optimal_bdt_cut:.2f}', transform=ax.transAxes, verticalalignment='top')
            ax.ticklabel_format(style='sci', axis='x')
            ax.set_ylabel(ylabel)
            ax.set_xlabel("$m(\\mathrm{Z'})$ [GeV]")
            # cannot use layout_tight, will cause saving errors
            save_plot(plt, f'best_{ana_label}_{plotName}', flag_tight_layout=False, bbox_inches='tight')
            plt.close()

    # _____________________________________________
    # Create Significance Plots
    if 'fom_mt_scan' in plots :
        files_by_mass = { # Group files by mass point
            mass: [
                f for f in expand_wildcards([sig_files])
                if f'mMed-{mass}' in f and 'mDark-10' in f and 'rinv-0p3' in f
            ]
            for mass in [200,300,400]
        }
        # Prepare a figure
        fig = plt.figure(figsize=(10, 10))
        spec = fig.add_gridspec(ncols=1,
                          nrows=2,
                          width_ratios=[1],
                          height_ratios=[1.6, 1])
        ax_upper = fig.add_subplot(spec[0, 0])
        ax_lower = fig.add_subplot(spec[1, 0], sharex=ax_upper)
        plt.setp(ax_upper.get_xticklabels(), visible=False)

        mT_bins = np.linspace(180, 650, 48)
        ana_label = ana_type
        if rt_ddt_file is not None:
            ana_label += 'withRTDDT'

        def _get_ddt_yield(mT, pT, rT, var, weight, cut_val=None, mz=None):
            # Getting the histogram used to draw plots
            rt_mask = make_RT_selection(mT, pT, rT, rt_sel, rt_ddt_file)
            score_mask = calculate_varDDT(mT, pT, rt_mask, var, cut_val, ddt_map_file, smear=smear) > 0 if cut_val is not None else np.ones_like(mT,dtype=bool)
            SR_mask = score_mask & rt_mask
            mt_mask = (mT > (mz - 100)) & (mT < (mz + 100)) if mz is not None else np.ones_like(mT, dtype=bool)
            mt_fill = mT[SR_mask & mt_mask]
            weight_fill = weight[SR_mask & mt_mask]
            return np.histogram(mt_fill, bins=mT_bins, weights=weight_fill)[0]

        for cut_val, ls, alpha in zip([None] + var_cuts, ['solid', 'dashed', 'dashdot', 'dotted'], [1.0,0.8,0.5,0.3]):
            bkg_hist = _get_ddt_yield(mT, pT, rT, primary_var, bkg_weight, cut_val=cut_val, mz=None)
            ax_upper.step(mT_bins[:-1], bkg_hist*100, color='black', ls=ls)
            ax_lower.plot([], [], color='gray', ls=ls, label=f"Cut value: {cut_val}")
            if cut_val is None:
                ax_upper.plot([], [], color='black', ls='solid', label="Background (x100)")
            for (mz, mz_files), color in zip(files_by_mass.items(), plt.rcParams['axes.prop_cycle'].by_key()['color']):
                s = f'bsvj_{mz:d}_10_0.3'
                sig_X, sig_pT, sig_mT, sig_rT, sig_weight = make_ddt_inputs(mz_files, ana_variant['features'])
                if verbosity > 0 : print("M(Z') = ", mz, " Events: ", len(sig_X), " weights: ", sig_weight)
                sig_primary_var = ana_variant["inputs_to_primary"](sig_X)
                sig_hist = _get_ddt_yield(sig_mT, sig_pT, sig_rT, sig_primary_var, sig_weight, cut_val=cut_val, mz=mz)
                ax_upper.step(mT_bins[:-1], sig_hist, color=color, ls=ls)
                ax_lower.step(mT_bins[:-1], FOM(sig_hist,bkg_hist), color=color, ls=ls)
                if cut_val is None:
                    ax_upper.plot([],[],color=color, ls='solid', label=f"$m_Z'$ = {mz} [GeV]")

        ax_upper.set_yscale('log')
        ax_upper.set_ylabel('Number of events')
        ax_lower.set_xlabel('$m_{T}$ [GeV]')
        ax_lower.set_ylabel("FOM per-bin-value")
        ax_upper.legend()
        ax_lower.legend()
        # cannot use layout_tight, will cause saving errors
        save_plot(plt, f'FOM_mT_scan_{ana_label}')
        plt.close()

    # _____________________________________________
    # BDT only section
    if ana_type == 'BDT-based' :
        # _____________________________________________
        # Apply the DDT to different signals for one BDT cut value

        if 'sig_mt_single_BDT' in plots :
            # make plotting objects
            fig, ax = plt.subplots(figsize=(10, 8))
            # Options to make the plot fancier
            hep.cms.label(rlabel="(13 TeV)")

            # Loop over the mZ values
            for sig_file in expand_wildcards([sig_files]):
                sig_col = Columns.load(sig_file)
                mz = sig_col.metadata['mz']

                # Signal Column
                sig_X, sig_pT, sig_mT, sig_rT, sig_weight = make_ddt_inputs([sig_file], ana_variant['features'])
                if verbosity > 0 : print("M(Z') = ", mz, " Events: ", len(sig_X), " weights: ", sig_weight)

                # _____________________________________________
                # Open the trained models and get the scores
                sig_score = ana_variant["inputs_to_primary"](sig_X)

                # _____________________________________________
                # Apply the DDT  to the signal
                if verbosity > 0 : print("Applying the DDT signal")
                sig_score_ddt = calculate_varDDT(sig_mT, sig_pT, sig_score, sig_bdt_cut, ddt_map_file, smear=smear)

                # Make mT distributions
                if mz != 300: alpha = 0.3
                else: alpha = 1.0
                rt_mask = make_RT_selection(sig_mT, sig_pT, sig_rT, rt_sel, rt_ddt_file)
                SR_mask = (sig_score_ddt > 0.0) & rt_mask
                ax.hist(sig_mT[SR_mask], weights=sig_weight[SR_mask], bins=47, range=(180,650), histtype='step', label=f"m(Z')={mz}", alpha=alpha)

            ax.set_xlabel('$m_{\\mathrm{T}}$ [GeV]')
            ax.set_ylabel('Events')
            ax.legend()
            save_plot(plt,'sig_events_vs_mT')

            # log scale it
            ax.set_yscale('log')
            save_plot(plt,'log_sig_events_vs_mT')
            plt.close()

        # _____________________________________________
        # Apply the DDT to one signals for different BDT cut values

        if 'one_sig_mt_many_bdt' in plots :

            # grab signal data
            sig_cols = [Columns.load(f) for f in glob.glob(sig_files)]

            # Loop over the mZ values and only grab the mZ = 300 value
            sig_col = None
            for col in sig_cols:
                mz = col.metadata['mz']
                if mz == 300 : sig_col = col
            if sig_col == None :
                raise FileNotFoundError("The mZ' 300 file doesn't exist. Please make sure you provide it in order to make the one sigma plots.")


            # make plotting objects
            fig, ax = plt.subplots(figsize=(10, 8))
            # Options to make the plot fancier
            hep.cms.label(rlabel="(13 TeV)")

            mz = sig_col.metadata['mz']

            # Signal Column
            sig_X, sig_pT, sig_mT, sig_rT, sig_weight = make_ddt_inputs(sig_col, ana_variant['features'])
            if verbosity > 0 : print("M(Z') = ", mz, " Events: ", len(sig_X), " weights: ", sig_weight)
            sig_score = calc_bdt_scores(sig_X, model_file=model_file)

            if verbosity > 0 : print("Applying the DDT signal")
            sig_score_ddt = []
            for cut_val in ana_variant['cut_values'] :
                sig_score_ddt.append(calculate_varDDT(sig_mT, sig_pT, sig_score, cut_val, ddt_map_file,smear=smear))

            # Plot histograms for the DDT scores for different BDT cuts
            fig, ax = plt.subplots(figsize=(10, 8))
            # Options to make the plot fancier
            hep.cms.label(rlabel="(13 TeV)")
            for cuts, scores in zip(ana_variant['cut_values'], sig_score_ddt):
                if cuts != 0.7: alpha = 0.3
                else: alpha = 1.0
                ax.hist(scores, weights=sig_weight, bins=50, range=(-1.0, 1.0), alpha=alpha, histtype='step', label=f'BDT Cut {cuts}')
            ax.set_xlabel('DDT(BDT)')
            ax.set_ylabel('Events')
            ax.legend()
            save_plot(plt,'DDT_sig_score_mz300')
            plt.close()

            # Apply DDT > 0.0 for the different BDT score transformations
            fig, ax = plt.subplots(figsize=(10, 8))
            # Options to make the plot fancier
            hep.cms.label(rlabel="(13 TeV)")
            for cuts, scores in zip(ana_variant['cut_values'], sig_score_ddt):
                SR_mask = scores > 0.0 # DDT score > 0.0 is equivalent to BDT score about BDT cut value
                ax.hist(sig_mT[SR_mask], weights=sig_weight[SR_mask], bins=47, range=(180,650), histtype='step', label=f'DDT(BDT cut = {cuts})')

            ax.set_xlabel('$m_{\\mathrm{T}}$ [GeV]')
            ax.set_ylabel('Events')
            ax.legend()
            save_plot(plt,'sig_mz300_events_vs_mT')

            # log scale it
            ax.set_yscale('log')
            save_plot(plt,'log_sig_mz300_events_vs_mT')

            # Do it again normalized to unit area
            fig, ax = plt.subplots(figsize=(10, 8))
            # Options to make the plot fancier
            hep.cms.label(rlabel="(13 TeV)")
            for cuts, scores in zip(ana_variant['cut_values'], sig_score_ddt):
                SR_mask = scores > 0.0 # DDT score > 0.0 is equivalent to BDT score about BDT cut value
                ax.hist(sig_mT[SR_mask], weights=sig_weight[SR_mask], bins=47, range=(180,650), histtype='step', label=f'DDT(BDT cut = {cuts})', density=True)

            ax.set_xlabel('$m_{\\mathrm{T}}$ [GeV]')
            ax.set_ylabel('Events')
            ax.legend()
            save_plot(plt,'norm_sig_mz300_events_vs_mT')
            plt.close()

if __name__ == '__main__':
    main()
