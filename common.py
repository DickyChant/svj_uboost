import os, os.path as osp, logging, re, time, json, argparse, sys, math, shutil, subprocess
import matplotlib.pyplot as plt
from collections import OrderedDict
from contextlib import contextmanager
import svj_ntuple_processing as svj
from scipy.ndimage import gaussian_filter
import requests
import numpy as np
from datetime import datetime
import json
import tqdm
import multiprocessing as mp
import multiprocessing.pool
from cycler import cycler

np.random.seed(1001)


# Where training data will be stored
DATADIR = osp.join(osp.dirname(osp.abspath(__file__)), 'data')


def setup_logger(name: str = "bdt") -> logging.Logger:
    """Sets up a Logger instance.

    If a logger with `name` already exists, returns the existing logger.

    Args:
        name (str, optional): Name of the logger. Defaults to "demognn".

    Returns:
        logging.Logger: Logger object.
    """
    if name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.info("Logger %s is already defined", name)
    else:
        fmt = logging.Formatter(
            fmt=(
                "\033[34m[%(name)s:%(levelname)s:%(asctime)s:%(module)s:%(lineno)s]\033[0m"
                + " %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
    return logger


logger = setup_logger()


def debug(flag: bool = True) -> None:
    """Convenience switch to set the logging level to DEBUG.

    Args:
        flag (bool, optional): If true, set the logging level to DEBUG. Otherwise, set
            it to INFO. Defaults to True.
    """
    if flag:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)


cms_style = {
    "font.sans-serif": ["TeX Gyre Heros", "Helvetica", "Arial"],
    "font.family": "sans-serif",
    #
    "mathtext.fontset": "custom",
    "mathtext.rm": "helvetica",
    "mathtext.bf": "helvetica:bold",
    "mathtext.sf": "helvetica",
    "mathtext.it": "helvetica:italic",
    "mathtext.tt": "helvetica",
    "mathtext.cal": "helvetica",
    #
    "figure.figsize": (10.0, 10.0),
    "font.size": 26,
    "axes.labelsize": "medium",
    "axes.unicode_minus": False,
    "xtick.labelsize": "small",
    "ytick.labelsize": "small",
    "legend.fontsize": "small",
    "legend.handlelength": 1.5,
    "legend.borderpad": 0.5,
    "legend.frameon": False,
    "xtick.direction": "in",
    "xtick.major.size": 12,
    "xtick.minor.size": 6,
    "xtick.major.pad": 6,
    "xtick.top": True,
    "xtick.major.top": True,
    "xtick.major.bottom": True,
    "xtick.minor.top": True,
    "xtick.minor.bottom": True,
    "xtick.minor.visible": True,
    "ytick.direction": "in",
    "ytick.major.size": 12,
    "ytick.minor.size": 6.0,
    "ytick.right": True,
    "ytick.major.left": True,
    "ytick.major.right": True,
    "ytick.minor.left": True,
    "ytick.minor.right": True,
    "ytick.minor.visible": True,
    "grid.alpha": 0.8,
    "grid.linestyle": ":",
    "axes.linewidth": 2,
    "axes.prop_cycle": cycler("color", ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2" , "#832db6" , "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd" ]),
    "savefig.transparent": False,
    "xaxis.labellocation": "right",
    "yaxis.labellocation": "top",
    'text.usetex' : True,
    }

def set_mpl_fontsize(small=16, medium=22, large=26):
    """Sets matplotlib font sizes to sensible defaults.

    Args:
        small (int, optional): Font size for text, axis titles, and ticks. Defaults to
            18.
        medium (int, optional): Font size for axis labels. Defaults to 22.
        large (int, optional): Font size for figure title. Defaults to 26.
    """
    plt.rc('font', size=small)          # controls default text sizes
    plt.rc('axes', titlesize=small)     # fontsize of the axes title
    plt.rc('axes', labelsize=medium)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=small)    # fontsize of the tick labels
    plt.rc('legend', fontsize=medium)    # legend fontsize
    plt.rc('figure', titlesize=large)  # fontsize of the figure title
    from matplotlib.pyplot import style as plt_style
    plt_style.use(cms_style)
    plt.rc('text', usetex=True)
    plt.rc(
        'text.latex',
        preamble=(
            r'\usepackage{helvet} '
            r'\usepackage{sansmath} '
            r'\sansmath '
            )
        )

def put_on_cmslabel(ax, text='Simulation Preliminary', year=2018):
    fontsize = 27
    ax.text(
        .0, 1.005,
        r'\textbf{CMS}\,\fontsize{21pt}{3em}\selectfont{}{\textit{'+text+'}}',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax.transAxes,
        usetex=True,
        fontsize=fontsize
        )
    # assume float means lumi, convert to fb-1
    if isinstance(year,float):
        year = "{:.1f} ".format(year/1000)
        year = year+r"$\mathrm{fb}^{-1}$"
    ax.text(
        1.0, 1.005,
        '{} (13 TeV)'.format(year),
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax.transAxes,
        usetex=True,
        fontsize=int(19./23. * fontsize)
        )

import matplotlib.pyplot as plt
set_mpl_fontsize()



def pull_arg(*args, **kwargs) -> argparse.Namespace:
    """Reads a specific argument out of sys.argv, and then deletes that argument from
    sys.argv.

    This useful to build very adaptive command line options to scripts. It does
    sacrifice auto documentation of the command line options though.

    Returns:
        argparse.Namespace: Namespace object for only the specific argument.
    """

    """
    Reads a specific argument out of sys.argv, and then
    deletes that argument from sys.argv.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(*args, **kwargs)
    args, other_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + other_args
    return args


@contextmanager
def timeit(msg):
    """
    Prints duration of a block of code in the terminal.
    """
    try:
        logger.info(msg)
        sys.stdout.flush()
        t0 = time.time()
        yield None
    finally:
        t1 = time.time()
        logger.info(f"Done {msg[0].lower() + msg[1:]}, took {t1-t0:.2f} secs")


time_and_log = timeit # backwards compatibility


def imgcat(path) -> None:
    """
    Only useful if you're using iTerm with imgcat on the $PATH:
    Display the image in the terminal.
    """
    if shutil.which('imgcat'):
        os.system('imgcat ' + path)


def expand_wildcards(pats):
    import seutils
    import glob
    expanded = []
    for pat in pats:
        if '*' in pat:
            if seutils.path.has_protocol(pat):
                expanded.extend(seutils.ls_wildcard(pat))
            else:
                expanded.extend(glob.glob(pat))
        else:
            expanded.append(pat)
    return expanded


class Scripter:
    """
    Command line utility.

    When an instance of this class is used as a contextwrapper on a function, that
    function will be considered a 'command'.

    When Scripter.run() is called, the script name is pulled from the command line, and
    the corresponding function is executed.

    Example:

        In file test.py:
        >>> scripter = Scripter()
        >>> @scripter
        >>> def my_func():
        >>>     print('Hello world!')
        >>> scripter.run()

        On the command line, the following would print 'Hello world!':
        $ python test.py my_func
    """

    def __init__(self):
        self.scripts = {}

    def __call__(self, fn):
        """
        Stores a command line script with its name as the key.
        """
        self.scripts[fn.__name__] = fn
        return fn

    def run(self):
        script = pull_arg("script", choices=list(self.scripts.keys())).script
        logger.info(f"Running {script}")
        self.scripts[script]()


@contextmanager
def quick_ax(figsize=(10, 10), outfile="tmp.png"):
    """
    Context manager to open a matplotlib Axes.
    Upon closing, saves it to a file and calls
    imgcat (an iTerm2 command line util) on it
    to display the plot in the terminal.
    """
    try:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        yield ax
    finally:
        plt.savefig(outfile, bbox_inches="tight")
        try:
            os.system(f"imgcat {outfile}")
        except Exception:
            pass


@contextmanager
def quick_fig(figsize=(10, 10), outfile="tmp.png"):
    """
    Context manager to open a matplotlib Figure.
    Upon closing, saves it to a file and calls
    imgcat (an iTerm2 command line util) on it
    to display the plot in the terminal.
    """
    try:
        fig = plt.figure(figsize=figsize)
        yield fig
    finally:
        plt.savefig(outfile, bbox_inches="tight")
        try:
            os.system(f"imgcat {outfile}")
        except Exception:
            pass

@contextmanager
def quick_subplots(*args, **kwargs):
    """
    Context manager to open a matplotlib Figure.
    Upon closing, saves it to a file and calls
    imgcat (an iTerm2 command line util) on it
    to display the plot in the terminal.
    """
    outfile = kwargs.pop('outfile', 'tmp.png')
    try:
        fig, axes = plt.subplots(*args, **kwargs)
        yield fig, axes
    finally:
        plt.savefig(outfile, bbox_inches="tight")
        try:
            os.system(f"imgcat {outfile}")
        except Exception:
            pass

def _mp_count_compute(factor=None, max_thread=None):
    # Simple switch for how many workers we should open in a pool
    if factor is None:
        factor = 0.75 if mp.cpu_count() < 16 else 0.5
    if max_thread is None:
        max_thread = mp.cpu_count()
    return int(min([mp.cpu_count() * factor , max_thread]))

@contextmanager
def mp_pool(factor=None, max_thread=None):
    try:
        n_threads = _mp_count_compute(factor, max_thread)
        logger.info(f"Creating mp.Pool with {n_threads} workers")
        yield mp.Pool(n_threads)
    finally:
        pass


@contextmanager
def mp_thread_pool(factor=None, max_thread=None):
    # Threading is required for if the function call uses variables that are
    # defined outside the function scope
    try:
        n_threads = _mp_count_compute(factor, max_thread)
        print(f"Creating thread pool with {n_threads} threads")
        yield mp.pool.ThreadPool(n_threads)
    finally:
        pass

#__________________________________________________
# Automatic cross section getter

class Record(dict):
    @property
    def xs(self):
        return self['crosssection']['xs_13tev']

    @property
    def br(self):
        try:
            return self['branchingratio']['br_13tev']
        except KeyError:
            return 1.

    @property
    def kfactor(self):
        if 'kfactor' in self:
            for key, val in self['kfactor'].items():
                if key.startswith('kfactor_'):
                    return val
        return 1.

    @property
    def effxs(self):
        return self.xs*self.br*self.kfactor


def load_treemaker_crosssection_txt():
    """
    Downloads the cross section file from the TreeMaker repository and returns
    the contents. If the file has been previously downloaded it is not re-downloaded.
    """
    import requests
    cache = '/tmp/treemaker_xs.txt'
    if not osp.isfile('/tmp/treemaker_xs.txt'):
        url = 'https://raw.githubusercontent.com/TreeMaker/TreeMaker/Run2_UL/WeightProducer/python/MCSampleValues.py'
        text = requests.get(url).text
        with open(cache, 'w') as f:
            text = text.lower()
            f.write(text)
            return text
    else:
        with open(cache) as f:
            return f.read()


def get_record(key):
    """
    Looks for the sample key (e.g. "QCD_Pt_1400to1800") in the cross section
    file from TreeMaker
    """
    text = load_treemaker_crosssection_txt()
    match = re.search('"'+key+'"' + r' : ({[\w\W]*?})', text, re.MULTILINE)
    if not match: raise Exception(f'Could not find record for {key}')
    # Turn it into a dict of dicts
    record_txt = (
        match.group(1)
        .replace('xsvalues', 'dict')
        .replace('brvalues', 'dict')
        .replace('kfactorvalues', 'dict')
        )
    return Record(eval(record_txt))


def mt_wind(cols, mt_high, mt_low):
    mt_cut = (cols.arrays['mt']>mt_low) & (cols.arrays['mt']<mt_high)
    return mt_cut

def filter_pt(cols, min_pt):
    """
    Filters for a minimum pt (only valid for QCD).
    Does not filter out non-QCD backgrounds or signals.
    """
    filtered = []
    for c in cols:
        if c.metadata.get('ptbin', [1e6])[0] < min_pt:
            continue
        filtered.append(c)
    return filtered

def filter_ht(cols, min_ht, bkg_type=None):
    """
    Filters for a minimum pt (only valid for ttjets/wjets/zjets).
    Does not filter out QCD or signal.
    If bkg_type is None, it filters ttjets AND wjets AND zjets.
    """
    filtered = []
    for c in cols:
        if bkg_type and c.metadata.get('bkg_type', None) != bkg_type:
            filtered.append(c)
            continue
        if c.metadata.get('htbin',[1e6, 1e6])[0] < min_ht:
            continue
        filtered.append(c)
    return filtered


#__________________________________________________
# Histogram classes

# in units of pb-1 (xsec units: pb)
lumis = {
    "2016": 36310,
    "2017": 42070,
    "2018PRE": 20973,
    "2018POST": 38588,
}
lumis["2018"] = lumis["2018PRE"]+lumis["2018POST"] # 59561
lumis["RUN2"] = lumis["2016"]+lumis["2017"]+lumis["2018"] # 137941

# from madgraph (BR to dark included), MADPT>300, jet matching efficiency included, for gq = 0.25
# w/ Z-like k-factor 1.23
signal_xsecs = {
    200 : 9.143,
    250 : 6.910,
    300 : 5.279,
    350 : 4.077,
    400 : 3.073,
    450 : 2.448,
    500 : 1.924,
    550 : 1.578,
}

def get_event_weight(obj,lumi=None, noPU=False):
    if isinstance(obj,svj.Columns):
        if lumi is None:
            lumi = lumis[str(obj.metadata['year'])]
        if obj.metadata['sample_type'] == 'data': # Early exit before accessing weight
            return np.ones_like(obj.arrays['evt'], dtype=np.float32)
        tree_weights = obj.to_numpy(['weight']).ravel()
        puweights = obj.to_numpy(["puweight"]).ravel()
        puweights = np.ones_like(puweights) if noPU else puweights
        if obj.metadata["sample_type"]=="sig":
            mz = obj.metadata["mz"]
            if mz in signal_xsecs:
                xsec = signal_xsecs[mz]
            else:
                # uses interpolation
                xsec = central.xs
            nevents = obj.cutflow['raw']
            event_weight = lumi*xsec/nevents
            logger.info(f'Event weight: {lumi}*{xsec}/{nevents} = {event_weight}')
            return event_weight * puweights
        elif obj.metadata["sample_type"]=="bkg":
            if len(tree_weights)>0: logger.info(f'Event weight: {lumi}*{tree_weights[0]}*{puweights[0]} = {lumi*tree_weights[0]*puweights[0]}')
            return lumi*tree_weights*puweights

    elif isinstance(obj,Histogram):
        return obj.metadata.get('event_weight',1)

    else:
        raise RuntimeError(f'Unknown weight method for object of class {type(obj).__name__}')


SF_PATH = 'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/cutbased_ddt/'
SF_FILE = 'models/jetvar_model_uncertainty.json'

def get_single_event_weight(weights):
    if isinstance(weights,float) or isinstance(weights,int): return weights
    elif len(weights)>0: return weights[0]
    else: return 1.0

def add_cutflows(*objs):
    # add cutflows if present, accounting for weights
    keys = [list(obj.cutflow.keys()) if hasattr(obj,'cutflow') else [] for obj in objs]
    if keys and all(keys) and all(k == keys[0] for k in keys):
        weights = [get_single_event_weight(get_event_weight(obj, noPU=True)) for obj in objs]
        return OrderedDict((k, sum(obj.cutflow[k]*weight for obj,weight in zip(objs,weights))) for k in keys[0])
    else:
        if keys and all(keys):
            keylist = '\n'.join(f'  {k}' for k in keys)
            logger.warning(f'Unable to add cutflows with different keys:\n{keylist}')
        return OrderedDict()

class Histogram:
    """
    Histogram container class.

    Keeps track of binning, values, errors, and metadata.
    Designed to be easily JSON-serializable.
    """
    @classmethod
    def from_dict(cls, dict):
        inst = cls.__new__(cls)
        inst.binning = np.array(dict['binning'])
        inst.vals = np.array(dict['vals'])
        inst.errs = np.array(dict['errs'])
        inst.metadata = dict['metadata'].copy()
        if 'cutflow_keys' and 'cutflow_vals' in dict:
            inst.cutflow = OrderedDict((k,v) for k,v in zip(dict['cutflow_keys'],dict['cutflow_vals']))
        return inst

    def __init__(self, binning, vals=None, errs=None):
        self.binning = binning
        self.vals = np.zeros(self.nbins) if vals is None else vals
        self.errs = np.sqrt(self.vals) if errs is None else errs
        self.metadata = {}
        self.cutflow = OrderedDict()

    @property
    def nbins(self):
        return len(self.binning)-1

    def json(self):
        # Convert anything that remotely looks like a float to python float.
        for k, v in self.metadata.items():
            try:
                self.metadata[k] = float(v)
            except (ValueError,TypeError) as e:
                pass
        return dict(
            type = 'Histogram',
            binning = list(self.binning),
            vals = list(self.vals),
            errs = list(self.errs),
            metadata = self.metadata.copy(),
            cutflow_keys = list(self.cutflow.keys()),
            cutflow_vals = [v.item() if type(v).__module__=='numpy' else v for v in self.cutflow.values()], # convert from np int64 to serializable type
            )

    def __repr__(self):
        d = np.column_stack((self.vals, self.errs))
        return (
            f'<H n={self.nbins} int={self.vals.sum():.3f}'
            f' binning={self.binning[0]:.1f}-{self.binning[-1]:.1f}'
            f' vals/errs=\n{d}'
            '>'
            )

    def copy(self):
        the_copy = Histogram(self.binning.copy(), self.vals.copy(), self.errs.copy())
        the_copy.metadata = self.metadata.copy()
        the_copy.cutflow = self.cutflow.copy()
        return the_copy

    def __add__(self, other):
        """Add another Histogram or a numpy array to this histogram. Returns new object."""
        ans = self.copy()

        if isinstance(other, Histogram):
            ans.vals = self.vals + other.vals
            ans.errs = np.sqrt(self.errs**2 + other.errs**2)
            ans.cutflow = add_cutflows(self, other)

        elif hasattr(other, 'shape') and self.vals.shape == other.shape:
            # Add a simple np histogram on top of it
            ans.vals += other
            ans.errs = np.sqrt(self.errs**2 + other)

        return ans

    def __radd__(self, other):
        if other == 0:
            return self.copy()
        raise NotImplementedError

    def __mul__(self, factor):
        """Multiply by a constant"""
        ans = self.copy()
        if isinstance(factor, (int, float)):
            ans.vals = factor*ans.vals
            ans.errs = factor*ans.errs
        else:
            raise NotImplemented
        return ans

    @property
    def norm(self):
        return self.vals.sum()

    def rebin(self, n=2):
        """
        Merge n bins together to make a coarser histogram.
        Mostly useful for plotting.
        """
        if n==1: return self.copy()
        n_bins_new = math.ceil(self.nbins / float(n))

        binning_new = self.binning[::n]
        if binning_new[-1] != self.binning[-1]:
            binning_new = np.append(binning_new, self.binning[-1])

        # Build a map from old binning to new binning
        map = np.repeat(np.arange(n_bins_new), n)[:self.nbins]

        values_new = np.zeros(n_bins_new)
        np.add.at(values_new, map, self.vals)

        errs_new = np.zeros(n_bins_new)
        np.add.at(errs_new, map, self.errs**2)
        errs_new = np.sqrt(errs_new)

        h = Histogram(binning_new, values_new, errs_new)
        h.metadata = self.metadata.copy()
        return h

    def cut(self, xmin=-np.inf, xmax=np.inf):
        """
        Throws away all bins with left boundary < xmin or right boundary > xmax.
        Mostly useful for plotting purposes.
        Returns a copy.
        """
        # safety checks
        if xmin>xmax:
            raise ValueError("xmin ({}) greater than xmax ({})".format(xmin,xmax))

        h = self.copy()
        imin = np.argmin(self.binning < xmin) if xmin > self.binning[0] else 0
        imax = np.argmax(self.binning > xmax) if xmax < self.binning[-1] else self.nbins+1
        h.binning = h.binning[imin:imax]
        h.vals = h.vals[imin:imax-1]
        h.errs = h.errs[imin:imax-1]
        return h

    def mask(self, keep):
        ans = self.copy()
        ans.vals[~keep] = 0
        ans.errs[~keep] = 0
        return ans

def mask_isolated_bins(counts):
    mask_bin = []
    for i in range(len(counts)):
        if (i<=0 or counts[i-1]==0) and (i>=len(counts)-1 or counts[i+1]==0):
            mask_bin.append(False)
        else:
            mask_bin.append(True)
    return np.array(mask_bin)

def compute_bkg_isolatedevt_mask(mT):
    """
    Given the mT array of background events from a single background sample, return a True/False array of
    whether to keep the event. If the events appears in an isloated bin in the
    standard mT binning scheme. The event is rejected
    """
    # Creating bins from the main histogram of interest, we don't really care about the actual
    # range as long as it ls larger than the defined region-of-interest of the fit
    mt_binw, mt_min, mt_max = MTHistogram.default_binning
    mt_min = np.around(mt_min / 2, mt_binw)
    mt_max = np.around(mt_max * 2, mt_binw)
    mt_edges = np.arange(mt_min, mt_max, mt_binw)
    bin_idx = np.digitize(mT, mt_edges) # Getting which bin the item should be in
    bin_idx[bin_idx==0] = 1 # put underflow into first bin
    bin_idx[bin_idx==len(mt_edges)] = len(mt_edges)-1 # put overflow into last bin
    bin_idx = bin_idx - 1 # shift to be consistent with histogram indexing
    bin_count, _ = np.histogram(mT, bins=mt_edges) # Getting the number of entries in each bin
    mask_bin = mask_isolated_bins(bin_count)
    return mask_bin[bin_idx] # Extracting to per-event masking via array index

def _create_binning(binw, left, right):
    bins = left + binw * np.arange(math.ceil((right-left)/binw)+1)
    # Force casting to python floats, as numpy values causes issues with JSON serialization
    return [float(x) for x in bins]

class HistoBins(type):
    @property
    def bins(cls):
        if cls._bins is None:
            cls._bins = _create_binning(*cls.default_binning)
        return cls._bins
    @bins.setter
    def bins(cls, val):
        cls._bins = val
    @property
    def non_standard_binning(cls):
        return cls._non_standard_binning
    @non_standard_binning.setter
    def non_standard_binning(cls, val):
        cls._non_standard_binning = val

class VarArrHistogram(Histogram, metaclass=HistoBins):
    """
    Wrapper around histogram that initializes the value and weight arrays.
    It will also carry around the current and default binning information
    """
    default_binning = (10,0,100) # Width, min, max
    _bins = None # List of bin values
    _non_standard_binning = False
    name = '' # Name of variable to use

    @property
    def bins(self):
        return self.__class__.bins

    @bins.setter
    def bins(self, val):
        self.__class__.bins = val

    @property
    def non_standard_binning(self):
        return self.__class__.non_standard_binning

    @non_standard_binning.setter
    def non_standard_binning(self, val):
        self.__class__.non_standard_binning = val

    @classmethod
    def default_binw(cls)->float:
        return cls.default_binning[0]

    @classmethod
    def default_binmin(cls)->float:
        return cls.default_binning[1]

    @classmethod
    def default_binmax(cls)->float:
        return cls.default_binning[2]

    @classmethod
    def create_binning(cls, binw, left, right):
        return _create_binning(binw, left, right)

    @classmethod
    def empty(cls):
        return Histogram(cls.bins)

    def __init__(self, cols, weights=None):
        var_arr = self._create_var_array(cols)
        vals = np.histogram(var_arr, self.bins, weights=weights)[0].astype(float)
        weights2 = weights if weights is None else weights **2
        errs = np.sqrt(np.histogram(var_arr, self.bins, weights=weights2)[0].astype(float))
        super().__init__(self.bins, vals, errs)

    @classmethod
    def _create_var_array(cls,cols):
        """Method for creating the values array used to fill the histogram"""
        return cols.to_numpy([cls.name]).flatten()



# List of variables with defined binning
class MTHistogram(VarArrHistogram):
    name='mt'
    default_binning = (10, 180, 650)

class ECFN2B2Histogram(VarArrHistogram):
    name = 'ecfn2b2'
    default_binning = (0.01, 0, 0.5)

class ECFN2B2DarkHistogram(ECFN2B2Histogram):
    name = 'ecfn2b2dark'

class ECFN2B2GenHistogram(ECFN2B2Histogram):
    name = 'ecfn2b2gen'

class ECFM2B1Histogram(VarArrHistogram):
    name = 'ecfm2b1'
    default_binning = (0.005,0,0.2)

class RTHistogram(VarArrHistogram):
    name = 'rt'
    default_binning = (0.03, 1.0, 2.5)

class METDPhiHistogram(VarArrHistogram):
    name = 'metdphi'
    default_binning = (0.08, 0, 3.2)

def find_subclasses(cls):
    subclasses = []
    to_check = [cls]
    while to_check:
        current = to_check.pop()
        for subcl in current.__subclasses__():
            if subcl not in subclasses:
                subclasses.append(subcl)
                to_check.append(subcl)
    return list(subclasses)

registered_varhists = { subcl.name: subcl for subcl in find_subclasses(VarArrHistogram) }



class Encoder(json.JSONEncoder):
    """
    Standard JSON encoder, but support for the Histogram class
    """
    def default(self, obj):
        if isinstance(obj, Histogram):
            return obj.json()
        return super().default(obj)


class Decoder(json.JSONDecoder):
    """
    Standard JSON decoder, but support for the Histogram class
    """
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, d):
        try:
            is_histogram = d['type'] == 'Histogram'
        except (AttributeError, KeyError):
            is_histogram = False
        if is_histogram:
            return Histogram.from_dict(d)
        return d

#__________________________________________________
# Data pipeline

class Columns(svj.Columns):
    """
    Data structure that contains all the training data (features)
    and information about the sample.

    See: https://github.com/boostedsvj/svj_ntuple_processing/blob/main/svj_ntuple_processing/__init__.py#L357
    """
    @classmethod
    def load(cls, *args, **kwargs):
        inst = super().load(*args, **kwargs)
        # Transforming bytes keys to str keys
        old_cf = inst.cutflow
        inst.cutflow = OrderedDict()
        for key in old_cf.keys():
            if isinstance(key, bytes):
                inst.cutflow[key.decode()] = old_cf[key]
            else:
                inst.cutflow[key] = old_cf[key]
        return inst

    def __repr__(self):
        return (
            '<Column '
            + ' '.join([f'{k}={v}' for k, v in self.metadata.items() if k!='src'])
            + '>'
            )

    @property
    def key(self):
        return (
            osp.basename(self.metadata['src'])
            .replace('.npz', '')
            ).split('_TuneCP5_13TeV')[0].lower()

    @property
    def record(self):
        if not hasattr(self, '_record'):
            self._record = get_record(self.key)
        return self._record

    @property
    def presel_eff(self):
        if self.cutflow['raw']==0: return 0.
        return self.cutflow['preselection'] / self.cutflow['raw']

    @property
    def xs(self):
        if hasattr(self, 'manual_xs'):
            # Only for setting a manual cross section
            return self.manual_xs
        elif 'bkg_type' in self.metadata:
            return self.record.effxs
        else:
            if not hasattr(self, '_signal_xs_fit'):
                self._signal_xs_fit = np.poly1d(
                    requests
                    .get('https://raw.githubusercontent.com/boostedsvj/svj_madpt_crosssection/main/fit_madpt300.txt')
                    .json()
                    )
            return self._signal_xs_fit(self.metadata['mz'])

    @property
    def effxs(self):
        return self.xs * self.presel_eff

    @property
    def weight_per_event(self):
        return self.effxs / len(self)


def columns_to_numpy(
    signal_cols, bkg_cols, features,
    downsample=.4, weight_key='weight',
    mt_high=650, mt_low=180
    ):
    """
    Takes a list of signal and background Column instances, and outputs
    a numpy array with `features` as the columns.
    """
    X = []
    y = []
    bkg_weight = []
    signal_weight = []

    logger.info(f'Downsampling bkg, keeping fraction of {downsample}')
    # Get the features for the bkg samples
    for cols in bkg_cols:
        mtwind = mt_wind(cols, mt_high, mt_low)
        this_X = cols.to_numpy(features)[mtwind]
        this_weight = cols.arrays[weight_key][mtwind]
        if downsample < 1.:
            #select = np.random.choice(len(cols), int(downsample*len(cols)), replace=False)
            select = np.random.choice(len(this_weight), int(downsample*len(this_weight)), replace=False)
            this_X = this_X[select]
            this_weight = this_weight[select]
        X.append(this_X)
        bkg_weight.append(this_weight)
        y.append(np.zeros(len(this_X)))

    # Get the features for the signal samples
    for cols in signal_cols:
        sigmtwind = mt_wind(cols, mt_high, mt_low)
        X.append(cols.to_numpy(features)[sigmtwind])
        #print(features)
        len_sig_cols=len(cols.arrays[features[0]][sigmtwind])
        #print(cols.to_numpy(features)[sigmtwind])
        #print(len(cols.to_numpy(features)[sigmtwind]))
        #length_of_signalCol=len(cols.arrays(features)[mtwind])
        #print(length_of_signalCol, len(cols))
        y.append(np.ones(len_sig_cols))
        # All signal model parameter variations should get equal weight,
        # but some signal samples have more events.
        # Use 1/n_events as a weight per event.
        signal_weight.append((1./len_sig_cols)*np.ones(len_sig_cols))

    bkg_weight = np.concatenate(bkg_weight)
    signal_weight = np.concatenate(signal_weight)
    # Set total signal weight equal to total bkg weight
    signal_weight *= np.sum(bkg_weight) / np.sum(signal_weight)
    weight = np.concatenate((bkg_weight, signal_weight))

    X = np.concatenate(X)
    y = np.concatenate(y)
    return X, y, weight


def add_key_value_to_json(json_file, key, value):
    with open(json_file, 'r') as f:
        json_str = f.read()
    json_str = json_str.rstrip()
    json_str = json_str[:-1] # Strip off the last }
    json_str += f',"{key}":{json.dumps(value)}}}'
    with open(json_file, 'w') as f:
        f.write(json_str)
    logger.info(f'Added "{key}":{json.dumps(value)} to {json_file}')


def add_manual_weight_column(signal_cols, bkg_cols):
    """
    Adds the manual weight calculation as a column
    """
    total_bkg_weight = 0
    for c in bkg_cols:
        c.arrays['manualweight'] = np.ones(len(c)) * c.weight_per_event
        total_bkg_weight += c.arrays['manualweight'].sum()

    # Set signal weights scaled relatively to one another (but not yet w.r.t. bkg)
    total_signal_weight = 0
    for c in signal_cols:
        c.arrays['manualweight'] = np.ones(len(c)) / len(c)
        total_signal_weight += c.arrays['manualweight'].sum()

    # Scale signal weights correctly w.r.t. bkg
    for c in signal_cols:
        c.arrays['manualweight'] *= total_bkg_weight / total_signal_weight


def read_training_features(model_file):
    """
    Reads the features used to train a model from a .json file.
    Only works for xgboost-trained models.
    """
    with open(model_file, 'rb') as f:
        model = json.load(f)
        return model['features']

def ddt_windowcuts(mt, pt):
    '''
    Basically a tool to constantly check the kinematics during the DDT processes
    '''
    cuts = (mt>100) & (mt<1000) & (pt>110) & (pt<1500)
    return cuts

def weighted_percentile(x, w, percent):
    """
    Returning the percentile while properly handling event weights.
    """
    # Sorting the input entries according to the values
    sorted_x = x[np.argsort(x)]
    sorted_w = w[np.argsort(x)]
    # Generating the cumulative sum
    sorted_c = np.cumsum(sorted_w)
    # Scaling the cumulative sum so that the actual data points are centered at half the weight
    sorted_c = ((sorted_c - (sorted_w / 2)) / sorted_c[-1]) * 100

    # Special case handling, extreme percent values
    if percent > sorted_c[-1]:
        return sorted_x[-1]
    if percent < sorted_c[0]:
        return sorted_x[0]

    # Getting the points where the cumulative sum pass above or below the percentile of interest
    lower_val = sorted_x[sorted_c <= percent][-1]
    upper_val = sorted_x[sorted_c >= percent][0]

    lower_sum = sorted_c[sorted_c <= percent][-1]
    upper_sum = sorted_c[sorted_c >= percent][0]

    # Getting the interpolation weight if the percentile does not land on a single number
    int_w = 0.5 if upper_sum == lower_sum else (percent - lower_sum)/(upper_sum - lower_sum)

    return upper_val *int_w + lower_val * (1-int_w)

def varmap(mt, pt, rt_sel, var, weight, percent, cut_val):
    '''
    3D map for DDT, defined in (mt/pt, pt, rt) space. The rt array in this case should be a boolean array indicating
    if the array passed or failed the RT selection.
    '''

    # Apply the rho-ddt window cuts to the data (still useful for pt/mt range)
    cuts = ddt_windowcuts(mt, pt)
    mt_pt = mt/pt
    NBINS = 73

    # Create a 2D histogram of mt and pt, weighted by event weights
    C, (MT_PT_edges, PT_edges, RT_edges) = np.histogramdd(
        np.array([mt_pt[cuts], pt[cuts], rt_sel[cuts]]).transpose(),
        bins=(NBINS, NBINS, [0,1,2]),
        weights=weight[cuts]
    )

    # Initialize the variable map
    VAR_map = np.zeros_like(C)

    # Get data arrays for events passing the cuts
    VAR = var[cuts]
    WEIGHT = weight[cuts]
    mt_pt_bin_idx = np.digitize(mt_pt[cuts], MT_PT_edges) - 1
    pt_bin_idx = np.digitize(pt[cuts], PT_edges) - 1
    rt_bin_idx = np.digitize(rt_sel[cuts], RT_edges) -1

    def _calc_bin(bin_idx):
        i,j,k = bin_idx
        BIN_CUT = (mt_pt_bin_idx == i) & (pt_bin_idx == j) & (rt_bin_idx == k)
        if np.sum(BIN_CUT) > 0:
            return weighted_percentile(VAR[BIN_CUT], WEIGHT[BIN_CUT], 100 - percent)
        return 0

    with mp_thread_pool() as p:
        bin_list = [(i,j,k) for i in range(len(MT_PT_edges)-1) for j in range(len(PT_edges)-1) for k in [0,1]]
        result_list = list(tqdm.tqdm(p.imap(_calc_bin, bin_list), desc="DDT bin", total=len(bin_list)))
        for (i,j,k), result in zip(bin_list, result_list):
            VAR_map[i][j][k] = result

    return VAR_map, MT_PT_edges, PT_edges, RT_edges


# Class that converts numpy arrays into list so they can be easily stored in json files
class NumpyArrayEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyArrayEncoder, self).default(obj)

def ddt_bins(mt, pt, rt_sel, MT_PT_edges, PT_edges, RT_edges):
    '''
    Map bin of each event as looked up when the map is applied. The ddt_window cut is no longer applied there, so
    overflow entries in mt/pt and pt are folded into the edge bins.
    '''
    pt_bin = np.clip(np.digitize(pt, PT_edges) - 1, 0, len(PT_edges) - 2)
    mt_pt_bin = np.clip(np.digitize(mt/pt, MT_PT_edges) - 1, 0, len(MT_PT_edges) - 2)
    rt_bin = np.digitize(rt_sel, RT_edges)-1
    return mt_pt_bin, pt_bin, rt_bin

def ddt_neff(mt, pt, rt_sel, weight, MT_PT_edges, PT_edges, RT_edges):
    '''
    Effective number of MC events, N_eff = (sum w)^2 / (sum w^2), in each bin of a DDT map, with the bin assignment of
    varmap (events inside the DDT window; events on the upper edges are not assigned).
    '''
    cuts = ddt_windowcuts(mt, pt)
    shape = (len(MT_PT_edges) - 1, len(PT_edges) - 1, len(RT_edges) - 1)
    mt_pt_bin = np.digitize(mt[cuts]/pt[cuts], MT_PT_edges) - 1
    pt_bin = np.digitize(pt[cuts], PT_edges) - 1
    rt_bin = np.digitize(rt_sel[cuts], RT_edges) - 1
    in_map = (mt_pt_bin >= 0) & (mt_pt_bin < shape[0]) & (pt_bin >= 0) & (pt_bin < shape[1]) & (rt_bin >= 0) & (rt_bin < shape[2])
    flat_idx = np.ravel_multi_index((mt_pt_bin[in_map], pt_bin[in_map], rt_bin[in_map]), shape)
    w = weight[cuts][in_map].astype(np.float64)
    sumw = np.bincount(flat_idx, weights=w, minlength=int(np.prod(shape))).reshape(shape)
    sumw2 = np.bincount(flat_idx, weights=w*w, minlength=int(np.prod(shape))).reshape(shape)
    return np.where(sumw2 > 0, sumw**2 / np.maximum(sumw2, 1e-300), 0.0)

def regularize_ddt_map(var_map, neff, sigma):
    '''
    Smooths the DDT thresholds in (mt/pt, pt) with a Gaussian of width sigma (in bins), separately for each RT slice.
    Each bin enters with its effective number of MC events (ddt_neff) as weight (normalised convolution), so sparsely
    populated bins are pulled towards their well populated neighbours rather than the other way around. Bins without MC
    events (stored as 0) do not contribute and do not dilute their neighbours, unlike in gaussian_filter(var_map), where
    they enter as thresholds of 0; within reach of the kernel they receive the weighted average of their populated
    neighbours.
    '''
    smoothed = np.zeros(var_map.shape)
    for k in range(var_map.shape[2]):
        num = gaussian_filter(var_map[:, :, k] * neff[:, :, k], sigma)
        den = gaussian_filter(neff[:, :, k], sigma)
        smoothed[:, :, k] = np.where(den > 1e-12, num / np.maximum(den, 1e-300), 0.0)
    return smoothed

def smooth_ddt_map(var_map, neff, sigma, shifts):
    '''
    The map as applied with on-the-fly smoothing: regularize_ddt_map, then the stored shift of each RT slice on its
    filled bins (ddt_smoothing_shifts).
    '''
    smoothed = regularize_ddt_map(var_map, neff, sigma)
    for k, shift in enumerate(shifts):
        filled = smoothed[:, :, k] != 0
        smoothed[:, :, k][filled] += shift
    return smoothed

def ddt_smoothing_shifts(smoothed_map, mt, pt, rt_sel, var, weight, edges, targets):
    '''
    Per RT slice, the common offset that brings the weighted fraction of events inside the DDT window that pass a
    smoothed map (var above the smoothed threshold of their bin) back to the target fraction of that slice. Smoothing
    changes the overall efficiency where the threshold surface is curved; the offset keeps the working point.

    Inputs:
    - smoothed_map: regularize_ddt_map of the stored map
    - mt, pt, rt_sel, var, weight: the events the map was built from; rt_sel = RT slice of each event as it will be
      applied (with an RT-DDT map that is itself smoothed on the fly, the RT selection smoothed the same way)
    - edges: (MT_PT_edges, PT_edges, RT_edges) of the map
    - targets: target pass fraction (0-1) of each RT slice
    '''
    cuts = ddt_windowcuts(mt, pt)
    mt_pt_bin, pt_bin, rt_bin = ddt_bins(mt[cuts], pt[cuts], rt_sel[cuts], *edges)
    VAR = var[cuts]
    WEIGHT = weight[cuts]
    shifts = []
    for k in range(smoothed_map.shape[2]):
        in_slice = rt_bin == k
        if not np.any(in_slice):
            shifts.append(0.0)
            continue
        x = VAR[in_slice].astype(np.float64) - smoothed_map[mt_pt_bin[in_slice], pt_bin[in_slice], k]
        w = WEIGHT[in_slice].astype(np.float64)
        order = np.argsort(x)
        cumw = np.cumsum(w[order]) / np.sum(w)
        shifts.append(float(x[order][min(np.searchsorted(cumw, 1 - targets[k]), len(x) - 1)]))
    return shifts

def _smoothing_shifts_for_map(var_map, edges, mt, pt, rt_sel, var, weight, neff, smooth_sigmas, targets, recalibration_rt_sel):
    '''{str(float(sigma)): per-slice shifts} of one map for each sigma'''
    shifts = {}
    for sigma in smooth_sigmas:
        slices = rt_sel if recalibration_rt_sel is None else recalibration_rt_sel[sigma]
        shifts[str(float(sigma))] = ddt_smoothing_shifts(regularize_ddt_map(var_map, neff, sigma), mt, pt, slices, var, weight, edges, targets)
    return shifts

def add_ddt_smoothing_info(ddt_name, mt, pt, rt_sel, var, weight, smooth_sigmas, cut_vals=None, recalibration_rt_sel=None, out_name=None):
    '''
    Adds the information for on-the-fly smoothing (N_eff per bin, one shift per RT slice for each width) to the
    metadata of an existing DDT map file, for instance one made before this option existed. The stored thresholds are
    not changed.

    Inputs:
    - mt, pt, rt_sel, var, weight: the events the map was built from, with the same selection, weights and RT slices
    - smooth_sigmas: smoothing widths (bins) to prepare
    - cut_vals: working points to prepare (default: all in the file)
    - recalibration_rt_sel: {sigma: RT slice of each event} if the RT selection applied with this map is smoothed on the
      fly too (default: rt_sel)
    - out_name: output file (default: ddt_name, updated in place)
    The target efficiency of each working point is taken from the metadata if present, otherwise it is the efficiency of
    the stored map on these events, per RT slice.
    '''
    with open(ddt_name, 'r') as f:
        var_dict = json.load(f)
    meta = var_dict.setdefault("metadata", {})
    keys = [k for k in var_dict if k != "metadata"] if cut_vals is None else [str(c) for c in cut_vals if str(c) in var_dict]
    shifts = meta.get("smoothing_shifts", {})
    cuts = ddt_windowcuts(mt, pt)
    neff = None
    for key in keys:
        var_map, MT_PT_edges, PT_edges, RT_edges = var_dict[key]
        var_map = np.array(var_map)
        edges = (np.array(MT_PT_edges), np.array(PT_edges), np.array(RT_edges))
        if neff is None:
            neff = ddt_neff(mt, pt, rt_sel, weight, *edges)
        percent = meta.get("efficiency_map", {}).get(key)
        if percent is not None:
            targets = [percent/100] * var_map.shape[2]
        else:
            mt_pt_bin, pt_bin, rt_bin = ddt_bins(mt[cuts], pt[cuts], rt_sel[cuts], *edges)
            passed = var[cuts] > var_map[mt_pt_bin, pt_bin, rt_bin]
            w = weight[cuts].astype(np.float64)
            targets = [float(np.sum(w[passed & (rt_bin == k)]) / max(np.sum(w[rt_bin == k]), 1e-300)) for k in range(var_map.shape[2])]
        shifts.setdefault(key, {}).update(_smoothing_shifts_for_map(var_map, edges, mt, pt, rt_sel, var, weight, neff, smooth_sigmas, targets, recalibration_rt_sel))
    meta["neff"] = neff
    meta["smoothing_shifts"] = shifts
    with open(out_name or ddt_name, 'w') as f:
        json.dump(var_dict, f, cls=NumpyArrayEncoder)

def create_DDT_map_dict(mt, pt, rt_sel, var, weight, percents, cut_vals, ddt_name, smooth_sigmas=(), recalibration_rt_sel=None):
    '''
    Creates a dictionary of DDT 2D maps for a range of cut_vals and background efficiencies (percents).
    Each DDT map is a 3D array of tagger thresholds binned in (mt/pt, pt, rt) space, where RT is a boolean array
    indicating whether the events passes the RT selection of interest
    smoothed and stored along with the corresponding bin edges.

    Inputs:
    - mt, pt (continuous event variables)
    - rt_sel: boolean selection of whether the event passes the RT selection of interest
    - var: the tagger variable (e.g. BDT score)
    - weight: event weights
    - percents: list of background efficiencies (e.g. [10, 20, 30])
    - cut_vals: corresponding working point names (e.g. [0.1, 0.2, 0.3])
    - ddt_name: output filename for the JSON dictionary
    - smooth_sigmas: widths (bins) for which the information for on-the-fly smoothing (calculate_varDDT(smooth_sigma))
      is stored in the metadata: N_eff per bin and, for each width and working point, one shift per RT slice. The
      stored thresholds are the same with or without it.
    - recalibration_rt_sel: {sigma: RT slice of each event} if the RT selection applied with this map is smoothed on the
      fly too (default: rt_sel)
    '''
    cuts = ddt_windowcuts(mt, pt)

    var_dict = {}
    neff = None
    shifts = {}

    for cut_val, percent in zip(cut_vals, percents):
        print(f"Creating DDT 2D map for cut value {cut_val}, efficiency {percent}%")
        var_map, MT_PT_edges, PT_edges, RT_edges = varmap(mt, pt, rt_sel, var, weight, percent, cut_val)
        var_dict[str(cut_val)] = (var_map.tolist(), MT_PT_edges.tolist(), PT_edges.tolist(), RT_edges)
        if smooth_sigmas:
            edges = (MT_PT_edges, PT_edges, RT_edges)
            if neff is None:
                neff = ddt_neff(mt, pt, rt_sel, weight, *edges)
            shifts[str(cut_val)] = _smoothing_shifts_for_map(var_map, edges, mt, pt, rt_sel, var, weight, neff, smooth_sigmas, [percent/100] * var_map.shape[2], recalibration_rt_sel)

    # Adding some generation informations
    var_dict["metadata"] = {
        "event_count": len(weight),
        "sum_of_weight": float(np.sum(weight)),
        "efficiency_map": { str(cut_val): float(percent) for cut_val, percent in zip(cut_vals, percents) }
    }
    if smooth_sigmas:
        var_dict["metadata"]["neff"] = neff
        var_dict["metadata"]["smoothing_shifts"] = shifts

    if ddt_name is None:
        ddt_name = 'ddt_' + str(var) + '_' + datetime.now().strftime('%Y%m%d') + '.json'
    with open(ddt_name, 'w') as f:
        json.dump(var_dict, f, cls=NumpyArrayEncoder)

def calculate_varDDT(mt, pt, rt_sel, var, cut_val, ddt_name, smear=1.0, smooth_sigma=0.0):
    '''
    Applies a DDT transformation to 'var' using a DDT map in (mt, pt) space.

    Inputs:
    - mt, pt (continous event variable)
    - rt_sel, boolean selection for whether the events pass the RT selection scheme of interest
    - var: tagger variable (e.g. BDT score)
    - cut_val: key in the DDT map dict to use (e.g. 0.1, 0.2)
    - ddt_name: path to JSON file containing the DDT map
    - smear: width (bins) of the Gaussian smear applied to the map
    - smooth_sigma: if > 0, the N_eff-weighted smoothing of this width (bins) is applied to the stored map on the fly
      instead of the Gaussian smear (regularize_ddt_map plus the stored per-slice shifts). The map file must carry the
      information for this width (create_DDT_map_dict(smooth_sigmas) or add_ddt_smoothing_info); the file is not changed.

    Returns:
    - varDDT: array of DDT-transformed tagger values
    '''

    if not osp.exists(ddt_name):
        raise FileNotFoundError(f"The file {ddt_name} does not exist.")

    with open(ddt_name, 'r') as f:
        var_dict = json.load(f)

    if str(cut_val) not in var_dict:
        raise KeyError(f"The key {cut_val} does not exist in the dictionary.")

    # Get the smoothed map and bin edges
    var_map, MT_PT_edges, PT_edges, RT_edges = var_dict[str(cut_val)]
    var_map = np.array(var_map)
    MT_PT_edges = np.array(MT_PT_edges)
    PT_edges = np.array(PT_edges)
    RT_edges = np.array(RT_edges)

    if smooth_sigma > 0:
        # N_eff-weighted smoothing on the fly; the stored map is not changed
        meta = var_dict.get("metadata", {})
        shifts = meta.get("smoothing_shifts", {}).get(str(cut_val), {}).get(str(float(smooth_sigma)))
        if "neff" not in meta or shifts is None:
            raise KeyError(f"{ddt_name} has no smoothing information for cut value {cut_val} and smooth_sigma {smooth_sigma}: "
                           "add it with add_ddt_smoothing_info (apply_DDT.py --smooth_sigmas)")
        var_map_smooth = smooth_ddt_map(var_map, np.array(meta["neff"]), smooth_sigma, shifts)
    else:
        # Applying a smearing on the DDT map. Explicitly disabling the smearing on the RT axis
        var_map_smooth = gaussian_filter(var_map, (smear, smear, 0.0))

    # Bin index lookup with digitize (overflow entries are folded into the edge bins)
    mt_pt_bin, pt_bin, rt_bin = ddt_bins(mt, pt, rt_sel, MT_PT_edges, PT_edges, RT_edges)

    # Apply DDT: var - DDT map value at the appropriate bin
    varDDT = var - var_map_smooth[mt_pt_bin, pt_bin, rt_bin]

    return varDDT

def apply_hemveto(cols):
    cols = cols.select(svj.veto_HEM(cols.arrays['ak4_subl_eta'],cols.arrays['ak4_subl_phi'],cols.arrays['ak4_subl_pt']))
    cols.cutflow['hem_veto'] = len(cols)
    return cols

def check_if_model_exists(model_file, xrootd_url):
    # Check if the file exists locally
    if not os.path.exists(model_file):
        print(f"File {model_file} not found. Downloading from {xrootd_url}...")
        try:
            os.makedirs(os.path.dirname(model_file), exist_ok=True)  # Ensure directory exists
            subprocess.run(["xrdcp", xrootd_url+model_file, model_file], check=True)
            print(f"Downloaded {model_file} successfully.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error downloading {model_file}: {e}")
            return None

# Direct selection
SELECTION_RT_SIGNAL_REGION = 1.18
# DDT selection for RT
SELECTION_RTDDT_SIGNAL_REGION = 1.20
RT_DDT_PATH = 'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/rt_ddt/'
RT_DDT_FILE = 'models/rt_ddt_map_ANv6_3d.json'

RT_DDT_SMEAR = 0.2
NO_RT_DDT_SMEAR = 0.5
# Width (bins) of the N_eff-weighted on-the-fly smoothing of all DDT maps in the selections below (calculate_varDDT
# smooth_sigma); 0 = off, the Gaussian smears above apply. The map files must carry the smoothing information for this
# width (apply_DDT.py --smooth_sigmas, add_ddt_smoothing_info).
DDT_SMOOTH_SIGMA = 0.0

def apply_rt_signalregion(cols):
    cols = cols.select(cols.arrays['rt'] > SELECTION_RT_SIGNAL_REGION)
    cols.cutflow['rt_signalregion'] = len(cols)
    return cols

def _get_rt_ddt_val(cols, rt_ddt_map=RT_DDT_FILE, xrootd_url=RT_DDT_PATH):
    check_if_model_exists(rt_ddt_map, xrootd_url)
    mT = cols.to_numpy(['mt']).ravel()
    pT = cols.to_numpy(['pt']).ravel()
    rT = cols.to_numpy(['rt']).ravel()
    mask = np.ones_like(mT, dtype=bool)
    return calculate_varDDT(mT, pT, mask, rT, SELECTION_RTDDT_SIGNAL_REGION, rt_ddt_map, smear=RT_DDT_SMEAR, smooth_sigma=DDT_SMOOTH_SIGMA)

def apply_rt_signalregion_ddt(cols, rt_ddt_map=RT_DDT_FILE, xrootd_url=RT_DDT_PATH):
    ddt_val = _get_rt_ddt_val(cols, rt_ddt_map, xrootd_url)
    cols = cols.select(ddt_val > 0.0)
    cols.cutflow['rt_signalregion_ddt'] = len(cols)
    return cols


DDT_PATH_CUTBASED = 'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/cutbased_ddt/'
DDT_FILE_CUTBASED = 'models/cutbased_ddt_map_ANv6_3d.json'
DDT_FILE_CUTBASED_RT_DDT = 'models/cutbased_ddt_map_withRT_ANv6_3d.json'

def cutbased_ddt(cols, lumi, cut_val, ddt_map_file, xrootd_url):
    check_if_model_exists(ddt_map_file, xrootd_url)
    # Get features necessary to apply the DDT
    mT = cols.to_numpy(['mt']).ravel() # make one d ... don't ask why it's not
    pT = cols.to_numpy(['pt']).ravel()
    ecfm2b1 = cols.to_numpy(['ecfm2b1']).ravel()
    # For this function, we are assuming the RT signal region is applied, a separation method
    # Will need to be computed for the antiloose control region
    rt_mask = np.ones_like(mT, dtype=bool)
    ddt_val = calculate_varDDT(mT, pT, rt_mask, ecfm2b1, cut_val, ddt_map_file, smear=NO_RT_DDT_SMEAR, smooth_sigma=DDT_SMOOTH_SIGMA)
    return ddt_val

def apply_cutbased(cols, cut_val=0.09):
    cols = apply_rt_signalregion(cols)
    cols = cols.select(cols.arrays['ecfm2b1'] > cut_val)
    cols.cutflow['cutbased'] = len(cols)
    return cols

def apply_cutbased_ddt(cols, lumi, cut_val, ddt_map_file=DDT_FILE_CUTBASED, xrootd_url=DDT_PATH_CUTBASED) :
    cols = apply_rt_signalregion(cols)
    ddt_val = cutbased_ddt(cols, lumi, cut_val, ddt_map_file, xrootd_url)

    # Now cut on the DDT above 0.0 (referring to above the ecfm2b1 cut value)
    cols = cols.select(ddt_val > 0.0) # mask for the selection
    cols.cutflow['cutbased_ddt'] = len(cols)
    return cols

def apply_rtcutbased_ddt(cols, lumi, cut_val, ddt_map_file=DDT_FILE_CUTBASED_RT_DDT, xrootd_url=DDT_PATH_CUTBASED):
    cols = apply_rt_signalregion_ddt(cols)
    ddt_val = cutbased_ddt(cols, lumi, cut_val, ddt_map_file, xrootd_url)
    cols = cols.select(ddt_val > 0.0)
    cols.cutflow['cutbased_rt_ddt'] = len(cols)
    return cols

def apply_cutbasedCR(cols):
    cols = apply_rt_signalregion(cols)
    cols = cols.select(cols.arrays['ecfm2b1'] < 0.032)
    cols.cutflow['cutbasedCR'] = len(cols)
    return cols

def apply_cutbasedCRloose(cols):
    cols = cols.select(cols.arrays['ecfm2b1'] < 0.032)
    cols.cutflow['cutbasedCRloose'] = len(cols)
    return cols

def apply_anticutbased_ddt(cols, lumi, cut_val, ddt_map_file=DDT_FILE_CUTBASED, xrootd_url=DDT_PATH_CUTBASED) :
    cols = apply_rt_signalregion(cols)
    ddt_val = cutbased_ddt(cols, lumi, cut_val, ddt_map_file, xrootd_url)
    cols = cols.select(ddt_val < 0.0) # mask for the selection
    cols.cutflow['anticutbased_ddt'] = len(cols)
    return cols

def apply_anticutbased(cols, cut_val=0.09):
    cols = apply_rt_signalregion(cols)
    cols = cols.select(cols.arrays['ecfm2b1'] < cut_val)
    cols.cutflow['anticutbased'] = len(cols)
    return cols

def apply_antirtcutbased_ddt(cols, lumi, cut_val, ddt_map_file=DDT_FILE_CUTBASED, xrootd_url=DDT_PATH_CUTBASED) :
    cols = apply_rt_signalregion_ddt(cols)
    ddt_val = cutbased_ddt(cols, lumi, cut_val, ddt_map_file, xrootd_url)
    cols = cols.select(ddt_val < 0.0) # mask for the selection
    cols.cutflow['anticutbased_ddt'] = len(cols)
    return cols

def _calc_rtcutbased_ddt_val(cols, lumi, cut_val, rt_ddt_file, rt_cut_val, ddt_map_file, xrootd_url):
    # Calculating the cutbased DDT value, if the RT DDT file is provided, we we will compute the results with the double DDT method
    check_if_model_exists(ddt_map_file, xrootd_url)
    # Get features necessary to apply the DDT
    mT = cols.to_numpy(['mt']).ravel()
    pT = cols.to_numpy(['pt']).ravel()
    rT = cols.to_numpy(['rt']).ravel()
    ecfm2b1 = cols.to_numpy(['ecfm2b1']).ravel()
    t_mask = np.ones_like(mT, dtype=bool)
    rt_mask = (rT > rt_cut_val)
    if rt_ddt_file is not None:
        rt_mask = calculate_varDDT(mT, pT, t_mask, rT, rt_cut_val, rt_ddt_file, smear=RT_DDT_SMEAR, smooth_sigma=DDT_SMOOTH_SIGMA) > 0.0
    ddt_val = calculate_varDDT(mT, pT, rt_mask, ecfm2b1, cut_val, ddt_map_file, smear=RT_DDT_SMEAR, smooth_sigma=DDT_SMOOTH_SIGMA)
    return ddt_val


def apply_antiloosecutbased_ddt(cols, lumi, cut_val, ddt_map_file=DDT_FILE_CUTBASED, xrootd_url=DDT_PATH_CUTBASED) :
    ddt_val = _calc_rtcutbased_ddt_val(cols, lumi, cut_val, None, SELECTION_RT_SIGNAL_REGION, ddt_map_file, xrootd_url)
    cols = cols.select(ddt_val < 0.0) # mask for the selection
    cols.cutflow['anticutbased_ddt'] = len(cols)
    return cols

def apply_antiloosertcutbased_ddt(cols, lumi, cut_val, ddt_map_file=DDT_FILE_CUTBASED_RT_DDT, xrootd_url=DDT_PATH_CUTBASED):
    ddt_val = _calc_rtcutbased_ddt_val(cols, lumi, cut_val, RT_DDT_FILE, SELECTION_RTDDT_SIGNAL_REGION, ddt_map_file, xrootd_url)
    cols = cols.select(ddt_val < 0.0) # mask for the selection
    cols.cutflow['anticutbased_rtddt'] = len(cols)
    return cols

def apply_antiloosecutbased(cols, cut_val=0.09):
    cols = cols.select(cols.arrays['ecfm2b1'] < cut_val)
    cols.cutflow['antiloosecutbased'] = len(cols)
    return cols

# Relative path to the BDT
# This specific BDT was chosen to be used during the L3 review
bdt_model_file = 'models/svjbdt_obj_rev_version.json'

def split_bdt(sel):
    parts = sel.split('=')
    if len(parts)==2:
        try:
            bdt_cut = float(parts[1])
        except ValueError:
            # Handle the case where the number following 'bdt=' is not valid
            print(f"Invalid number {parts[1]} following 'bdt='.")
    else:
        raise InvalidSelectionException(sel=selection)
    return parts[1]

def calc_bdt_scores(X, model_file=bdt_model_file):
    import xgboost as xgb
    # Load the model and get the predictions
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(model_file)
    with time_and_log(f'Calculating xgboost scores for {bdt_model_file}...'):
        score = xgb_model.predict_proba(X)[:,1]
    del xgb_model
    return score


DDT_PATH_BDTBASED='root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/BDT_based/'
DDT_FILE_BDTBASED='models/bdt_ddt_map_ANv6_3d.json'
DDT_FILE_BDTBASED_RT_DDT='models/bdt_ddt_withRT_ANv6_3d.json'

def apply_bdtbased(cols,wp,lumi,anti=False,model_file=bdt_model_file,ddt_map_file=DDT_FILE_BDTBASED, xrootd_url=DDT_PATH_BDTBASED):
    check_if_model_exists(ddt_map_file, xrootd_url)
    check_if_model_exists(model_file, xrootd_url)

    cols = apply_rt_signalregion(cols) if "_withRT_" not in ddt_map_file else apply_rt_signalregion_ddt(cols)
    if len(cols)==0:
        return cols
    X = cols.to_numpy(read_training_features(model_file))
    score = calc_bdt_scores(X)
    mT = cols.to_numpy(['mt']).ravel()
    pT = cols.to_numpy(['pt']).ravel()
    rT_mask = np.ones_like(mT, dtype=bool)
    bdt_ddt_score = calculate_varDDT(mT, pT, rT_mask, score, wp, ddt_map_file, smear=RT_DDT_SMEAR, smooth_sigma=DDT_SMOOTH_SIGMA)

    if anti:
        cols = cols.select(bdt_ddt_score < 0.0) # mask for the selection
        cols.cutflow['ddt(antibdt)'] = len(cols)
    else:
        cols = cols.select(bdt_ddt_score > 0.0) # mask for the selection
        cols.cutflow['ddt(bdt)'] = len(cols)
    return cols

def apply_antiloosebdt(cols,wp,lumi,rt_ddt_file=None,model_file=bdt_model_file,ddt_map_file=DDT_PATH_BDTBASED,xrootd_url=DDT_PATH_BDTBASED):
    # Running the BDT calculation. If the RT_DDT files is provided, the double DDT method will be used
    check_if_model_exists(ddt_map_file, xrootd_url)
    check_if_model_exists(model_file, xrootd_url)

    if len(cols)==0:
        return cols
    X = cols.to_numpy(read_training_features(model_file))
    score = calc_bdt_scores(X)
    mT = cols.to_numpy(['mt']).ravel()
    pT = cols.to_numpy(['pt']).ravel()
    rT = cols.to_numpy(["rt"]).ravel()
    t_mask = np.ones_like(mT, dtype=bool)
    rt_mask = (rT > SELECTION_RT_SIGNAL_REGION)
    if rt_ddt_file is not None:
        rt_mask = calculate_varDDT(mT, pT, t_mask, rT, SELECTION_RTDDT_SIGNAL_REGION, rt_ddt_file, smear=RT_DDT_SMEAR, smooth_sigma=DDT_SMOOTH_SIGMA) > 0.0
    bdt_ddt_score = calculate_varDDT(mT, pT, rt_mask, score, wp, ddt_map_file, smear=RT_DDT_SMEAR, smooth_sigma=DDT_SMOOTH_SIGMA)

    cols = cols.select(bdt_ddt_score < 0.0) # mask for the selection
    cols.cutflow['loose_ddt(antibdt)'] = len(cols)
    return cols

"""Calculating scale factor for jet variable modeling"""

SF_PATH = 'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/cutbased_ddt/'
SF_FILE = 'models/jetvar_model_uncertainty.json'

def get_model_sf(cols, var="cen", sf_file=SF_FILE, xrootd_url=SF_PATH):
    check_if_model_exists(sf_file, xrootd_url)
    sf_container = json.load(open(sf_file, "r"))
    var_bin = np.array(sf_container["ddt_bins"])
    sf = np.array(sf_container["0.1"]["sf"]["val"])
    stat_unc =  np.array(sf_container["0.1"]["sf"]["stat_unc"])
    syst_unc =  np.array(sf_container["0.1"]["sf"]["syst_unc"])
    unc = np.sqrt(stat_unc **2 + syst_unc**2)

    mT = cols.to_numpy(['mt']).ravel()
    pT = cols.to_numpy(['pt']).ravel()
    ecf = cols.to_numpy(['ecfm2b1']).ravel()
    m = np.ones_like(mT, dtype=bool)
    ddt_val = calculate_varDDT(mT, pT, m, ecf, 0.1, DDT_FILE_CUTBASED_RT_DDT, smear=RT_DDT_SMEAR, smooth_sigma=DDT_SMOOTH_SIGMA)

    bin_idx = np.digitize(ddt_val, bins=var_bin)
    if var == "cen":
        return sf[bin_idx]
    elif var == "up":
        return sf[bin_idx] * (1+unc[bin_idx])
    else:
        return sf[bin_idx] * (1-unc[bin_idx])


class InvalidSelectionException(Exception):
    def __init__(self, msg='Unknown selection {}; choices are "preselection", "cutbased", or "bdt=X.XXX".', sel="", *args, **kwargs):
        msg = msg.format(sel)
        super().__init__(msg, *args, **kwargs)
