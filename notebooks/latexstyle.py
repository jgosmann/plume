import matplotlib as mpl

ltx_to_mpl_pt_factor = 72.0 / 72.27


def setup():
    mpl.rcParams['lines.linewidth'] = 0.4 * ltx_to_mpl_pt_factor
    mpl.rcParams['lines.markeredgewidth'] = 0.2 * ltx_to_mpl_pt_factor
    mpl.rcParams['lines.markersize'] = 2.4 * ltx_to_mpl_pt_factor
    mpl.rcParams['patch.linewidth'] = 0.4 * ltx_to_mpl_pt_factor
    mpl.rcParams['font.size'] = 6 * ltx_to_mpl_pt_factor
    mpl.rcParams['axes.linewidth'] = 0.4 * ltx_to_mpl_pt_factor
    #mpl.rcParams['axes.labelsize'] = 'large'

    mpl.rcParams['xtick.major.size'] = 1.6 * ltx_to_mpl_pt_factor
    mpl.rcParams['xtick.minor.size'] = 0.8 * ltx_to_mpl_pt_factor
    mpl.rcParams['xtick.major.width'] = 0.2 * ltx_to_mpl_pt_factor
    mpl.rcParams['xtick.minor.width'] = 0.2 * ltx_to_mpl_pt_factor
    mpl.rcParams['xtick.major.pad'] = 1.6 * ltx_to_mpl_pt_factor
    mpl.rcParams['xtick.minor.pad'] = 1.6 * ltx_to_mpl_pt_factor

    mpl.rcParams['ytick.major.size'] = 1.6 * ltx_to_mpl_pt_factor
    mpl.rcParams['ytick.minor.size'] = 0.8 * ltx_to_mpl_pt_factor
    mpl.rcParams['ytick.major.width'] = 0.2 * ltx_to_mpl_pt_factor
    mpl.rcParams['ytick.minor.width'] = 0.2 * ltx_to_mpl_pt_factor
    mpl.rcParams['ytick.major.pad'] = 1.6 * ltx_to_mpl_pt_factor
    mpl.rcParams['ytick.minor.pad'] = 1.6 * ltx_to_mpl_pt_factor

    mpl.rcParams['grid.linewidth'] = 0.2 * ltx_to_mpl_pt_factor
    mpl.rcParams['legend.borderpad'] = 0.2 * ltx_to_mpl_pt_factor

    mpl.rcParams['figure.dpi'] = 160
    mpl.rcParams['figure.figsize'] = (5.7874 * 0.925, 4.3406 * 0.925)


def style_axes(*axes):
    for ax in axes:
        ax.spines.get('right').set_visible(False)
        ax.spines.get('top').set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
