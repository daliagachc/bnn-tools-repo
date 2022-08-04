# probably this functions should go

"""
6/2/21

diego.aliaga at helsinki dot fi
based on Runlongs code
"""

####################
# before using functions from here
# most likely you need to first import bnn_array (not here but in the destination script)

# import most used packages
# import os
# import glob
# import sys
# import pprint
# import datetime as dt
# import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
# import seaborn as sns
# import cartopy as crt

# import pandas as pd
# import scipy.interpolate
# import bnn_tools.bnn_array

from xarray.plot.plot import _infer_interval_breaks as infer_interval_breaks

## constants come here

# All the variables are in the SI metric units, e.g., dp in m

def format_ticks(ax):
    import matplotlib.dates as mdates
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    locator = mdates.AutoDateLocator(minticks=50, maxticks=60)
    ax.xaxis.set_minor_locator(locator)
    # ax.xaxis.set_minor_formatter(formatter)

    for xlabels in ax.get_xticklabels():
        xlabels.set_rotation(0)
        xlabels.set_ha("center")

def format_ticks2(ax,M,m):
    import matplotlib.dates as mdates
    locator = mdates.AutoDateLocator(minticks=int(M/2), maxticks=int(2*M))
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    locator = mdates.AutoDateLocator(minticks=int(m/2), maxticks=(m*2))
    ax.xaxis.set_minor_locator(locator)
    # ax.xaxis.set_minor_formatter(formatter)

    for xlabels in ax.get_xticklabels():
        xlabels.set_rotation(0)
        xlabels.set_ha("center")




####################
# bnn funs
####################

def from_time2sec(o):
    date = o['time']
    s1 = date - np.datetime64(0, 'Y')
    s2 = s1 / np.timedelta64(1, 's')
    o = o.assign_coords({'secs': s2})
    return o


def from_sec2time(o):
    secs = o['secs'].astype('datetime64[s]')
    o = o.assign_coords({'time': secs})
    return o


def from_Dp2lDp(o):
    lDp = np.log10(o['Dp'])
    o = o.assign_coords({'lDp': lDp})
    return o


def from_lDp2Dp(o):
    Dp = 10 ** (o['lDp'])
    o = o.assign_coords({'Dp': Dp})
    return o


def from_lDp2dlDp(o):
    o = set_lDp(o)
    lDp = o['lDp']
    borders = infer_interval_breaks(lDp)
    d = borders[1:] - borders[:-1]
    d1 = lDp * 0 + d
    o = o.assign_coords({'dlDp': d1})
    return o


def from_Dp2dDp(o):
    #todo this should de done in lDp and not Dp since interval breaks are prone to erros in Dp
    Dp = o['Dp']
    borders = infer_interval_breaks(Dp)
    d = borders[1:] - borders[:-1]
    d1 = Dp * 0 + d
    o = o.assign_coords({'dDp': d1})
    return o


def set_time(o):
    dims = list(o.dims)
    coords = list(o.coords)
    # o_coords = set(coords)-set(dims)

    if 'time' not in coords:
        o = from_sec2time(o)

    if 'time' not in dims:
        o = o.swap_dims({'secs': 'time'})

    return o


def set_Dp(o):
    dims = list(o.dims)
    coords = list(o.coords)
    # o_coords = set(coords)-set(dims)

    if 'Dp' not in coords:
        # print(coords)
        o = from_lDp2Dp(o)

    if 'Dp' not in dims:
        o = o.swap_dims({'lDp': 'Dp'})

    return o


def set_lDp(o):
    dims = list(o.dims)
    coords = list(o.coords)
    # o_coords = set(coords)-set(dims)

    if 'lDp' not in coords:
        o = from_Dp2lDp(o)

    if 'lDp' not in dims:
        o = o.swap_dims({'Dp': 'lDp'})

    return o


def set_sec(o):
    dims = list(o.dims)
    coords = list(o.coords)
    # o_coords = set(coords)-set(dims)

    if 'secs' not in coords:
        o = from_time2sec(o)

    if 'secs' not in dims:
        o = o.swap_dims({'time': 'secs'})

    return o


def plot_psd(o, **kwargs):
    o_ = o
    if isinstance(o_, xr.Dataset):
        o_ = o_['dndlDp']

    o_ = set_time(o_)
    o_ = set_Dp(o_)
    onn = o_.notnull()

    #differentiate between null and negative.
    # - null is plotted as missing
    # - negative becomes low value.
    o_ = o_.where(o_ > 0, .00001).where(onn)

    # q1 = o.quantile(.05)
    # q2 = o.quantile(.95)
    vmin = 1e1
    vmax = 1e6

    # set a nice color bar
    cm = plt.get_cmap('plasma').copy()
    # cm.set_bad(color=cm(0))
    cm.set_bad(color='w')

    s1 = dict(
        x='time',
        y='Dp',
        norm=mpl.colors.LogNorm(),
        cmap=cm,
        vmin=vmin,
        vmax=vmax,
        yscale='log',
    )

    s1.update(kwargs)

    # print(s1)

    res = o_.plot(
        **s1
    )

    axs = res.axes

    if not isinstance(axs,np.ndarray):
        # ax = plt.gca()
        axs = np.array([axs])

    for ax in axs.flatten():
        format_ticks(ax)

        ax.grid(c='w', ls='--', alpha=.5)
        ax.grid(c='w', ls='-', which='minor', lw=.5, alpha=.3)

    plt.gcf().set_figwidth(10)
    return res 


def get_dN(o, d1, d2):
    if isinstance(o, xr.Dataset):
        o = o['dndlDp']
    assert o.name == 'dndlDp'
    o = from_lDp2dlDp(o)
    o = set_Dp(o)
    o1 = o.loc[{'Dp': slice(d1, d2)}]
    dmin = o1['Dp'].min().item()
    dmax = o1['Dp'].max().item()

    o1['dN'] = (o1 * o1['dlDp'])

    dN = o1['dN']

    return dN, dmin, dmax


def get_N(o, d1, d2):
    dN, dmin, dmax = get_dN(o, d1, d2)

    N = dN.sum('Dp')
    N.name = 'N'
    return N, dmin, dmax


def resample_ts(o, dt_secs):
    orig_dim = list(o.dims)

    orgi_dt = set_sec(o)['secs'].diff('secs').median().item()

    assert dt_secs >= orgi_dt, 'you are trying to downsample when you should be upsampling'

    o_ = set_time(o)

    o_['time'] = o_['time'] + pd.Timedelta(dt_secs/2,'seconds')

    o_ = o_.resample({'time':pd.Timedelta(dt_secs,'seconds')}).mean()


    # ddt = np.round(o_['secs'] / dt_secs) * dt_secs
    # o_ = o_.groupby(ddt).mean()

    if 'time' in orig_dim:
        o_ = set_time(o_)
    if 'secs' in orig_dim:
        o_ = set_sec(o_)

    return o_

def resample_ts_old(o, dt_secs):
    orig_dim = list(o.dims)

    orgi_dt = set_sec(o)['secs'].diff('secs').median().item()

    assert dt_secs >= orgi_dt, 'you are trying to downsample when you should be upsampling'

    o = set_sec(o)
    ddt = np.round(o['secs'] / dt_secs) * dt_secs
    o = o.groupby(ddt).mean()

    if 'time' in orig_dim:
        o = set_time(o)
    if 'secs' in orig_dim:
        o = set_sec(o)

    return o


def upsample_ts(o, dt):

    orgi_dt = set_sec(o)['secs'].diff('secs').mean().item()

    assert dt <= orgi_dt, 'you are trying to upsample when you should be downsampling'

    orig_dim = list(o.dims)

    o = set_time(o)

    o = o.resample({'time': pd.Timedelta(dt, unit='s')}).interpolate('linear')

    if 'time' in orig_dim:
        o = set_time(o)
    if 'secs' in orig_dim:
        o = set_sec(o)

    return o


def dp_regrid_old(*, da, n_subs, log_dy):

    darray = set_lDp(da)
    dy = log_dy / n_subs
    dm = np.ceil(darray['lDp'].min().item() / log_dy) * log_dy
    dM = np.ceil(darray['lDp'].max().item() / log_dy) * log_dy

    dms = np.arange(dm - (((n_subs - 1) / 2) * dy), dM, dy)

    d1 = darray.interp({'lDp': dms})

    dout = d1.coarsen(**{'lDp': n_subs}, boundary='trim').mean().reset_coords(drop=True)

    # dout['time'] = dout['secs'].astype('datetime64[s]')

    # dout['Dp'] = 10 ** dout['lDp']

    return dout


def dp_regrid(da, *, n_subs, log_dy):

    if isinstance(da,xr.DataArray):
        assert da.name == 'dndlDp'
        da_  = da
    elif isinstance(da,xr.Dataset):
        da_ = da['dndlDp']

    ds1 = set_lDp(da_).reset_coords(drop=True)

    dy = log_dy/n_subs

    ints = infer_interval_breaks(ds1['lDp'],check_monotonic=True)

    # print(ints)

    ym_ = ints[0]
    yM_ = ints[-1]

    yM = np.round((yM_/dy))*dy
    ym = np.round((ym_/dy))*dy

    yl = np.arange(ym,yM,dy)

    d1 = ds1.interp(
        {'lDp': yl},
        kwargs=dict(fill_value="extrapolate")
    )


    g = d1.groupby(np.round(d1['lDp']/log_dy)*log_dy)

    # return g

    # we take time as a dummy var.
    cs = g.count().median('time').reset_coords(drop=True)
    # return cs

    dmean = g.mean()
    dmean = dmean.where(dmean['lDp']>=ym_).where(dmean['lDp']<=yM_)


    #todo change to 2 below
    # return dmean,cs
    # dsout = dmean[{'lDp':cs>=(n_subs/2)}]
    dsout = dmean.where(cs>=(n_subs/2)).dropna('lDp',how='all')

    # dout['time'] = dout['secs'].astype('datetime64[s]')

    # dout['Dp'] = 10 ** dout['lDp']

    return dsout

def get_exact_N(dc1, Dp_min, Dp_max):
    """
    counts the exact number of particles in the range Dp_min Dp_max using linear intregration
    Parameters
    ----------
    dc1_ : array like
    Dp_min : float
    Dp_max : float

    Returns
    -------
    array like

    """
    assert dc1.name == 'dndlDp', 'you can only calc N on dndlogDp'
    assert Dp_min < Dp_max, 'd1 not < d2'

    dc1_ = dc1.bnn.set_lDp()
    _breaks = infer_interval_breaks(dc1_['lDp'])
    lDp1 = _breaks[0]
    lDp2 = _breaks[-1]

    ld1 = np.log10(Dp_min)
    ld2 = np.log10(Dp_max)

    assert ld1 >= lDp1
    assert ld2 <= lDp2


    orig_dis = dc1_['lDp'].diff('lDp').median().item()

    new_full_dis = ld2 - ld1

    dis = np.min([orig_dis, new_full_dis])

    dis_i = int(np.ceil(new_full_dis / dis) * 2)

    new_ld_list = np.linspace(ld1, ld2, dis_i)

    new_arr = dc1_.interp({'lDp': new_ld_list}, method='linear')

    # new_arr['dndlDp'].bnn.set_Dp().plot()

    # dc1['dndlDp'].bnn.set_Dp().plot(norm=mpl.colors.LogNorm(vmin=1e1),yscale='log')

    new_inte = set_Dp(new_arr).integrate('lDp')

    new_inte = new_inte.expand_dims({'Dp_interval': [pd.Interval(Dp_min,Dp_max)]})
    new_inte.name = 'N'

    return new_inte