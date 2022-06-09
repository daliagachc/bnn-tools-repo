# probably this functions should go

"""
6/2/21

diego.aliaga at helsinki dot fi
based on Runlongs code
"""

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
    locator = mdates.AutoDateLocator(minticks=60, maxticks=60)
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
    if isinstance(o, xr.Dataset):
        o = o['dndlDp']

    o = set_time(o)
    o = set_Dp(o)

    # q1 = o.quantile(.05)
    # q2 = o.quantile(.95)
    vmin = 1e1
    vmax = 1e6

    # set a nice color bar
    cm = plt.get_cmap('plasma').copy()
    cm.set_bad(color=cm(0))

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

    o.plot(
        **s1
    )

    ax = plt.gca()

    format_ticks(ax)

    ax.grid(c='w', ls='--', alpha=.5)
    ax.grid(c='w', ls='-', which='minor', lw=.5, alpha=.3)

    plt.gcf().set_figwidth(10)


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


def resample_ts(o, dt):
    orig_dim = list(o.dims)

    orgi_dt = set_sec(o)['secs'].diff('secs').mean().item()

    assert dt >= orgi_dt, 'you are trying to downsample when you should be upsampling'

    o = set_sec(o)
    ddt = np.round(o['secs'] / dt) * dt
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


def dp_regrid(da,*, n_subs, log_dy):

    ds1 = set_lDp(da).reset_coords(drop=True)

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
        # kwargs=dict(fill_value="extrapolate")
    )


    g = d1.groupby(np.round(d1['lDp']/log_dy)*log_dy)

    cs = g.count()['dndlDp'].median('time').reset_coords(drop=True)
    # return cs

    dsout = g.mean()[{'lDp':cs>=(n_subs/2)}]

    # dout['time'] = dout['secs'].astype('datetime64[s]')

    # dout['Dp'] = 10 ** dout['lDp']

    return dsout