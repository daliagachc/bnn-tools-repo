"""
this module contains basic funcionts that are used in other modules
it should bot import anything from this packages except maybe constants

some of the function are loosely based on Runlongs code

this i as summary of basic_funs.py:

- `format_ticks(ax)`: Format ticks on the x-axis of a matplotlib plot with dates.
- `format_ticks2(ax,M,m)`: Format the x-axis of a matplotlib plot to show dates and ticks.
- `from_time2sec(o)`: Convert time to seconds since 1970-01-01.
- `from_sec2time(o)`: Convert seconds since 1970-01-01 to time.
- `from_Dp2lDp(o)`: Convert particle diameter to log10(particle diameter).
- `from_lDp2Dp(o)`: Convert log10(particle diameter) to particle diameter.
- `from_lDp2dlDp(o)`: Calculate the interval between log10(particle diameters).
- `from_Dp2dDp(o)`: Convert particle diameter to the difference in particle diameters for consecutive bins.
- `set_time(o)`: Set the time as either coordinate or dimension depending on the current form.
- `set_Dp(o)`: Set the Dp as either coordinate or dimension depending on the current form.
- `set_lDp(o)`: Set the lDp as either coordinate or dimension depending on the current form.
- `set_sec(o)`: Set the sec as either coordinate or dimension depending on the current form.
- `plot_psd(o, **kwargs)`: Plot particle size distribution with some optional arguments.
- `get_dN(o, d1, d2)`: Calculate the differential number concentration between d1 and d2 micrometers.
- `get_N(o, d1, d2)`: Calculate the total number concentration between d1 and d2 micrometers.
- `resample_ts(o, dt_secs)`: Resamples a time series with a given time step. It assumes that the time stamp is centered aligned. The result is also center aligned.
- `upsample_ts(o, dt)`: Upsamples a time series with a given time step.
- `dp_regrid(da, *, n_subs, log_dy)`: Regrids the diameter distribution of
  an aerosol particle dataset linearly over a logarithmic diameter range.
  The number of subintervals per bin is controlled by `n_subs`--the more the
  better accuracy and more computation power needed--and the size of
  the interval in logarithmic diameter is controlled by `log_dy`.
- `get_exact_N(dc1, Dp_min, Dp_max)`: Counts the exact number of particles in the range Dp_min to Dp_max using linear integration.
- `bokeh_plot_psd(ds,width)`: Plots particle size distribution using Bokeh plotting library.
- `basic_bokeh_plot(ds, width)`: Creates a basic Bokeh plot.
- `get_colorbar(color_mapper, p, vmax, vmin)`: Creates a color bar for a Bokeh plot.
- `remote_jupyter_proxy_url(port)`: Configures Bokeh's show method when a proxy must be configured.
"""

####################
# before using functions from here
# most likely you need to first import bnn_array (not here but in the destination script)


import numpy as np
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr


from xarray.plot.utils import _infer_interval_breaks as infer_interval_breaks

## constants come here

# All the variables are in the SI metric units, e.g., dp in m

def format_ticks(ax):
    """
    Format ticks on the x-axis of a matplotlib plot with dates.
    In increases the numbers of ticks
    This funcitons is superseded by format_ticks2 and is only kept for
    backward compatibility

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to format ticks.

    Returns
    -------
    None

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [4, 5, 6])
    >>> format_ticks(ax)
    """

    import matplotlib.dates as mdates

    # Set up locator and formatter for major ticks
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # Set up locator for minor ticks
    locator = mdates.AutoDateLocator(minticks=50, maxticks=60)
    ax.xaxis.set_minor_locator(locator)

    # Set the rotation and horizontal alignment of xtick labels
    for xlabels in ax.get_xticklabels():
        xlabels.set_rotation(0)
        xlabels.set_ha("center")

def format_ticks2(ax,M,m):
    """
    Format the x-axis of a matplotlib plot to show dates and ticks.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to be modified.
    M : int
        The maximum number of major ticks to be displayed.
    m : int
        The maximum number of minor ticks to be displayed.

    Returns
    -------
    None

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import datetime
    >>> fig, ax = plt.subplots()
    >>> dates = [datetime.datetime(2022, 1, 1) + datetime.timedelta(days=i) for i in range(365)]
    >>> values = range(365)
    >>> ax.plot(dates, values)
    >>> format_ticks2(ax, 10, 50)
    >>> plt.show()

    """
    import matplotlib.dates as mdates
    locator = mdates.AutoDateLocator(minticks=int(M/2), maxticks=int(2*M))
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    locator = mdates.AutoDateLocator(minticks=int(m/2), maxticks=(m*2))
    ax.xaxis.set_minor_locator(locator)

    for xlabels in ax.get_xticklabels():
        xlabels.set_rotation(0)
        xlabels.set_ha("center")




####################
# bnn funs
####################

from typing import Union
def from_time2sec(o: Union[xr.Dataset, xr.DataArray]) -> Union[xr.Dataset, xr.DataArray]:
    """
    Convert time to seconds since 1970-01-01

    Parameters
    ----------
    o : Union[xr.Dataset, xr.DataArray]
        DataArray or Dataset with a 'time' coordinate

    Returns
    -------
    Union[xr.Dataset, xr.DataArray]
        DataArray or Dataset with a 'secs' coordinate added,
        where 'secs' is the time in seconds since 1970-01-01.
    """
    date = o['time']
    s1 = date - np.datetime64(0, 'Y')
    s2 = s1 / np.timedelta64(1, 's')
    o = o.assign_coords({'secs': s2})
    return o


def from_sec2time(o: Union[xr.Dataset, xr.DataArray]) -> Union[xr.Dataset, xr.DataArray]:
    """
    Convert seconds since 1970-01-01 to time

    Parameters
    ----------
    o : Union[xr.Dataset, xr.DataArray]
        DataArray or Dataset with a 'secs' coordinate

    Returns
    -------
    Union[xr.Dataset, xr.DataArray]
        DataArray or Dataset with a 'time' coordinate added,
        where 'time' is the date and time corresponding to the
        number of seconds since 1970-01-01.
    """
    secs = o['secs'].astype('datetime64[s]')
    o = o.assign_coords({'time': secs})
    return o


def from_Dp2lDp(o: Union[xr.Dataset, xr.DataArray]) -> Union[xr.Dataset,
xr.DataArray]:
    """
    Convert particle diameter to log10(particle diameter)

    Parameters
    ----------
    o : xr.Dataset
        Dataset with a 'Dp' variable

    Returns
    -------
    xr.Dataset
        Dataset with a 'lDp' variable added,
        where 'lDp' is the base-10 logarithm of 'Dp'.
    """
    lDp = np.log10(o['Dp'])
    o = o.assign_coords({'lDp': lDp})
    return o


def from_lDp2Dp(o: Union[xr.Dataset, xr.DataArray]) -> Union[xr.Dataset,
xr.DataArray]:
    """
    Convert log10(particle diameter) to particle diameter

    Parameters
    ----------
    o : xr.Dataset
        Dataset with a 'lDp' variable

    Returns
    -------
    xr.Dataset
        Dataset with a 'Dp' variable added,
        where 'Dp' is the base-10 exponential of 'lDp'.
    """
    Dp = 10 ** (o['lDp'])
    o = o.assign_coords({'Dp': Dp})
    return o


def from_lDp2dlDp(o: Union[xr.Dataset, xr.DataArray]) -> Union[xr.Dataset,
xr.DataArray]:
    """
    Calculate the interval between log10(particle diameters)

    Parameters
    ----------
    o : xr.Dataset
        Dataset with a 'lDp' variable

    Returns
    -------
    xr.Dataset
        Dataset with a 'dlDp' variable added,
        where 'dlDp' is the difference in log10(particle diameter)
        between consecutive bins.
    """
    lDp = o['lDp']
    borders = infer_interval_breaks(lDp)
    d = borders[1:] - borders[:-1]
    d1 = lDp * 0 + d
    return o



def from_Dp2dDp(o: Union[xr.Dataset, xr.DataArray]) -> Union[xr.Dataset,
xr.DataArray]:
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
    """
    Resamples a time series with a given time step.
    it assumes that the time stamp is centered aligned.
    the result is also ceter aligned

    Parameters
    ----------
    o : Union[xr.Dataset, xr.DataArray]
        A dataset or data array with a time dimension.
    dt_secs : float
        The new time step in seconds. Must be greater than or equal to the original time step.

    Returns
    -------
    Union[xr.Dataset, xr.DataArray]
        The resampled time series.

    Notes
    -----
    This function resamples the time series to a new time step using linear interpolation.
    It is intended for use with xarray data arrays and datasets.

    Examples
    --------
    >>> # Resample a dataset with a new time step of 10 seconds
    >>> new_ds = resample_ts(orig_ds, 10)

    >>> # Resample a data array with a new time step of 60 seconds
    >>> new_da = resample_ts(orig_da, 60)
    """


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


    # g = d1.groupby(np.round(d1['lDp']/log_dy)*log_dy)

    g = (
        d1
        .assign_coords({'lDp': np.round(d1['lDp'] / log_dy) * log_dy})
        .groupby('lDp')
    )

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
    Counts the exact number of particles in the range Dp_min to Dp_max using linear integration.

    Parameters
    ----------
    dc1 : xr.DataArray
        Array-like variable containing the number distribution function with log-normal diameter.
    Dp_min : float
        Minimum diameter in meters for which particle number is to be calculated.
    Dp_max : float
        Maximum diameter in meters for which particle number is to be calculated.

    Returns
    -------
    xr.DataArray
        Array-like variable containing the exact number of particles in the specified diameter range.
    """
    assert dc1.name == 'dndlDp', 'you can only calc N on dndlogDp'
    assert Dp_min < Dp_max, 'd1 not < d2'

    dc1_ = dc1.bnn.set_lDp()
    _breaks = infer_interval_breaks(dc1_['lDp'])
    # print(_breaks)
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

try:
  import bokeh.models
  from bokeh.models import LogColorMapper, ColorBar, CustomJS
  from bokeh.layouts import Row
  from bokeh.plotting import Figure, output_notebook, show
except: 
  print('warning bokeh not installed')

def bokeh_plot_psd(ds,width = 1000):

    output_notebook()
    cb, p = basic_bokeh_plot(ds,width)
    show(Row(p,cb))
    return p,cb


def basic_bokeh_plot(ds, width=1000):
    ######### EDGES #########
    vmin = 100
    vmax = 10_000
    # get intervals time
    x0, x1 = infer_interval_breaks(ds['time'])[[0, -1]]
    # get interval break. add 9 for nanometers
    y0, y1 = infer_interval_breaks(ds['lDp'])[[0, -1]] + 9
    ######### FIGURE ############
    _w = bokeh.models.tools.WheelZoomTool()
    p = Figure(x_axis_type='datetime', width=width, height=400,
               tools=["pan", "box_zoom", "reset", _w, "crosshair", "hover"],
               y_axis_type="log")
    p.toolbar.active_scroll = _w
    color_mapper = LogColorMapper(palette="Plasma256", low=vmin, high=vmax)
    p.image([ds.transpose('lDp', 'time').data],
            x=x0, dw=x1 - x0, y=10 ** y0, dh=10 ** y1 - 10 ** y0,
            color_mapper=color_mapper)
    ######### COLOR BAR #############
    cb = get_colorbar(color_mapper, p, vmax, vmin)
    return cb, p


def get_colorbar(color_mapper, p, vmax, vmin):
    cb = Figure(width=100, y_axis_type="log", height=p.height)
    cb.dot(0, 0)
    cb.x_range.start = 0
    cb.x_range.end = 1
    cb.x_range.bounds = 'auto'
    cb.x_range.min_interval = 1
    # cb.y_range.bounds=[10,100]
    cb.xaxis.visible = False
    cb.min_border_left = 60
    cb.toolbar.active_scroll = cb.tools[1]
    cb.toolbar_location = None
    cb.y_range.start = vmin
    cb.y_range.end = vmax
    cbar = ColorBar(color_mapper=color_mapper, padding=0, margin=0)
    callback = CustomJS(args=dict(color_mapper=color_mapper, y_range=cb.y_range), code="""
        color_mapper.low = y_range.start
        color_mapper.high = y_range.end
        color_mapper.change.emit();
    """)
    cb.y_range.js_on_change('start', callback)
    cb.add_layout(cbar, 'right')
    cbar.major_label_text_font_size = "0.1px"
    return cb


def remote_jupyter_proxy_url(port):
    """
    Callable to configure Bokeh's show method when a proxy must be
    configured.

    If port is None we're asking about the URL
    for the origin header.
    """
    # asdf
    import os
    import urllib

    # base_url = os.environ['EXTERNAL_URL']
    base_url = "http://128.214.253.41"
    host = urllib.parse.urlparse(base_url).netloc

    # If port is None we're asking for the URL origin
    # so return the public hostname.
    if port is None:
        return host

    service_url_path = os.environ["JUPYTERHUB_SERVICE_PREFIX"]
    proxy_url_path = "proxy/%d" % port

    user_url = urllib.parse.urljoin(base_url, service_url_path)
    full_url = urllib.parse.urljoin(user_url, proxy_url_path)
    return full_url
