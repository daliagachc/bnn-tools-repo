"""
add accessors to xarray so that conversions are easier
"""
import xarray as xr
import numpy as np
from xarray.plot.plot import _infer_interval_breaks as infer_interval_breaks
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
import pandas as pd

# @xr.register_dataset_accessor("geo")
# class GeoAccessor:
#     def __init__(self, xarray_obj):
#         self._obj = xarray_obj
#         self._center = None
#
#     @property
#     def center(self):
#         """Return the geographic center point of this dataset."""
#         if self._center is None:
#             # we can use a cache on our accessor objects, because accessors
#             # themselves are cached on instances that access them.
#             lon = self._obj.latitude
#             lat = self._obj.longitude
#             self._center = (float(lon.mean()), float(lat.mean()))
#         return self._center
#
#     def plot(self):
#         """Plot data on a map."""
#         return "plotting!"



def _dp_regrid_old(*, da, n_subs, log_dy):

    darray = da.bnn.set_lDp()
    dy = log_dy / n_subs
    dm = np.ceil(darray['lDp'].min().item() / log_dy) * log_dy
    dM = np.ceil(darray['lDp'].max().item() / log_dy) * log_dy

    dms = np.arange(dm - (((n_subs - 1) / 2) * dy), dM, dy)

    d1 = darray.interp({'lDp': dms})

    dout = d1.coarsen(**{'lDp': n_subs}, boundary='trim').mean().reset_coords(drop=True)

    # dout['time'] = dout['secs'].astype('datetime64[s]')

    # dout['Dp'] = 10 ** dout['lDp']

    return dout


def _dp_regrid(*, da, n_subs, log_dy):

    ds1 = da.bnn.set_lDp().reset_coords(drop=True)

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




@xr.register_dataset_accessor("bnn")
@xr.register_dataarray_accessor("bnn")
class BNN:



    def __init__(self, xarray_obj):
        self._obj = xarray_obj
    #         self._center = None

    #     @property


    def dp_regrid(self, n_subs, log_dy):
        o = self._obj
        o1 = _dp_regrid(da = o,n_subs = n_subs,log_dy=log_dy)
        return o1


    def from_time2sec(self,col='time'):
        """change time to sec"""
        date = self._obj[col]
        s1 = date - np.datetime64(0, 'Y')
        s2 = s1 / np.timedelta64(1, 's')
        return self._obj.assign_coords({'secs':s2})

    def from_sec2time(self,col='secs'):
        o =self._obj
        secs = o[col].astype('datetime64[s]')
        return o.assign_coords({'time':secs})

    def from_Dp2lDp(self,col='Dp'):
        o =self._obj
        lDp =  np.log10(o[col])
        return o.assign_coords({'lDp':lDp})

    def from_lDp2Dp(self):
        o =self._obj
        Dp =  10**(o['lDp'])
        return o.assign_coords({'Dp':Dp})

    def from_lDp2dlDp(self):
        o =self._obj
        o = o.bnn.set_lDp()
        lDp =  o['lDp']
        borders  = infer_interval_breaks(lDp)
        d = borders[1:] - borders[:-1]
        d1 = lDp*0 + d
        return o.assign_coords({'dlDp':d1})

    def from_Dp2dDp(self):
        o =self._obj
        Dp =  o['Dp']
        borders  = infer_interval_breaks(Dp)
        d = borders[1:] - borders[:-1]
        d1 = Dp*0 + d
        return o.assign_coords({'dDp':d1})


    def set_time(self):
        o = self._obj
        dims = list(o.dims)
        coords = list(o.coords)
        # o_coords = set(coords)-set(dims)


        if 'time' not in coords:
            o = o.bnn.from_sec2time()

        if 'time' not in dims:

            o=o.swap_dims( { 'secs':'time'})

        return o

    def set_Dp(self):

        o = self._obj
        dims = list(o.dims)
        coords = list(o.coords)
        # o_coords = set(coords)-set(dims)


        if 'Dp' not in coords:
            # print(coords)
            o = o.bnn.from_lDp2Dp()

        if 'Dp' not in dims:

            o=o.swap_dims( { 'lDp':'Dp'})

        return o

    def set_lDp(self):
        # o1 = self._obj
        # o1 = self.from_sec2time()
        # o2 = o1.swap_dims( { 'Dp':'lDp'})



        o = self._obj
        dims = list(o.dims)
        coords = list(o.coords)
        # o_coords = set(coords)-set(dims)


        if 'lDp' not in coords:
            o = o.bnn.from_Dp2lDp()

        if 'lDp' not in dims:

            o=o.swap_dims( { 'Dp':'lDp'})

        return o



    def set_sec(self):
        o = self._obj
        dims = list(o.dims)
        coords = list(o.coords)
        # o_coords = set(coords)-set(dims)


        if 'secs' not in coords:
            o = o.bnn.from_time2sec()

        if 'secs' not in dims:

            o=o.swap_dims( { 'time':'secs'})

        return o



    def plot_psd(self, **kwargs):
        import matplotlib.dates as mdates


        o = self._obj
        if isinstance(o,xr.Dataset):
            o = o['dndlDp']

        o = o.bnn.set_time()
        o = o.bnn.set_Dp()

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
            yscale = 'log',
        )


        s1.update(kwargs)



        o.plot(
            **s1
        )

        ax = plt.gca()

        locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)


        locator = mdates.AutoDateLocator(minticks=60, maxticks=60)
        ax.xaxis.set_minor_locator(locator)
        # ax.xaxis.set_minor_formatter(formatter)

        ax.grid(c='w',ls='--',alpha=.5)
        ax.grid(c='w',ls='-',which='minor',lw=.5,alpha=.3)

        for xlabels in ax.get_xticklabels():
            xlabels.set_rotation(0)
            xlabels.set_ha("center")

        plt.gcf().set_figwidth(10)

    def get_dN(self, d1, d2):
        o = self._obj

        if isinstance(o,xr.Dataset):
            o = o['dndlDp']
        assert o.name == 'dndlDp'
        o = o.bnn.from_lDp2dlDp()
        o = o.bnn.set_Dp()
        o1 = o.loc[{'Dp':slice(d1,d2)}]
        dmin = o1['Dp'].min().item()
        dmax = o1['Dp'].max().item()

        o1['dN'] = (o1 * o1['dlDp'])

        dN = o1['dN']

        return dN, dmin, dmax

    def get_N(self, d1, d2):
        o = self._obj

        dN, dmin,dmax = o.bnn.get_dN(d1,d2)

        N = dN.sum('Dp')
        N.name = 'N'
        return N, dmin, dmax




    def resample_ts(self,dt):
        o = self._obj
        orig_dim = list(o.dims)



        orgi_dt = o.bnn.set_sec()['secs'].diff('secs').mean().item()


        assert dt>=orgi_dt, 'you are trying to downsample when you should be upsampling'




        o = o.bnn.set_sec()
        ddt = np.round(o['secs']/dt)*dt
        o = o.groupby(ddt).mean()

        if 'time' in orig_dim:
            o = o.bnn.set_time()
        if 'secs' in orig_dim:
            o = o.bnn.set_sec()

        return o

    def upsample_ts(self, dt):

        o = self._obj

        orgi_dt = o.bnn.set_sec()['secs'].diff('secs').mean().item()


        assert dt<=orgi_dt, 'you are trying to upsample when you should be downsampling'

        orig_dim = list(o.dims)


        o = o.bnn.set_time()

        o = o.resample({'time':pd.Timedelta(dt,unit='s')}).interpolate('linear')


        if 'time' in orig_dim:
            o = o.bnn.set_time()
        if 'secs' in orig_dim:
            o = o.bnn.set_sec()

        return o



