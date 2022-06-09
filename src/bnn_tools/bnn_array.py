"""
add accessors to xarray so that conversions are easier
"""
import xarray as xr
import numpy as np
from xarray.plot.plot import _infer_interval_breaks as infer_interval_breaks
# import matplotlib as mpl
# import matplotlib.colors
# import matplotlib.pyplot as plt
import pandas as pd

import bnn_tools.basic_funs as bfu

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




@xr.register_dataset_accessor("bnn")
@xr.register_dataarray_accessor("bnn")
class BNN:



    def __init__(self, xarray_obj):
        self._obj = xarray_obj
    #         self._center = None

    #     @property


    def dp_regrid(self, n_subs, log_dy):
        o = self._obj
        o1 = bfu.dp_regrid(o,n_subs = n_subs,log_dy=log_dy)
        return o1


    def from_time2sec(self):
        """change time to sec"""
        return bfu.from_time2sec(self._obj)

    def from_sec2time(self):
        return bfu.from_sec2time(self._obj)

    def from_Dp2lDp(self):
        return bfu.from_Dp2lDp(self._obj)

    def from_lDp2Dp(self):
        return bfu.from_lDp2Dp(self._obj)

    def from_lDp2dlDp(self):
        return bfu.from_lDp2dlDp(self._obj)

    def from_Dp2dDp(self):
        return bfu.from_Dp2dDp(self._obj)

    def set_time(self):
        return  bfu.set_time(self._obj)

    def set_Dp(self):

        return bfu.set_Dp(self._obj)

    def set_lDp(self):

        return bfu.set_lDp(self._obj)

    def set_sec(self):
        return bfu.set_sec(self._obj)

    def plot_psd(self, **kwargs):

        return bfu.plot_psd(self._obj, **kwargs)


    def get_dN(self, d1, d2):
        return bfu.get_dN(self._obj,d1,d2)


    def get_N(self, d1, d2):
        return bfu.get_N(self._obj,d1,d2)

    def resample_ts(self,dt):
        return bfu.resample_ts(self._obj, dt)

    def upsample_ts(self, dt):

        return bfu.upsample_ts(self._obj, dt)

    def u(self,u):
        o:xr.DataArray = self._obj
        o.attrs.update({'units':u})

    def ln(self,ln):
        o:xr.DataArray = self._obj
        o.attrs.update({'long_name':ln})




