# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python [conda env:q5]
#     language: python
#     name: conda-env-q5-py
# ---

# %%
# load and autoreload
from IPython import get_ipython

# noinspection PyBroadException
try:
    _magic = get_ipython().magic
    _magic("load_ext autoreload")
    _magic("autoreload 2")
except:
    pass


import datetime as dt
import glob
import os
import pprint
import sys

import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

# %%
import bnn_tools.basic_funs as bfu
import bnn_tools.bnn_array
import bnn_tools.coag_sink as cs
import bnn_tools.funs as fu

# %%
#here im using temp and pres as a constant although it is much better to 
#use the timeseries
T = 273
P = 532_000

# %%
p = "/Users/aliaga/Documents/Work_DA/Work_inar/paper-2/small_data/v2_tratead_psd_data_smps_nais_chc_alto/v2_tratead_psd_data_smps_nais_chc_alto.nc"

# %%
ds = xr.open_dataarray(p)

# %%
ds["id"].to_series()

# %%
ds["loc"].to_series()

# %%
ds

# %%
_id = "nais_ion_neutral_smps"

# %%
_lo = ["alto", "chc"]

# %%
_d = "2018-05-28"

# %%
d = ds.loc[{"id": _id, "loc": _lo, "time": _d}].reset_coords(drop=True)

# %%
d.bnn.plot_psd(row="loc")

# %%
d.bnn.set_Dp().median("time").plot(xscale="log", row="loc")
ax = plt.gca()
ax.grid()

# %%
dNcm, d1, d2 = d.bnn.get_dN(0, 1)
dNm = dNcm * 1e6  # transform from cm2 to m3

# %%
bCS = cs.calc_bin_CS(dN_m3=dNm, T=T, P=P)

# %%
bCS.plot(
    x="time",
    yscale="log",
    robust=False,
    row="loc"
    #          norm=mpl.colors.LogNorm(vmin=1e-5),
)

# %%
dNcm.plot(x="time", yscale="log", robust=True, norm=mpl.colors.LogNorm(), row="loc")

# %%

# %%
CS = cs.calc_CS(dN_m3=dNm, T=T, P=P)

# %%
CS.plot(row="loc")

# %%
f, axs = plt.subplots(3, 2, constrained_layout=True, sharex=True, dpi=200)

# %%
axs[]

# %%
f, axss = plt.subplots(
    3, 2, constrained_layout=True, sharex=True, dpi=200, figsize=(12, 8)
)
cy = (0, 0.12)
cm, cM = 0, 0.007
nm, nM = 1e1,1e4
for i, loc in enumerate(["chc", "alto"]):
    axs = axss[:, i]

    dNcm.loc[{"loc": loc}].plot(
        x="time", yscale="log", robust=False, norm=mpl.colors.LogNorm(vmin=nm,vmax=nM), ax=axs[2]
    )
    CS.loc[{"loc": loc}].plot(ax=axs[0], ylim=cy)

    bCS.loc[{"loc": loc}].plot(
        x="time",
        yscale="log",
        robust=False,
        #          norm=mpl.colors.LogNorm(vmin=1e-5),
        ax=axs[1],
        vmin = cm,
        vmax = cM
    )

    for ax in axs:
        ax.grid(which="major", alpha=0.5)
        bfu.format_ticks(ax)

# %%

# %%
