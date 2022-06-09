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

# %% [markdown]
# # imports

# %%

# %%
# load and autoreload
from IPython import get_ipython


# this automatically updates changes in other file's funcionts. 
# it is very handy for data exploration
try:
    _magic = get_ipython().run_line_magic
    _magic('load_ext', 'autoreload')
    _magic('autoreload', '2')
except:
    pass


# import most used packages
# import os
# import glob
# import sys
# import pprint
# import datetime as dt
# import pandas as pd
import numpy as np
# import matplotlib as mpl
# import matplotlib.colors
import matplotlib.pyplot as plt 
# import xarray as xr
# import seaborn as sns
# import cartopy as crt

# %%
# import bnn_tools

# %%
# import customs functions

import bnn_tools.funs as fu
import bnn_tools.bnn_array

# %% [markdown]
# # open sum nais file 

# %%

# %%
p = '../example_data/NAISn20220515np.sum'
# dataset (xarray) of the sum data. 
ds = fu.open_sum2ds(p)

# %%
ds.bnn.plot_psd()

# %%
p = '../example_data/NAISn20220515nds.sum'

# dataset (xarray) of the sum data. 
ds_ineg = fu.open_sum2ds(p)

ds_ineg.bnn.plot_psd(vmax=1e4)

# %%
s1 = dict(a=1,b=2)

s2 = dict(a=3,c=4)

# %%
s1.update(s2)
s1

# %% [markdown]
# # resample time in a file

# %%
ds1 = ds.bnn.resample_ts(600)

ds1.bnn.plot_psd()

# %%
ds1 = ds.bnn.resample_ts(3600)

ds1.bnn.plot_psd()

# %% [markdown]
# # regridding

# %%
ds3 = ds.bnn.dp_regrid(10,.2)
ds3.bnn.plot_psd()

# %% [markdown]
# # get N from psd

# %%
N, d1,d2 = ds.bnn.get_N(0,1)

# %%
# it also gives you the actual limits for the particle size. 
d1,d2

# %%
N.plot()

# %%

# %% [markdown]
# # open psm file 
#
# you need to check with magda the specifications of the psm inversion file 

# %%
pp = '../example_data/izanapsm2022_inv_kernel_20220515.dat'

# %%
ds_psm = fu.open_psm2ds(pp)

# %%
ds_psm.bnn.plot_psd()

# %%
ds_psm.bnn.upsample_ts(300).bnn.plot_psd()

# %%

# %%
# be careful when upsampling or downsampling. here an error
ds_psm.bnn.upsample_ts(900).bnn.plot_psd()

# %% [markdown]
# # open smsp 

# %%

# %%
p3 = '../example_data/202205101400_SMPS.txt'

# %%
ds_s = fu.open_smps2ds(p3).loc[{'time':'2022-05-15'}]

# %%
ds_s.bnn.plot_psd()

# %%

# %%

# %%

# %% [markdown]
# # combine two along DP

# %%
ts = 300
dpr = .05
Dp_cut = 3e-9
Dp_cut1 = 20e-9


dsn1 = ds.bnn.resample_ts(ts)[['dndlDp']]

dsn2 = dsn1.bnn.dp_regrid(10,dpr)

dsp1 = ds_psm.bnn.upsample_ts(ts).bnn.set_Dp()

dsp2 = dsp1.bnn.dp_regrid(10,dpr)


dss1 = ds_s.bnn.upsample_ts(ts)[['dndlDp']]


dss2 = dss1.bnn.dp_regrid(10,dpr)

# %%
dss2

# %%

# %%
dss2.bnn.plot_psd()

# %%
dc = fu.combine_2_spectras(dsp2,dsn2,cut_dim='lDp',cut_point=np.log10(Dp_cut))

dc1 = fu.combine_2_spectras(dc,dss2,cut_dim='lDp',cut_point=np.log10(Dp_cut1))

# %%
dsp2

# %%

# %%

# %%

# %%
dc.bnn.plot_psd(vmax=1e5)
f = plt.gcf()
f.set_figwidth(20)

# %%
ds_ineg.bnn.plot_psd(vmax=1e5)
f = plt.gcf()
f.set_figwidth(20)

# %% [markdown]
# # combine two files along Time 

# %% [markdown]
# # filters

# %% [markdown]
# # plot

# %%
