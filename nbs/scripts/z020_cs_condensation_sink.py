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
import os 
import glob
import sys
import pprint
import datetime as dt
import pandas as pd 
import numpy as np
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt 
import xarray as xr 
import seaborn as sns
import cartopy as crt

# %%
import bnn_tools

# %%
# import customs functions

import bnn_tools.funs as fu
import bnn_tools.bnn_array
import bnn_tools.coag_sink as cs
import bnn_tools.basic_funs as bfu

# %% [markdown] tags=[]
# # open sum nais file 

# %%
_p = os.path.dirname(
    os.path.dirname(bnn_tools.__path__[0])
)

_p


# %%
p = os.path.join(_p,'example_data/NAISn20220515np.sum')
# dataset (xarray) of the sum data. 
ds = fu.open_sum2ds(p)

# %%
pp = os.path.join(_p,'example_data/izanapsm2022_inv_kernel_20220515.dat')

# %%
ds_psm = fu.open_psm2ds(pp)

# %%
ds_psm.bnn.plot_psd()

# %% [markdown] tags=[]
# # open smsp 

# %%

# %%
p3 = os.path.join(_p,'example_data/202205101400_SMPS.txt')

# %%
ds_s = fu.open_smps2ds(p3).loc[{'time':'2022-05-15'}]

# %%
ds_s.bnn.plot_psd()

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
dc = fu.combine_2_spectras(dsp2,dsn2,cut_dim='lDp',cut_point=np.log10(Dp_cut))

dc1 = fu.combine_2_spectras(dc,dss2,cut_dim='lDp',cut_point=np.log10(Dp_cut1))

# %%

# %%
dc1.bnn.plot_psd(vmax=1e5)
f = plt.gcf()
f.set_figwidth(20)

# %%
# dc in m3 
dcM = dc1 * 1e6

# %%
T = 270 
P = 100000


# %%
def _plt():
    dcM1 = dcM.bnn.set_Dp()
    cs_ = cs.calc_CS(dcM1,T,P)
    cs_['dndlDp'].bnn.set_time().plot()
    f = plt.gcf()
    f.set_figwidth(20)
    bfu.format_ticks(plt.gca())
_plt()

# %% [markdown]
# try again but now with particles from 10 to 1000nm

# %%
dcM1 = dcM.bnn.set_Dp().loc[{'Dp':slice(d1,d2)}]

# %%

# %%

# %%

dN_m3


# %%
def _plt1(d1_,d2_):
    
    dN, d1,d2 = dcM.bnn.get_dN(d1_,d2_)
    dN_m3 = dN 
    

    cs_ = cs.calc_CS(dN_m3=dN_m3,P=P,T=T)
    cs_.bnn.set_time().plot(label=f'{d1*1e9:.1f}nm,   {d2*1e9:.1f}nm')
    ax = plt.gca()
    ax.legend()
    bfu.format_ticks(ax)
    f = plt.gcf()
    f.set_figwidth(20)

_plt1(1e-9,1000e-9)
_plt1(10e-9,1000e-9)
_plt1(50e-9,1000e-9)
_plt1(100e-9,1000e-9)


# %%
xr.DataArray.

# %%
dcM1,d1,d2 = dcM.bnn.set_Dp()['dndlDp'].bnn.get_dN(0,1)

# %%
dens1 = dens2 = 1200 
alpha = 1 
dN = dcM1
d1 = 1e-9
d2 = 2e-9

# %%
_r = cs.calc_coag_snk_xr(dN_tot_m3 = dN , d1 = d1, d2 = d2 , P = P, T = T , alpha = alpha, dens1 = dens1, dens2 = dens2)

# %%
__r = _r['CoagSnk'].bnn.set_time()/1e6



__r.bnn.set_time().plot()
bfu.format_ticks(plt.gca())

# %%
_r['CoagSnk']

# %%
