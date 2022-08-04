# -*- coding: utf-8 -*-
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
    _magic("load_ext", "autoreload")
    _magic("autoreload", "2")
except:
    pass


import datetime as dt
import glob

# import most used packages
import os
import pprint
import sys

import cartopy as crt
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

# %%
import bnn_tools

# %%
# import customs functions

import bnn_tools.basic_funs as bfu
import bnn_tools.bnn_array
import bnn_tools.coag_sink as cs
import bnn_tools.funs as fu


# %% [markdown]
#

# %% [markdown]
# ## mean free path

# %%
def plt_mean_free_path():
    T_ = xr.DataArray(np.linspace(250, 310, 100), dims="T", attrs={"units": "K"})
    P_ = xr.DataArray(
        np.linspace(50_000, 120_000, 100), dims="P", attrs={"units": "Pa"}
    )

    Tb, Pb = xr.broadcast(T_.assign_coords({"T": T_}), P_.assign_coords({"P": P_}))
    mfp = cs.calc_MFP(T=Tb, P=Pb)
    mfp.name = "mean free path"

    mfp.bnn.u("m")
    #     global res
    #     global ax
    res = mfp.plot.contour(levels=20, add_colorbar=True)
    ax = plt.gca()
    ax.clabel(res, inline=1, fmt="%1.2e", fontsize=12)  # label every second level
    ax.grid(which="both")


plt_mean_free_path()


# %% [markdown]
# ## viscocity 

# %%
def _plot_vis():
    T_ = xr.DataArray(np.linspace(250, 310, 100), dims="T", attrs={"units": "K"})
    T_ = T_.assign_coords({"T": T_})

    _vis = cs.calc_viscosity(T_)
    _vis.name = "viscosity"
    _vis.bnn.u("Pa*s")

    _vis.plot()


_plot_vis()


# %% [markdown]
# ## calc_coag_coef_fuchs

# %%
def _plt_coag_coef():
    d1_ = xr.DataArray(np.geomspace(1e-9, 1e-5), dims="d1", attrs={"units": "m"})
    d2_ = xr.DataArray(np.geomspace(1e-9, 1e-5), dims="d2", attrs={"units": "m"})
    d1, d2 = xr.broadcast(
        d1_.assign_coords({"d1": d1_}), d2_.assign_coords({"d2": d2_})
    )
    alpha = 1
    dens1 = dens2 = 1_000
    T = 300
    P = 101_000
    ccf = cs.calc_coag_coef_fuchs(d1, dens1, d2, dens2, T, P, alpha)
    ccf.name = r"$\beta$"
    ccf.bnn.u("m3 / s")
    ccf.plot(norm=mpl.colors.LogNorm(), yscale="log", xscale="log")
    return ccf


ccf = _plt_coag_coef()


# %%
def _plt_pandis(ccf):
    cm = ccf * 1e6
    d2_ = xr.DataArray(
        np.geomspace(1e-9, 1e-5), dims="d2_", coords={"d2_": np.geomspace(1e-9, 1e-5)}
    )
    d1_ = xr.DataArray(
        np.geomspace(1e-9, 1e-5), dims="d2_", coords={"d2_": np.geomspace(1e-9, 1e-5)}
    )
    cm_ = cm.interp({"d2": d2_, "d1": d1_}).reset_coords(drop=True)
    cmap = plt.get_cmap("plasma")

    for i in (np.array([1, 10, 100, 1_000, 10_000]) / 1e9)[::-1]:
        cm.interp({"d1": i}).where(cm["d2"] <= i).plot(
            yscale="log",
            xscale="log",
            label=f"{i * 1e9:1.0f}nm",
            color=cmap(
                (np.log10(i) - np.log10(1e-9)) / np.log10(10_000),
            ),
        )

    cm_.plot(c="k", label="one-to-one", zorder=100)
    ax = plt.gca()
    ax.legend(loc=1, bbox_to_anchor=(-0.2, 1))
    ax.grid(which="both", lw=0.2, c=".8")
    ax.set_title("reproduce seinfeld and pandis fig 13.5")
    f = plt.gcf()
    f.set_figheight(8)


_plt_pandis(ccf)

# %% [markdown]
# - it seems quite good 
#     - however for 10 um we are overestimated—doubling

# %%

# %%

# %% [markdown]
# # open sum nais file 

# %%
_p = os.path.dirname(os.path.dirname(bnn_tools.__path__[0]))

_p


# %%
p = os.path.join(_p, "example_data/NAISn20220515np.sum")
# dataset (xarray) of the sum data.
ds = fu.open_sum2ds(p)

# %%
pp = os.path.join(_p, "example_data/izanapsm2022_inv_kernel_20220515.dat")

# %%
ds_psm = fu.open_psm2ds(pp)

# %%
ds_psm.bnn.plot_psd()

# %% [markdown]
# # open smsp 

# %%

# %%
p3 = os.path.join(_p, "example_data/202205101400_SMPS.txt")

# %%
ds_s = fu.open_smps2ds(p3).loc[{"time": "2022-05-15"}]

# %%
ds_s.bnn.plot_psd(vmax=1e5)

# %%
ds_s.bnn.plot_psd(vmax=1e5, levels=9)

# %%
ds_s.resample({"time": "10.01T"}).median().bnn.plot_psd(vmax=1e5, levels=9)

# %%
_ds_

# %%
_ds_s = ds_s.bnn.set_lDp()
for i in range(5):
    _ds_s = _ds_s.rolling({"time": 3}, center=True, min_periods=1).mean()
    _ds_s = _ds_s.rolling({"lDp": 3}, center=True, min_periods=1).mean()

# %%
_ds_s.bnn.plot_psd(vmax=1e5, levels=17)

# %% [markdown]
# # combine two along DP

# %%
N, _, _ = ds_s.bnn.get_N(d1=10e-9, d2=1)

# %%
N1 = ds_s["dndlDp"].bnn.get_exact_N(15e-9, 500e-9)

# %%
N.plot()
N1.plot()

# %%
ts = 300
dpr = 0.05
Dp_cut = 3e-9
Dp_cut1 = 20e-9


dsn1 = ds.bnn.resample_ts(ts)[["dndlDp"]]

dsn2 = dsn1.bnn.dp_regrid(10, dpr)

dsp1 = ds_psm.bnn.upsample_ts(ts).bnn.set_Dp()

dsp2 = dsp1.bnn.dp_regrid(10, dpr)


dss1 = ds_s.bnn.upsample_ts(ts)[["dndlDp"]]


dss2 = dss1.bnn.dp_regrid(10, dpr)

# %%
dc = fu.combine_2_spectras(dsp2, dsn2, cut_dim="lDp", cut_point=np.log10(Dp_cut))

dc1 = fu.combine_2_spectras(dc, dss2, cut_dim="lDp", cut_point=np.log10(Dp_cut1))

# %%
dc1.bnn.set_time().bnn.dp_regrid(10, 0.3).bnn.set_Dp()["dndlDp"].plot(hue="Dp")
f = plt.gcf()
f.set_figwidth(20)
ax = plt.gca()
bfu.format_ticks(ax)

# %%
dc1.bnn.plot_psd(vmax=1e5)
f = plt.gcf()
f.set_figwidth(20)

# %%
xr.DataArray.rolling()

# %%
dc1_ = dc1
for i in range(5):
    dc1_ = dc1_.rolling({"secs": 3}, center=True, min_periods=1).mean()

# %%
dc1["dndlDp"].bnn.get_exact_N(10e-9, 500e-9).bnn.set_time().plot(
    marker=".", lw=0, c=".8"
)
dc1["dndlDp"].bnn.get_exact_N(3e-9, 500e-9).bnn.set_time().plot(
    marker=".", lw=0, c=".8"
)
dc1["dndlDp"].bnn.get_exact_N(10e-9, 20e-9).bnn.set_time().plot(
    marker=".", lw=0, c=".8"
)
dc1["dndlDp"].bnn.get_exact_N(3e-9, 7e-9).bnn.set_time().plot(marker=".", lw=0, c=".8")

dc1_["dndlDp"].bnn.get_exact_N(10e-9, 500e-9).bnn.set_time().plot(label="10-500nm")
dc1_["dndlDp"].bnn.get_exact_N(3e-9, 500e-9).bnn.set_time().plot(label="3-500nm")
dc1_["dndlDp"].bnn.get_exact_N(10e-9, 20e-9).bnn.set_time().plot(label="10-20nm")
dc1_["dndlDp"].bnn.get_exact_N(3e-9, 7e-9).bnn.set_time().plot(label="3-7nm")


bfu.format_ticks(plt.gca())
plt.gca().legend()
plt.gcf().set_figwidth(20)
# plt.gca().set_xlim('2022-05-15 09','2022-05-15 12')

# %%
dc1["dndlDp"].bnn.get_exact_N(3e-9, 500e-9).bnn.set_time().plot()
bfu.format_ticks(plt.gca())

# %%
dc1["dndlDp"].bnn.get_exact_N(3e-9, 7e-9).bnn.set_time().plot()
bfu.format_ticks(plt.gca())

# %%
_l = np.round(np.geomspace(3, 500, 8)).astype(int)


# %%
_tups = [np.array([_l[i_], _l[i_ + 1]]) for i_ in range(len(_l) - 1)]

# %%
dc1.bnn.set_Dp()["Dp"]

# %%
t_

# %%
cm = plt.get_cmap("plasma")

# %%

# %%
from matplotlib.lines import Line2D

print(Line2D.markers.keys())

# %%
m_ = ["^", "8", "s", "p", "*", "h", "H"]
for i, t_ in enumerate(_tups):
    _d = dc1["dndlDp"].bnn.get_exact_N(*(t_ * 1e-9))
    _d1 = _d.rolling({"secs": 15}, center=True).mean()
    (_d / _d1.max()).bnn.set_time().plot(
        x="time", label="", c=cm(i / len(_tups)), lw=0, marker=m_[i], alpha=0.2, ms=3
    )

    (_d1 / _d1.max()).bnn.set_time().plot(
        x="time",
        label=f"$Dp_{{{t_[0]}-{t_[1]}}}$",
        c=cm(i / len(_tups)),
        lw=2,
        marker=m_[i],
    )

ax = plt.gca()
ax.legend()
bfu.format_ticks(ax)
f = plt.gcf()
f.set_figwidth(20)
# ax.set_yscale('log')

ax.set_ylim(1e-2, 1.1)
ax.grid(which="both", c=".9")
ax.set_title("Normalized N");

# %%

# %%

# %%
d1 = 7e-9
d2 = 20e-9

# %%
dc1["dndlDp"].bnn.get_exact_N(d1, d2).bnn.set_time().plot()
bfu.format_ticks(plt.gca())
ax = plt.gca()
ax.grid(which="both", alpha=0.2)

# %%
# dc in m3
dcM = dc1 * 1e6

# %%
T = 270
P = 100000


# %%

# %%
def _plt1(d1_, d2_):

    dN, d1, d2 = dcM.bnn.get_dN(d1_, d2_)
    dN_m3 = dN

    cs_ = cs.calc_CS(dN_m3=dN_m3, P=P, T=T)
    cs_.bnn.set_time().plot(label=f"{d1*1e9:.1f}nm,   {d2*1e9:.1f}nm")
    ax = plt.gca()
    ax.legend()
    bfu.format_ticks(ax)
    f = plt.gcf()
    f.set_figwidth(20)


_plt1(1e-9, 1000e-9)
_plt1(10e-9, 1000e-9)
_plt1(50e-9, 1000e-9)
_plt1(100e-9, 1000e-9)


# %%

# %%

# %%
def _plt_coag_snk():
    dN_m3, d1, d2 = dcM.bnn.set_Dp()["dndlDp"].bnn.get_dN(0, 1)
    dens1 = dens2 = 1200
    alpha = 1
    d1 = 3e-9
    d2 = 10000e-9

    _r = cs.calc_coag_snk_xr(
        dN_tot_m3=dN_m3, d1=d1, d2=d2, P=P, T=T, alpha=alpha, dens1=dens1, dens2=dens2
    )

    __r = _r["CoagS"].bnn.set_time()
    __r.bnn.set_time().plot(
        norm=mpl.colors.LogNorm(vmin=1e-5, vmax=1e-3), yscale="log", robust=True
    )
    bfu.format_ticks(plt.gca())


_plt_coag_snk()

    # %%
    dN_m3,d1,d2 = dcM.bnn.set_Dp()['dndlDp'].bnn.get_dN(0,1)
    dens1 = dens2 = 1200 
    alpha = 1 
    d1 = 3e-9
    d2 = 4e-9

    _r = cs.calc_coag_snk_xr(dN_tot_m3 = dN_m3 , d1 = d1, d2 = d2 , P = P, T = T , alpha = alpha, dens1 = dens1, dens2 = dens2)

    __r = _r['CoagS'].bnn.set_time()
    __r.bnn.set_time().plot(norm=mpl.colors.LogNorm(vmin=1e-5,vmax = 1e-3),yscale='log', robust = True)
    bfu.format_ticks(plt.gca())

# %%

# %%
r1 = _r["CoagSnk"] * 1e-6
r1.bnn.u(r"cm$^3$ s$^{-1}$")
r1.bnn.set_time().plot()
ax = plt.gca()
ax.grid(which="both", color=".9")
plt.gcf().set_figwidth(20)
bfu.format_ticks(ax)


# %%

# %%
def method_name():
    f, axs = plt.subplot_mosaic("abc", figsize=(3 * 5, 3))

    dN_m3, d1, d2 = dcM.bnn.set_Dp()["dndlDp"].bnn.get_dN(0, 1)
    dens1 = dens2 = 1200
    alpha = 1
    d1 = 3e-9
    d2 = 1000e-9

    _Dp12 = dN_m3["Dp"].loc[{"Dp": slice(d1, d2)}]
    Dp12 = xr.DataArray(
        _Dp12.values, dims="Dp_d1_d2", coords={"Dp_d1_d2": _Dp12.values}
    )

    Dp_, Dp12_ = xr.broadcast(dN_m3["Dp"], Dp12)

    def _calc_beta(Dp12_, Dp_, alpha, axs, dens1, dens2):
        beta = cs.calc_coag_coef_fuchs(
            d1=Dp12_, dens1=dens1, d2=Dp_, dens2=dens2, T=T, P=P, alpha=alpha
        )
        b1 = beta.where(beta["Dp"] > beta["Dp_d1_d2"])
        beta.plot(yscale="log", xscale="log", norm=mpl.colors.LogNorm(), ax=axs["a"])
        b1.plot(yscale="log", xscale="log", norm=mpl.colors.LogNorm(), ax=axs["b"])
        ax = axs["c"]
        b1.sum("Dp").plot(label="b1", ax=ax)
        beta.sum("Dp").plot(label="beta", ax=ax)
        ax = plt.gca()
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.legend()
        axs["a"].set_title("beta")
        axs["b"].set_title("b1")
        f.tight_layout()

    _calc_beta(Dp12_, Dp_, alpha, axs, dens1, dens2)


method_name()

# %%
(dN_m3 + 0.00001).bnn.set_time().plot(norm=mpl.colors.LogNorm(vmin=1e7), yscale="log")

# %%
