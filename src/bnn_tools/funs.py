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
# import matplotlib as mpl
# import matplotlib.colors
# import matplotlib.pyplot as plt
import xarray as xr
# import seaborn as sns
# import cartopy as crt

import pandas as pd
import scipy.interpolate
import bnn_tools.bnn_array

## constants come here

# All the variables are in the SI metric units, e.g., dp in m


EC = 1.602176565e-19  # elementary charge
# Dimensions
DP = 'Dp'
TIME = 'time'

# variables
DNDLDP = 'dndlDp'
CONC = 'conc'
LOG_DP_DIS = 'log_Dp_dis'


def open_sum2ds_old_wrong(path_sum):
    """opens a sum path and created an xarray ds out of it
    - im assuming the dp size from sum is the geomean location of the bin
    2022-07-02_01-17-17_
         - this is a wrong assumption. it is the limits and thats why you have an extra lim
    """

    # open ds
    da = pd.read_csv(path_sum, sep='\s+', header=None)

    # get columns
    time = da.iloc[1:, 0]

    # transforming from matlab datenum to python's timestamps ->
    # https://stackoverflow.com/questions/13965740/converting-matlabs-datenum-format-to-python
    time = pd.to_datetime(time - 719529, unit='D')

    geo_mid_size = da.iloc[0, 2:]
    data = da.iloc[1:, 2:]
    conc = da.iloc[1:, 1]

    # mid_size = [np.sqrt(size.iloc[i]*size.iloc[i+1])
    #             for i in range(0,len(size)-1)]

    log_dis = [np.log10(geo_mid_size.iloc[i + 1]) - np.log10(geo_mid_size.iloc[i])
               for i in range(0, len(geo_mid_size) - 1)]

    log_dis = pd.Series([*log_dis, log_dis[-1]], index=geo_mid_size)
    log_dis.index.name = DP
    data = data.set_index(time)
    data.index.name = TIME
    data = data.set_axis(geo_mid_size, axis=1)
    data.axes[1].name = DP

    conc = conc.set_axis(time)
    conc.index.name = TIME
    d = data.unstack().to_xarray()
    ds = d.to_dataset(name=DNDLDP)
    ds[CONC] = conc

    ds[LOG_DP_DIS] = log_dis
    ds = ds.set_coords(LOG_DP_DIS)

    ds = ds.dropna(dim=DP, how='all')

    return ds


def open_sum2ds(path_sum):
    """opens a sum path and created an xarray ds out of it
    - im assuming the dp size from sum is the geomean location of the bin
    2022-07-02_01-17-17_
         - this is a wrong assumption. it is the limits and thats why you have an extra lim
    """

    # open ds
    da = pd.read_csv(path_sum, sep='\s+', header=None)

    # get columns
    time = da.iloc[1:, 0]

    # transforming from matlab datenum to python's timestamps ->
    # https://stackoverflow.com/questions/13965740/converting-matlabs-datenum-format-to-python
    time = pd.to_datetime(time - 719529, unit='D')

    geo_mid_edge = da.iloc[0, 2:]
    d = log_geo_mid_edge = np.log10(geo_mid_edge)
    log_geo_mid_size = [(d[i]+d[i+1])/2 for i in range(len(d)-1)]
    geo_mid_size = 10**log_geo_mid_size
    data = da.iloc[1:, 2:-1]
    conc = da.iloc[1:, 1]

    # mid_size = [np.sqrt(size.iloc[i]*size.iloc[i+1])
    #             for i in range(0,len(size)-1)]

    log_dis = [np.log10(geo_mid_size.iloc[i + 1]) - np.log10(geo_mid_size.iloc[i])
               for i in range(0, len(geo_mid_size) - 1)]

    log_dis = pd.Series([*log_dis, log_dis[-1]], index=geo_mid_size)
    log_dis.index.name = DP
    data = data.set_index(time)
    data.index.name = TIME
    data = data.set_axis(geo_mid_size, axis=1)
    data.axes[1].name = DP

    conc = conc.set_axis(time)
    conc.index.name = TIME
    d = data.unstack().to_xarray()
    ds = d.to_dataset(name=DNDLDP)
    ds[CONC] = conc

    ds[LOG_DP_DIS] = log_dis
    ds = ds.set_coords(LOG_DP_DIS)

    ds = ds.dropna(dim=DP, how='all')

    return ds

def mob_from_dp_T_P(dp, T, P, n=1):
    mfp = cal_mfp(T=T, P=P)
    vis = cal_vis(T=T)
    mob = dp2Z(dp, mfp, vis, n=n)
    return mob


def dp_from_mob_T_P(mob_in, T, P, n=1):
    """this method uses 3rd order interpolation and therefore is not
    exact but should by a very fast good approx"""

    # in case we are only using one number
    if type(mob_in) in [float, int]:
        return Z2dp(mob_in, cal_mfp(T, P), cal_vis(T), n=n)
    # else lets assume the type of mob_in will have max and min methods
    dp_min = Z2dp(mob_in.max(), cal_mfp(T, P), cal_vis(T), n=n)
    dp_max = Z2dp(mob_in.min(), cal_mfp(T, P), cal_vis(T), n=n)

    # print(dp_min,dp_max)
    dp_range = np.geomspace(0.95 * dp_min, 1.05 * dp_max, 1000)
    mob_range = mob_from_dp_T_P(dp_range, T, P, n=n)
    int_mob_to_dp = scipy.interpolate.interp1d(mob_range, dp_range, kind='cubic')

    dp_out = int_mob_to_dp(mob_in)

    return dp_out


##########
# main functions, see the formulae for mfp and vis below
# convert particle electrical moblity diameter (dp) to electrical mobility (Z)
def dp2Z(dp, mfp, vis, n=1):
    """the usual expression of electrical mobility, fractal dimension is
    assumed to be 1.0 see, e.g., Eq. 15.21, P322, Aerosol Technology,
    Second Edition by William Hinds
    :param dp: diameter particle
    :param mfp: mean free path
    :param vis: viscosity
    :param n:
    :return: electric mobility Z
    """
    Kn = cal_Kn(dp, mfp)
    Cs = cal_Cs(Kn)

    return n * EC * Cs / (3 * np.pi * dp) / vis


def Z2dp(Z, mfp, vis, n=1):
    """convert electrical mobility (Z) to particle electrical mobility diameter (
    dp) according to line 12, dp/Cs(dp) is inversely proportional to Z so some
    iteration is needed to solve dp from dp/Cs(dp), e.g., using a solver.
    This function use a straightforward iteration method since it is not
    computionally expensive
    - diego: i feel this function is prone to errors and is not vectorizable
    """
    dp_Cs = n * EC / (3 * np.pi * vis * Z)
    dp_old = dp_Cs
    while True:  # perhaps dangerous
        Kn = cal_Kn(dp_old, mfp)
        Cs = cal_Cs(Kn)
        dp_new = (dp_Cs * Cs + dp_old) / 2  # avoid oscillations
        if np.max(abs((dp_new - dp_old) / dp_old)) < 1e-3:
            dp_old = dp_new
            break
        else:
            dp_old = dp_new

    return dp_old


##########

# Mean free path of air, Eq. 2-10, P19, Aerosol Measurement, Third Edition
def cal_mfp(T=293.15, P=101325):
    return 66.5 * 10 ** (-9) * (101325 / P) * (T / 293.15) * (
            1 + 110.4 / 293.15) / (1 + 110.4 / T)  # in m


# Dynamic viscosity (rather than kinetic viscosity which is divided by
# density) of air using Sutherland equation, Eq. 2-8, P18, Aerosol
# Measurement, Third Edition
def cal_vis(T=293.15):
    return 18.203 * 10 ** (-6) * (293.15 + 110.4) / (T + 110.4) * (
            T / 293.15) ** 1.5  # Pa*s


# Knudsen number
def cal_Kn(dp, mfp):
    return 2 * mfp / dp


# Cunningham slip correction factor, equation and parameters by Allen & Raabe
# 1985, for solid particles Can also be found in Eq. 2-14, P20,  Aerosol
# Measurement, Third Edition
def cal_Cs(Kn):
    alpha = 1.142
    beta = 0.558
    gamma = 0.999
    return 1 + Kn * (alpha + beta * np.exp(-gamma / Kn))


def timestamp2matlab(timestamp):
    """return matlab time in days from pd.Timestamp"""
    dt = timestamp - pd.Timestamp('1970')
    ml_time = dt / pd.Timedelta(1, 'days') + 719529
    return ml_time


def ds2sum(ds, path_out):
    df = ds['dndlDp'].to_dataframe()
    df1 = df['dndlDp'].unstack('Dp')
    cols = df1.columns
    conc = ds['conc'].to_dataframe()['conc']
    df1['conc'] = conc
    df2 = df1[['conc', *cols]]

    df2 = df2.rename({'conc': 0.}, axis=1)
    df2.index.name = 0.0

    df2.index = timestamp2matlab(df2.index)

    df2.to_csv(path_out, sep='\t', float_format='%.15e')


def open_psm2ds(pp):
    def get_arr(pp):
        d = pd.read_csv(pp, sep='\t')

        mt = d.iloc[:, 0]
        mt.name = 'mt'

        d1 = d.T.sort_index().T

        data = d1.iloc[:, :-1]

        dp = d1.columns.astype(float) * 1e-9

        time = pd.to_datetime(mt - 719529, unit='D')

        da = xr.DataArray(data.values,
                          dims=['time', 'Dp_i'],
                          coords={'time': time}, name='dN'
                          ).to_dataset()

        return dp, da

    def resample(da, f_ldp_i, ldp, ser):
        dldp = ldp[1:] - ldp[:-1]
        da1 = da.assign_coords({'dldp': da['Dp_i'] * 0 + dldp})
        dis = np.arange(ser['ldp'].iloc[0], ser['ldp'].iloc[-1], .005)
        da1['dndlDp'] = da1['dN'] / da1['dldp']
        da2 = da1['dndlDp'].interp({'Dp_i': f_ldp_i(dis)}, method="nearest",
                                   kwargs=dict(fill_value="extrapolate"))
        da2['lDp'] = da2['Dp_i'] * 0 + dis
        da2['Dp'] = da2['Dp_i'] * 0 + 10 ** dis
        da2
        da3 = da2.swap_dims({'Dp_i': 'Dp'})
        da4 = da3.drop('Dp_i').to_dataset()
        return da4

    def get_int(ser):
        from scipy.interpolate import interp1d
        f_ldp_i = interp1d(ser['ldp'], ser['i'], kind='linear')
        # f_ldp_dp = interp1d(ser['ldp'],ser['dp'],kind='linear')
        return f_ldp_i

    def get_series(dp, ldp):
        ser = pd.Series(ldp, name='ldp').reset_index().rename({'index': 'i'}, axis=1)
        ser['i'] = ser['i'] - .5
        ser['dp'] = dp
        return ser

    dp, da = get_arr(pp)
    ldp = np.log10(dp)
    ser = get_series(dp, ldp)
    f_ldp_i = get_int(ser)
    da4 = resample(da, f_ldp_i, ldp, ser)

    da5 = da4.bnn.dp_regrid(10, .05)

    return da5


def is_regular_grid(coor, tolerance_percentage=1):
    from xarray.plot.plot import _infer_interval_breaks as infer_interval_breaks

    tol = tolerance_percentage / 100

    bks = infer_interval_breaks(coor, check_monotonic=True)

    d = pd.Series(bks).diff()

    n = d / np.mean(d)
    m = np.abs(n - 1)

    bo = (m > tol).sum() == 0

    #     print(tol)

    return bo


def combine_2_spectras(
        darray_min_,
        darray_max_,
        cut_dim,
        cut_point

):
    y1 = cut_point
    cor = cut_dim

    darray_min = darray_min_.reset_coords(drop=True)
    darray_max = darray_max_.reset_coords(drop=True)

    darray_min = darray_min.bnn.set_lDp()
    darray_max = darray_max.bnn.set_lDp()

    darray_min = darray_min.bnn.set_sec()
    darray_max = darray_max.bnn.set_sec()

    darray_min = darray_min.reset_coords(drop=True)
    darray_max = darray_max.reset_coords(drop=True)

    assert is_regular_grid(darray_min['lDp'])
    assert is_regular_grid(darray_max['lDp'])
    assert is_regular_grid(darray_min['secs'])
    assert is_regular_grid(darray_max['secs'])

    # i changed sum to max in the line below
    # assert (darray_min['secs'] - darray_max['secs']).max().item() == 0

    d11 = darray_min.loc[{cor: slice(None, y1)}]
    d22 = darray_max.loc[{cor: slice(y1, None)}]

    first = d22[cor].min().item()

    last = d11[cor].max().item()

    dx = d11[cor].to_series().diff().mean()

    dis = np.abs((first - (last)) / dx)

    if dis < .01:
        d11 = d11[{cor: slice(None, -1)}]

    d3 = xr.concat([d11, d22], cor)

    return d3

    # return d3,d22,d11

    assert is_regular_grid(d3['lDp'])
    assert is_regular_grid(d3['secs'])

    return d3


def open_smps2ds(p):
    UTC = 1
    d = pd.read_csv(p, encoding="ISO-8859-1", sep=';', skiprows=18)

    dd = d.columns.str.match('.*\d\d\.\d')

    sizes = d.columns[dd]

    data = d.loc[:, sizes]

    d['time_'] = d['Date'] + '_' + d['Start Time']

    d['time'] = pd.to_datetime(d['time_'], format='%m/%d/%y_%H:%M:%S') + pd.Timedelta(UTC, unit='h')

    d = d.set_index('time')

    d1 = d[sizes]

    d1.columns.name = 'Dp'
    d1.columns = d1.columns.astype('float') * 1e-9

    d2 = d1.stack()

    d2.name = 'dndlDp'

    ds = d2.to_xarray().to_dataset()

    dtt = d['Total Conc.(#/cm³)']

    dtt.name = 'total conc #/cm3'

    ds[dtt.name] = dtt.to_xarray()
    return ds