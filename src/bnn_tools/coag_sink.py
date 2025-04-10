import numpy as np
import pandas as pd
import xarray as xr
from xarray.plot.utils import _infer_interval_breaks as infer_interval_breaks

kB = 1.38064852e-23
'''Boltzmann constant in SI units'''


def calc_coag_coef_fuchs(d1, dens1, d2, dens2, T, P, alpha):
    """
    diameter1, density1, diameter2, density2, temperature, pressure,
    mass accomodation coefficient
    The output is : coagulation coefficient between
    two particles (in air), in m^3/s
    Considering that T and P may vary with time in some
    appications and the d1, d2 arrays may not be predetermined
    we use a least efficient way to calculate the coagulation coefficient

    Parameters
    ----------
    d1 :
        size one in m
    dens1 :
        dens particle one
    d2 :
        size two in m
    dens2 :
        dens particle two
    T :
        temperature in k
    P :
        pressure in Pa
    alpha :
        accomodation coef. (usually 1)

    Returns
    -------
    float
    """
    m1 = dens1 * np.pi / 6 * d1 ** 3
    m2 = dens2 * np.pi / 6 * d2 ** 3
    mfp = calc_MFP(T, P)
    vis = calc_viscosity(T)
    Kn1 = calc_Kn(d1, mfp)
    Kn2 = calc_Kn(d2, mfp)
    Cs1 = calc_CunSlip(Kn1)
    Cs2 = calc_CunSlip(Kn2)
    diff1 = calc_diffusivity(d1, Cs1, vis, T)
    diff2 = calc_diffusivity(d2, Cs2, vis, T)
    v1 = calc_velocity(m1, T)
    v2 = calc_velocity(m2, T)
    l1 = 8 * diff1 / np.pi / v1
    l2 = 8 * diff2 / np.pi / v2
    g1 = np.sqrt(2) / (3 * d1 * l1) * ((d1 + l1) ** 3 - (d1 ** 2 + l1 ** 2) ** (3 / 2)) - d1
    g2 = np.sqrt(2) / (3 * d2 * l2) * ((d2 + l2) ** 3 - (d2 ** 2 + l2 ** 2) ** (3 / 2)) - d2
    a = 2 * np.pi * (diff1 + diff2) * (d1 + d2)
    b = (d1 + d2) / (d1 + d2 + 2 * np.sqrt(g1 ** 2 + g2 ** 2))
    c = 8 / alpha * (diff1 + diff2) / np.sqrt(v1 ** 2 + v2 ** 2) / (d1 + d2)
    return a / (b + c)

def calc_MFP(T, P):
    """
    mean free path approx in air

    Returns
    -------
    float

    References
    ----------
    Mean free path of air, Eq. 2-10, P19, Aerosol Measurement, Third Edition
    """
    a = 66.5 * 1e-9
    b = (101325 / P)
    c = (T / 293.15)
    d = (1 + 110.4 / 293.15)
    e = (1 + 110.4 / T)
    return a * b * c * d / e  # in m

# Dynamic viscosity (rather than kinetic viscosity which is divided by density)
# of air using Sutherland equation, Eq. 2-8, P18, Aerosol Measurement, Third Edition
def calc_viscosity(T):
    a = 18.203 * 10 ** -6
    b = (293.15 + 110.4)
    c = (T + 110.4)
    d = (T / 293.15) ** 1.5
    return a * b / c * d  # Pa*s

# Knudsen number
def calc_Kn(dp, mfp):
    return 2 * mfp / dp

# Cunningham slip correction factor, equation and parameters by Allen & Raabe 1985, for solid particles
# Can also be found in Eq. 2-14, P20,  Aerosol Measurement, Third Edition
def calc_CunSlip(Kn):
    alpha = 1.142
    beta = 0.558
    gamma = 0.999
    a = np.exp(-gamma / Kn)
    return 1 + Kn * (alpha + beta * a)

# Diffusion coefficient
def calc_diffusivity(dp, Cs, vis, T):
    a = (3 * np.pi * vis * dp)
    b = kB * T * Cs
    return b / a  # in m^2/s

# Calculate particle thermal velocity (RMS), in m/s
def calc_velocity(m, T):  # m mass in kg
    a = 8 * kB * T
    b = a / np.pi
    return np.sqrt(b / m)


    # #####
    # # an example is given below
    # dp = collect(10 .^(0:0.1:3))*1e-9 # 1-1000 nm
    # dNdlogdp = ones(length(dp)) # the measured dN/dlogdp or dN/dp
    # # first, convert dN/dlogdp to dN, unless the raw data is dN already
    # dlogdp = vcat(log10(dp[2]/dp[1]), # Julia starts from 1, not zero
    #               [log10(dp[ii+1]/dp[ii]) for ii in 2:length(dp)-1],log10(dp[end]/dp[end-1]))
    # dN = dNdlogdp .* dlogdp # number concentration in m^-3 for each size bin. N_total = sum(dN) for every dp
    # # second, let's calculate the coagulation coefficients, beta (m^3/s) between every new particle and every larger particle
    # dpnew = dp[3:11] # 1.5 - 10 nm particles, for example
    # dNnew = dN[3:11]
    # dens1 = dens2 = 1200 # density, in kg/m3
    # beta = [CalCoagCoefFuchs(dpnew[ii], dens1, dp[jj], dens2) for ii in eachindex(dpnew), jj in eachindex(dp)] # eachindex() is equal to 1:length(dp)
    # # third, the coagulation sink as a function of dpnew (CoagS in 1/s) can be calculated as
    # CoagS = [sum(beta[ii,:].*dN) for ii in eachindex(dpnew)]
    # # finally, the CoagSnk term (in m^-3 s^-1) is obtained by calculating dot(CoagS, dpnew), i.e.,
    # CoagSnk = sum(CoagS.*dNnew)
    # # J = dN/dt + net CoagSnk + GR term = dN/dt + CoagSnk - CoagSrc + GR term
    # # We can discuss how to calculate J after this CoagS

def add_lims_lDp(dc):
    """
    add bin edges to the Dp dimension.
    the edges are calculated in log space
    Parameters
    ----------
    dc : xr.DataFrame

    Returns
    -------
    xr.DataFrame
        original `dc` with lDpm, lDpM, Dpm and DpM added.

    """
    blDp = infer_interval_breaks(dc['lDp'])
    blDpm = blDp[:-1]
    #DA why am I using 1 here instead of 0? Now it makes sense
    blDpM = blDp[1:]

    #limits in log
    dc['lDpm'] = dc['Dp'] * 0 + blDpm
    dc['lDpM'] = dc['Dp'] * 0 + blDpM

    #limits in linear
    dc['Dpm'] = 10 ** dc['lDpm']
    dc['DpM'] = 10 ** dc['lDpM']

    return dc

def calc_coag_snk_xr(*, dN_tot_m3, d1, d2, P, T, alpha, dens1, dens2):
    """
    Calculate coagulation sink.

    Parameters
    ----------
    dN_tot_m3 : float
        Number of particles in the bin per m3.
    d1 : float
        Lower diameter in m.
    d2 : float
        Upper diameter in m.
    P : float
        Pressure in pascals.
    T : float
        Temperature in Kelvin.
    alpha : float, optional
        Mass accommodation coefficient (usually 1).
    dens1 : float, optional
        Density 1 kg/m3, usually 1200.
    dens2 : float, optional
        Density 2 kg/m3, usually 1200.

    Returns
    -------
    tuple
        A tuple containing:
            - Number of particles lost to coagulation between d1 and d2 in [# m-3 s-1]
            - The true minimum diameter
            - The true max bin
    """

    assert d1<d2, \
        'd2 needs to be higher than d1'

    dN_total_ = add_lims_lDp(dN_tot_m3)

    assert 'Dp' in dN_total_.dims, \
        'probably you need to set Dp [m] instead of lDp as de dimension'

    #
    _Dp_d1_d2 = dN_total_['Dp'].loc[{'Dp': slice(d1, d2)}]

    Dp_d1_d2 = xr.DataArray(
        _Dp_d1_d2.values,
        dims='Dp_d1_d2',
        coords={'Dp_d1_d2': _Dp_d1_d2.values})

    _dN12 = dN_total_.loc[{'Dp': slice(d1, d2)}]
    dN12 = _dN12.rename({'Dp': 'Dp_d1_d2'})

    #     dN12 = xr.DataArray(_dN12.values, dims='Dp_d1_d2', coords={'Dp_d1_d2': _Dp_d1_d2.values})
    CoagS = calc_coagS(Dp_d1_d2, P, T, alpha, dN_total_, dens1, dens2)
    #     print(CoagS)
    CoagSnk_: xr.DataArray = (dN12 * CoagS).sum(['Dp_d1_d2'])
    CoagSnk = CoagSnk_.assign_attrs({'units': '#/m3'})
    CoagSnk.name = 'CoagSnk'
    # CoagSnk.bnn.u('1/s')

    # true_d1 = _dN12['Dpm'].min().item()
    # true_d2 = _dN12['DpM'].max().item()

    res_ds: xr.Dataset = CoagSnk.to_dataset()
    res_ds['Dpm'] = _dN12['Dpm'].min().assign_attrs({'units': 'm'})
    res_ds['DpM'] = _dN12['DpM'].max().assign_attrs({'units': 'm'})
    res_ds['CoagS'] = CoagS

    # dic_out = dict(
    #     CoagSnk = CoagSnk,
    #     true_d1 = true_d1,
    #     true_d2 = true_d2,
    #     CoagS   = CoagS  ,
    #     res     = res_ds
    # )

    res_ds
    return res_ds

def calc_coagS(Dp_d1_d2, P, T, alpha, dN_total_, dens1, dens2):
    """
    be aware that we are only considering coagS due to
    particle higher thant Dp_d1_d2

    Parameters
    ----------
    Dp_d1_d2 :
    P :
    T :
    alpha :
    dN_total_ :
    dens1 :
    dens2 :

    Returns
    -------

    """
    Dp_, Dp12_ = xr.broadcast(dN_total_['Dp'], Dp_d1_d2)
    beta_ = calc_coag_coef_fuchs(
        d1=Dp12_, dens1=dens1, d2=Dp_, dens2=dens2, T=T, P=P, alpha=alpha)

    # here we filter for beta from smaller particles
    # i am not sure if i should use > or =>
    beta = beta_.where(beta_['Dp']>=beta_['Dp_d1_d2'])
    CoagS_ = (beta * dN_total_).sum('Dp')
    CoagS = CoagS_.assign_attrs({'units': '1/s'})
    return CoagS


    # % dp and mfp in nm
    # function CS = CalCS(dp, mfp)
    # % diffusion coefficient of sulfuric acid, can be replaced with more accurate values
    # D = 9.4e-06; % m^2/s
    #
    # alpha = 1;
    # Kn = mfp/dp;
    # beta_m = (1+Kn) / (1+ (4/(3*alpha) + 0.377)*Kn + 4/(3*alpha)*Kn^2);
    # CS = 2*pi*dp*1e-9*D*beta_m; % CS contributed by a single particle
    # end

    # % dp and mfp in m

def calCS_single_par(dp_m, T, P):
    D_sa = 9.4e-06
    '''diffusion coefficient of sulfuric acid m^2/s
    can be replaced with more accurate values'''

    alpha = 1
    # Kn = mfp/dp;

    mfp = calc_MFP(T, P)
    Kn = calc_Kn(dp_m, mfp)

    a = (1 + Kn)
    b = (4 / (3 * alpha) + 0.377) * Kn
    c = 4 / (3 * alpha) * Kn ** 2
    beta_m = a / (1 + b + c)

    CS = 2 * np.pi * dp_m * D_sa * beta_m
    '''CS contributed by a single particle'''
    return CS

def calc_CS(*,dN_m3, T, P):
    '''data array containing the particle number concentration
    - Dp is in meters
    - Concentration in m-3, etc.
    - T in Kelvin
    - P in PA
    '''

    dN_m3: xr.DataArray
    cs_ = calc_bin_CS(dN_m3 = dN_m3, T = T, P = P)
    cs = cs_.sum('Dp')

    cs.name = 'CS'
    cs.bnn.u('1/s')

    '''condensation sink in units of 1/s '''
    return cs


def calc_bin_CS(*, dN_m3, T, P):
    Dp = dN_m3['Dp']
    cs_single_par = calCS_single_par(dp_m=Dp, T=T, P=P)
    cs_ = cs_single_par * dN_m3
    cs_.name = 'CS per bin'
    cs_.bnn.u('1/s')
    return cs_


def test_CoagSnk():
    dp = 10 ** np.arange(0, 3.001, .1) * 1e-9

    dNdlogdp = np.ones(len(dp))

    a = np.log10(dp[1:] / dp[0:-1])
    dlogdp = [*a, a[-1]]

    dN = dNdlogdp * dlogdp

    dpnew = dp[2:11]
    dNnew = dN[2:11]

    dens1 = dens2 = 1200

    beta = np.array(
        [[calc_coag_coef_fuchs(dpn, dens1, dp_, dens2, T=293.15, P=101325, alpha=1) for dpn in dpnew] for dp_ in dp])

    #     print(sum(sum(beta)))
    #     print(sum(beta))

    CoagS = [sum(b_ * dN) for b_ in beta.T]

    #     print(CoagS)

    CoagSnk = sum(CoagS * dNnew)
    test = 4.486361877087417e-12
    assert CoagSnk == test, f'{CoagSnk} is not {test}'
    return True

def test_xarray_form():
    dp = 10 ** np.arange(0, 3.001, .1) * 1e-9

    dNdlogdp = np.ones(len(dp))

    da = xr.DataArray(dNdlogdp, dims='Dp', coords={'Dp': dp})
    da.name = 'dndlDp'

    da['lDp'] = np.log10(da['Dp'])

    da['dlDp'] = da['lDp'] * 0 + pd.Series(
        infer_interval_breaks(da['lDp'], check_monotonic=True)).diff().dropna().values
    da['dN'] = da * da['dlDp']
    daN = da['dN']

    d1 = 1.5e-9
    d2 = 1e-8
    T = 293.15
    P = 101325
    alpha = 1

    dens1 = dens2 = 1200

    CoagSnk, td1, td2 = calc_coag_snk_xr(
        dN_tot_m3 = daN, d1 = d1, d2 = d2, P=P, T=T, alpha=alpha, dens1=dens1, dens2=dens2)
    test = 4.486361877087417e-12
    assert (CoagSnk - test) / CoagSnk < .00000001, f'{CoagSnk} is not {test}'
    #     print(((CoagSnk - test)/CoagSnk)*100)
    return True