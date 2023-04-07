# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---


# load and autoreload

from IPython import get_ipython

# noinspection PyBroadException
try:
    _magic = get_ipython().magic
    _magic('load_ext autoreload')
    _magic('autoreload 2')
except:
    pass

# noinspection PyUnresolvedReferences
import os
# noinspection PyUnresolvedReferences
import glob
# noinspection PyUnresolvedReferences
import sys
# noinspection PyUnresolvedReferences
import pprint
# noinspection PyUnresolvedReferences
import datetime as dt
import pandas as pd
import numpy as np
# noinspection PyUnresolvedReferences
import matplotlib as mpl
# noinspection PyUnresolvedReferences
import matplotlib.colors
# noinspection PyUnresolvedReferences
import matplotlib.pyplot as plt
# noinspection PyUnresolvedReferences
import xarray as xr

import bokeh.plotting as bkp
import bokeh.models as bkm
import bokeh.layouts as bkl
from bnn_tools import basic_funs as bfu

import time
from bnn_tools import gr_gui

CHC_LAT = -16.350427
CHC_LON = -68.131335

# cen_lat = 3
# cen_lon = 3

ALT_LAT = -16.510154
ALT_LON = -68.198745

R = 'R_CENTER'
T = 'TH_CENTER'
L = 'lt'
A = 'AGE'
C = 'CONC'
Z = 'ZMID'
yL = 'XLAT'
xL = 'XLONG'
GA = 'G_AREA'


def get_plot_flx_layout(cen_lat, cen_lon, d1, d2, ds, d, x_axis_type='log'):
    # noinspection PyUnusedLocal
    for fake_init in [1]:
        # d = '2018-05-15 12'
        z = 5000
        th = np.pi
        r = .2

        ds_t = ds.loc[{L: d}]

    # noinspection PyUnusedLocal
    for fake_p_center in [1]:
        p_cb, p_cen, surface_data_source = create_plot_surface(ds_t)
        th_point = create_th_slice_point(p_cen, cen_lon=cen_lon, cen_lat=cen_lat)

    (plot_slice,
     c_bot,
     data_source_vertical,
     z_r_point) = get_plot_slice(ds_t, th=3, z=z, r=r, x_axis_type=x_axis_type)

    (plot_time,
     circle_glyph_time,
     line_time_glyph) = get_time_plot(d1, d2, ds=ds, z=z, th=th, r=r)

    # noinspection PyUnusedLocal
    for fake_interactions in [1]:
        # noinspection PyUnusedLocal
        for fake_add_time_pan in [1]:
            add_time_pan(plot=plot_time,
                         point_data_source=circle_glyph_time.data_source,
                         slice_data_source=data_source_vertical,
                         surface_data_source=surface_data_source,
                         surface_point_data_source=th_point.data_source,
                         ds=ds,
                         cen_lat=cen_lat,
                         cen_lon=cen_lon
                         )

        drag_button = get_drag_button()

        add_z_pan_timeseries(
            cen_lat=cen_lat,
            cen_lon=cen_lon,
            plot_slice=plot_slice,
            ds=ds,
            line_data_source=line_time_glyph.data_source,
            th_point_data_source=th_point.data_source,
            plot_line=plot_time,
            z_r_point_data_source=z_r_point.data_source
        )

        # noinspection PyUnusedLocal
        for fake_add_mouse_pan in [1]:
            add_mouse_pan(
                p_surface=p_cen,
                p_vertical=plot_slice,
                ds=ds,
                data_source=data_source_vertical,
                drag_button=drag_button,
                data_source_circle=th_point.data_source,
                data_source_point_t=circle_glyph_time.data_source,
                cen_lon=cen_lon,
                cen_lat=cen_lat
            )

    # noinspection PyUnusedLocal
    for fake_cons in [1]:
        w_c = 400
        # w_l = 150
        h_c = 400
        h_b = 120

    # noinspection PyUnusedLocal
    for fake_figs in [1]:
        # p_cen = bkp.figure()
        # p_lef = bkp.figure()
        # p_bot = bkp.figure()
        # p_ble = bkm.Spacer()
        pass

    # noinspection PyUnusedLocal
    for fake_sizes in [1]:
        p_cen.width = w_c
        plot_slice.width = w_c

        p_cen.height = h_c
        # p_lef.height = h_c

        plot_slice.height = h_b
        # p_lef.width = w_l

        # p_ble.width = w_l
        # p_ble.height = h_b
        c_bot.height = h_b
        plot_time.height = h_b
        plot_time.width = w_c

    # noinspection PyUnusedLocal
    for fake_layout in [1]:
        lay = bkl.layout(
            [drag_button],
            [p_cen, p_cb],
            [plot_slice, c_bot],
            [plot_time],
        )
    return lay


def interp_ds_th_z_r(ds, th, z, r):
    _dic = {R: r, T: th}
    ds: xr.DataArray
    ds2 = ds.interp(_dic, kwargs=dict(fill_value='extrapolate'))

    dsna = ds2.dropna(Z)
    ds3 = dsna.interp({Z: z}, kwargs=dict(fill_value='extrapolate'))

    return ds3


def add_z_pan_timeseries(line_data_source, ds, cen_lon, cen_lat,
                         th_point_data_source, plot_slice, plot_line, z_r_point_data_source):
    tms = [0, 0]

    def z_pan_function(event):
        tms[1] = time.time()
        if tms[1] - tms[0] > .1:
            r = event.x
            z = event.y

            th_x = th_point_data_source.data['x'][0]
            th_y = th_point_data_source.data['y'][0]
            th, _, _ = get_angle(th_x, th_y, cen_lon, cen_lat)

            # print(th,z,r)

            ds_th_z_t_r = interp_ds_th_z_r(ds, th, z, r)
            x = ds_th_z_t_r[L].values
            y = ds_th_z_t_r.values
            line_data_source.data['x'] = x
            line_data_source.data['y'] = y
            plot_line.y_range.start = 0
            y_max = max(y) * 1.1
            if np.isnan(y_max):
                y_max = 1
            plot_line.y_range.end = y_max

            z_r_point_data_source.data['x'] = [r]
            z_r_point_data_source.data['y'] = [z]
            tms[0] = time.time()

    plot_slice.on_event('pan', z_pan_function)


def plot_flx_server(ds, d1, d2, cen_lat, cen_lon):
    _1 = pd.to_datetime(d1)
    _2 = pd.to_datetime(d2)
    _m = (_2 - _1) / 2 + _1
    d = _m
    lay = get_plot_flx_layout(cen_lat, cen_lon, d1, d2,ds,d)
    # _ = lay
    bkp.output_notebook()

    def _a(doc):
        doc.add_root(lay)

    bkp.show(_a, notebook_url=gr_gui.remote_jupyter_proxy_url)


def create_plot_surface(ds1):
    xs1, ys1 = get_surface_xs_ys(ds1)
    cs1 = get_cs_sum_z(ds1)
    c_map, p_cen, p_cb, surface_data_source = _plot_surface(cs1, xs1, ys1)
    return p_cb, p_cen, surface_data_source


def _plot_surface(cs1, xs1, ys1):
    p = bkp.figure(width=400, height=400, match_aspect=True, aspect_scale=1)
    p.toolbar.logo = None
    cm = bkm.LogColorMapper(palette="Plasma256", low=1e1, high=1e6)
    cs = {'field': 'cs', 'transform': cm}
    _dic = {'xs': xs1, 'ys': ys1, 'cs': cs1}
    surface_data_source = bkm.ColumnDataSource(_dic)
    p.patches('xs', 'ys', color=cs, source=surface_data_source, line_color='white', line_width=.1)
    cb = bfu.get_colorbar(cm, p, vmin=1e1, vmax=1e6)
    set_wheel(p)
    tool = p.select_one({'type': bkm.BoxZoomTool})
    tool.match_aspect = True
    p.toolbar.active_drag = None
    p.add_tools(bkm.CrosshairTool())
    # p.
    # p.aspect_scale=1
    return cm, p, cb, surface_data_source


def get_surface_xs_ys(ds1):
    def _get_cols(_l, ds1_):
        ys = ds1_.reset_coords()[_l].to_dataframe().values
        _ly = len(ys)
        ys1_ = np.split(ys.flatten(), _ly)
        return ys1_

    _la = ['LAT_00', 'LAT_01', 'LAT_11', 'LAT_10']
    _lo = ['LON_00', 'LON_01', 'LON_11', 'LON_10']
    ys1 = _get_cols(_la, ds1)
    xs1 = _get_cols(_lo, ds1)
    return xs1, ys1


def get_cs_sum_z(ds1):
    # dsz = ds1.interp({Z: z}, kwargs=dict(fill_value='extrapolate'))
    dsz = ds1.sum(Z)
    cs = dsz.to_series().values
    return cs


def create_th_slice_point(p: bkp.Figure, cen_lon, cen_lat):
    z_point = p.circle_x([cen_lon + 1], [cen_lat + 1],
                         size=10, fill_color=None, line_color='lightgray')
    # point_draw_tool = bkm.PointDrawTool(renderers=[z_point], num_objects=1)
    # p.add_tools(point_draw_tool)
    return z_point


def add_time_pan(plot, point_data_source, surface_data_source, slice_data_source, ds,
                 surface_point_data_source, cen_lat, cen_lon):
    tms = [0, 0]

    def pan_function(event):
        tms[1] = time.time()
        if tms[1] - tms[0] > .1:
            x = event.x
            point_data_source.data['x'] = [x]

            ds_t = interp_ds_time_ms(ds, x)

            x_surf = surface_point_data_source.data['x'][0]
            y_surf = surface_point_data_source.data['y'][0]
            ang, xx, yy = get_angle(x_surf, y_surf, cen_lon=cen_lon, cen_lat=cen_lat)

            ds_t_th = surface_intp_th(ds_t, th=ang)
            cs_th = get_th_cs(ds_t_th)
            slice_data_source.data['cs'] = cs_th

            cs_surf = get_cs_sum_z(ds_t)
            surface_data_source.data['cs'] = cs_surf

            tms[0] = time.time()

    plot.on_event('pan', pan_function)


def get_drag_button():
    labels = ["sync"]

    checkbox_button_group = bkm.CheckboxButtonGroup(labels=labels, active=[0])
    return checkbox_button_group


def add_mouse_pan(p_surface: bkp.Figure, p_vertical, ds, data_source, drag_button,
                  data_source_circle: bkm.ColumnDataSource, data_source_point_t, cen_lon, cen_lat):
    tms = [0, 0]

    def on_mouse_drag(event):
        if drag_button.active == [0]:
            tms[1] = time.time()
            if (tms[1] - tms[0]) > 0.1:
                x = event.x
                y = event.y
                data_source_circle.data['x'] = [x]
                data_source_circle.data['y'] = [y]
                tms[0] = tms[1]
                ang, xx, yy = get_angle(x, y, cen_lon=cen_lon, cen_lat=cen_lat)
                ds_t = interp_ds_time_ms(ds, data_source_point_t.data['x'][0])
                di = surface_intp_th(ds_t, th=ang)
                cs = get_th_cs(di)
                data_source.data['cs'] = cs
                p_vertical.x_range.start = 0.05
                p_vertical.x_range.end = (xx ** 2 + yy ** 2) ** .5

    p_surface.on_event('pan', on_mouse_drag)


def get_plot_slice(ds, th, z, r, x_axis_type='log'):
    cm = bkm.LogColorMapper(palette="Plasma256", low=1e1, high=1e6)
    color_source = {'field': 'cs', 'transform': cm}
    ds_slice = surface_intp_th(ds, th)
    x1, x2, y1, y2 = get_vertical_xs_ys(ds_slice)
    cs = get_th_cs(ds_slice)
    _dic = {
        'cs': cs,
        'x1': x1,
        'x2': x2,
        'y1': y1,
        'y2': y2,
    }
    data_source = bkm.ColumnDataSource(_dic)
    p = bkp.figure(x_axis_type=x_axis_type)
    p.toolbar.logo = None
    p.x_range = bkm.Range1d(.1, 15)
    p.quad('x1', 'x2', 'y1', 'y2',
           color=color_source,
           source=data_source,
           )
    set_wheel(p)
    p.add_tools(bkm.CrosshairTool())

    cb = bfu.get_colorbar(cm, p, vmin=1e1, vmax=1e6)
    cb.yaxis.ticker.desired_num_ticks = 3

    circle_glyph = p.circle_x([r], [z], size=10, fill_color=None, line_color='lightgray')
    p.toolbar.active_drag = None

    return p, cb, data_source, circle_glyph


def get_time_plot(d1, d2, ds, th, z, r):
    p = bkp.figure(x_axis_type='datetime',
                   # tools=[]
                   )

    ds_t_z_th = interp_ds_th_z_r(ds, th=th, z=z, r=r)
    x = ds_t_z_th[L].values
    y = ds_t_z_th.values
    p_line_glyph = p.line(x, y)
    # p.add_tools(bkm.WheelZoomTool(dimensions='width'))
    p.toolbar.logo = None
    dd1 = ensure_date_ms(d1)
    dd2 = ensure_date_ms(d2)
    p.x_range.start = dd1
    p.x_range.end = dd2
    circle_glyph = p.circle_x([dd1], [0], size=10, fill_color=None, line_color='lightgray')
    set_wheel(p)
    p.toolbar.active_drag = None
    p.xaxis.ticker.desired_num_ticks = 15
    p.add_tools(bkm.CrosshairTool())
    # p.yaxis.visible = False
    # p.y_range

    return p, circle_glyph, p_line_glyph


def set_wheel(p):
    a = None
    for _t in p.tools:
        if isinstance(_t, bkm.WheelZoomTool):
            a = _t
    p.toolbar.active_scroll = a


def interp_ds_time_ms(ds, t_ms):
    t_dt = time_from_ms_to_dt(t_ms)
    ds1 = ds.interp({L: t_dt}, kwargs=dict(fill_value='extrapolate'))
    return ds1


def get_angle(x, y, cen_lon, cen_lat):
    xx = x - cen_lon
    yy = y - cen_lat
    ang = np.mod(np.arctan2(xx, yy), 2 * np.pi)
    return ang, xx, yy


def surface_intp_th(ds: xr.Dataset, th):
    # ds1 = ds.interp({L:t}, kwargs=dict(fill_value='extrapolate'))
    ds2 = ds.interp({T: th}, kwargs=dict(fill_value='extrapolate'))
    return ds2


def get_th_cs(ds_slice):
    cs = ds_slice.to_series().values
    return cs


def get_vertical_xs_ys(ds_slice):
    cols = ['RN', 'RF', 'ZB', 'ZT']

    df = ds_slice.reset_coords()[cols].to_dataframe()

    x1 = df['RN'].values
    x2 = df['RF'].values
    y1 = df['ZB'].values
    y2 = df['ZT'].values

    return x1, x2, y1, y2


def ensure_date_ms(date):
    d_out_ = None
    if type(date) == str:
        d_out_ = string_time_to_ms(date)
    return d_out_


def time_from_ms_to_dt(t_ms):
    dt_ = pd.to_datetime(t_ms, unit='ms')
    return dt_


def string_time_to_ms(d):
    d1 = pd.to_datetime(d)
    d2 = (d1 - pd.to_datetime(0)) / np.timedelta64(1, 'ms')
    return d2


def open_ds_chc(d1, d2):
    p = "/media/volume/paper-2/data-flexpart/new_log_pol_ds_asl_v01.nc"
    def _rellC(d3):
        d3[xL] = d3[R] * np.sin(d3[T]) + CHC_LON
        d3[yL] = d3[R] * np.cos(d3[T]) + CHC_LAT
        return d3

    def _get_lt(ds):
        ds1 = ds
        ds1 = ds1.assign_coords({'lt': ds1['releases'] - pd.Timedelta(4, 'hours')})
        ds1 = ds1.swap_dims({'releases': 'lt'})
        return ds1

    dc: xr.Dataset = _rellC(_get_lt(xr.open_mfdataset(p)))
    dc.attrs.update({'lc': 'CHC'})

    ds1 = dc.loc[{'lt': slice(d1, d2)}]
    ds2 = ds1[C] / ds1[GA]
    ds2.name = 'C/A'

    import bnn_tools.basic_funs as bfu

    zz = bfu.infer_interval_breaks(ds2[Z])

    z1 = zz[0:-1]
    z2 = zz[1:None]

    ds2['ZT'] = z2 + ds2[Z] * 0
    ds2['ZB'] = z1 + ds2[Z] * 0

    zz = bfu.infer_interval_breaks(ds2[R])

    z1 = zz[0:-1]
    z2 = zz[1:None]

    ds2['RN'] = z2 + ds2[R] * 0
    ds2['RF'] = z1 + ds2[R] * 0
    ds2.load();
    return ds2

def open_ds_alto(d1, d2):
    p = "/media/volume/paper-2/data-flexpart/alto_new_log_pol_ds_asl.nc"
    def _rellC(d3):
        d3[xL] = d3[R] * np.sin(d3[T]) + ALT_LON
        d3[yL] = d3[R] * np.cos(d3[T]) + ALT_LAT
        return d3

    def _get_lt(ds):
        ds1 = ds
        ds1 = ds1.assign_coords({'lt': ds1['releases'] - pd.Timedelta(4, 'hours')})
        ds1 = ds1.swap_dims({'releases': 'lt'})
        return ds1

    dc: xr.Dataset = _rellC(_get_lt(xr.open_mfdataset(p)))
    dc.attrs.update({'lc': 'ALTO'})

    ds1 = dc.loc[{'lt': slice(d1, d2)}]
    ds2 = ds1[C] / ds1[GA]
    ds2.name = 'C/A'

    import bnn_tools.basic_funs as bfu

    zz = bfu.infer_interval_breaks(ds2[Z])

    z1 = zz[0:-1]
    z2 = zz[1:None]

    ds2['ZT'] = z2 + ds2[Z] * 0
    ds2['ZB'] = z1 + ds2[Z] * 0

    zz = bfu.infer_interval_breaks(ds2[R])

    z1 = zz[0:-1]
    z2 = zz[1:None]

    ds2['RN'] = z2 + ds2[R] * 0
    ds2['RF'] = z1 + ds2[R] * 0
    ds2.load();
    return ds2
