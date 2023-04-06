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
ALPHA = .5

for fake_imports in [1]:
    from IPython import get_ipython

    # noinspection PyBroadException
    try:
        _magic = get_ipython().run_line_magic
        _magic("load_ext", "autoreload")
        _magic("autoreload", "2")
    except:
        pass

    # noinspection PyUnresolvedReferences
    import datetime as dt
    # noinspection PyUnresolvedReferences
    import glob
    import os
    # noinspection PyUnresolvedReferences
    import pprint
    # noinspection PyUnresolvedReferences
    import sys
    import warnings

    import bnn_tools.basic_funs as bfu
    # noinspection PyUnresolvedReferences
    import bnn_tools.bnn_array
    # noinspection PyUnresolvedReferences
    import bnn_tools.coag_sink as cs
    # noinspection PyUnresolvedReferences
    import bnn_tools.funs as fu
    # noinspection PyUnresolvedReferences
    import matplotlib as mpl
    # noinspection PyUnresolvedReferences
    import matplotlib.colors
    # noinspection PyUnresolvedReferences
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy.optimize
    # noinspection PyUnresolvedReferences
    import seaborn as sns
    import xarray as xr

    warnings.filterwarnings("ignore", category=scipy.optimize.OptimizeWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # import bokeh
    # import bokeh.models.tools
    # import bokeh.palettes
    # from bokeh.layouts import Column, Row, gridplot, column
    # from bokeh.models import ColorBar, ColumnDataSource, CustomJS, LogColorMapper,Div
    # from bokeh.models.tools import CrosshairTool, PointDrawTool, PolyEditTool
    # from bokeh.models.widgets import DataTable, RangeSlider, TableColumn
    # from bokeh.plotting import Figure, figure, output_notebook, show
    # from bokeh.models import Range1d, LinearAxis, LogAxis, PolyDrawTool
    # from bokeh.models.tools import BoxEditTool

    import bokeh as bk
    import bokeh.layouts as bkl
    import bokeh.models as bkm
    import bokeh.plotting as bkp
    import bokeh.palettes

for fake_cons in [1]:
    COLS = bk.palettes.Category10[10]


# noinspection PyUnresolvedReferences
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


def plot_gr_guide(ti, gr, p_circle):
    y = np.geomspace(1, 100)
    # x = np.linspace(ti,ti+24*3600*2)
    # y = 1 + x*gr/3600
    x = (y - 1) / (gr / 3600000) + ti
    # p.line(x,y,color = 'black', line_width = 2)
    # p.line(x,y,color = 'white', line_dash = 'dashed')
    p_circle.data_source.data = {'x': x, 'y': y}
    # p_circle.data_source.data['y'] = y


def open_rad_file():
    f = "~/Apaper2-goes/goes_and_mod_irrad_chc_alt.nc"
    return xr.open_dataset(f).to_dataframe()


def open_bc_chc_file():
    df_bc_chc = xr.open_dataarray(
        "~/Abnn-gr/small_data/BC_chc_v01_giancarlo.nc").to_dataframe()
    return df_bc_chc


def set_wheel(p):
    a = None
    for _t in p.tools:
        if isinstance(_t, bkm.WheelZoomTool):
            a = _t
    p.toolbar.active_scroll = a


def _plot_gui_del(dfm, dfM):
    # output_notebook()
    #
    # p = plot_rad(dfm, dfM)
    #
    # def bkapp(doc):
    #     doc.add_root(p)
    #
    # show(bkapp, notebook_url=remote_jupyter_proxy_url)
    pass


def add_bc_del(p, bc):
    # Setting the second y axis range name and range
    p.extra_y_ranges = {"foo": bokeh.models.Range1d()}

    # Adding the second axis to the plot.
    p.add_layout(bokeh.models.LinearAxis(y_range_name="foo"), "right")
    p.line(bc.index, bc["BC"], y_range_name="foo")


def plot_top_line(dfm: pd.DataFrame, axis_types=None):
    if axis_types is None:
        axis_types = ['linear'] * len(dfm.columns)

    # assert len(dfm.columns) <= 2
    # print(dfm)

    for fake_plot1 in [1]:
        p = bkp.Figure(x_axis_type="datetime", width=1000, height=400, y_axis_type=axis_types[0])

        col0 = dfm.columns[0]
        p.line(
            dfm.index,
            dfm[col0],
            color=COLS[0],
            # y_range_name=y_range_name
            # legend_label=_a[i],
            line_width=2,
            # line_dash=_l[i],
            line_alpha=.5
        )
        col__min = dfm[col0].min()
        col__max = dfm[col0].max()

        if np.isnan(col__min):
            col__min = 0
        if np.isnan(col__max):
            col__max = 1

        # print(col__max,col__min)
        p.y_range.start = col__min
        p.y_range.end = col__max

        p.yaxis.axis_label = col0
        p.yaxis.axis_label_text_color = COLS[0]
        p.yaxis.axis_label_text_font_size = '6pt'
        p.yaxis.major_label_text_font_size = "6pt"
        p.yaxis.major_label_text_color = COLS[0]

    for i in range(1, len(dfm.columns)):
        col1 = dfm.columns[i]
        axs_dic = {'linear': bkm.LinearAxis, 'log': bkm.LogAxis}
        at2 = axs_dic[axis_types[i]]
        y_range_name = f'yrange_{i}'
        ax2 = at2(y_range_name=y_range_name)
        _min = dfm[col1].min()
        _max = dfm[col1].max()

        if np.isnan(_min):
            _min = 0
        if np.isnan(_max):
            _max = 1

        # print(_min,_max)
        p.extra_y_ranges = {**p.extra_y_ranges, y_range_name: bokeh.models.Range1d(_min, _max)}
        # Adding the second axis to the plot.
        p.add_layout(ax2, 'right')

        p.line(dfm.index, dfm[col1],
               color=COLS[i],
               y_range_name=y_range_name,
               line_width=2,
               line_alpha=.5
               )
        ax2.axis_label = col1
        ax2.axis_label_text_color = COLS[i]
        ax2.axis_label_text_font_size = '6pt'
        ax2.major_label_text_font_size = "6pt"
        ax2.major_label_text_color = COLS[i]

    set_wheel(p)
    p.add_tools(bkm.CrosshairTool(dimensions='width'))
    return p


def plot_bc(bc_df):
    p = bkp.figure(x_axis_type="datetime", y_axis_type="log")
    p.line(bc_df.index, bc_df["BC"] + 0.1)
    return p


def plot_wd(dfM):
    p = bkp.figure(x_axis_type="datetime")
    p.line(dfM.index, dfM["WD"] / 360 * 12)
    p.line(dfM.index, dfM["WDstdv"] / 360 * 12 * 5, color='red')
    # p.line(dfM.index, dfM["WD"] /360 * 12+12)
    return p


def plot_ws(dfM):
    p = bkp.figure(x_axis_type="datetime")
    p.line(dfM.index, dfM["WS"])
    return p


def open_ds_psd(d1, d2):
    ds = xr.open_dataarray("~/Abnn-gr/v2_tratead_psd_data_smps_nais_chc_alto.nc")

    i = "nais_ion_neg_neutral_smps"
    loc = "chc"

    ds1 = ds.loc[{"id": i, "loc": loc, "time": slice(d1, d2)}]
    return ds1


def basic_bokeh_plot(ds):
    ######### EDGES #########
    vmin = 100
    vmax = 10_000
    # get intervals time
    x0, x1 = bfu.infer_interval_breaks(ds["time"])[[0, -1]]
    # get interval break. add 9 for nanometers
    y0, y1 = bfu.infer_interval_breaks(ds["lDp"])[[0, -1]] + 9
    ######### FIGURE ############
    _w = bokeh.models.tools.WheelZoomTool()
    _c = bokeh.models.tools.CrosshairTool(dimensions="height")
    _c.name = "crosshair"

    _cw = bokeh.models.tools.CrosshairTool(dimensions="width")
    _cw.name = "crosshairw"
    p = bkp.Figure(
        x_axis_type="datetime",
        width=1000,
        height=400,
        tools=["pan", "box_zoom", "reset", _w, _c, _cw, "hover"],
        y_axis_type="log",
    )
    p.toolbar.active_scroll = _w
    color_mapper = bkm.LogColorMapper(palette="Plasma256", low=vmin, high=vmax)
    color_mapper.tags = ['CMAP']
    p.image(
        [ds.transpose("lDp", "time").data],
        x=x0,
        dw=x1 - x0,
        y=10 ** y0,
        dh=10 ** y1 - 10 ** y0,
        color_mapper=color_mapper,
    )
    ######### COLOR BAR #############
    cb = bfu.get_colorbar(color_mapper, p, vmax, vmin)

    p.xaxis.ticker.desired_num_ticks = 15
    # p.title = 'sdfsdf'

    return cb, p


def add_mouse_event(p, pl_line, pb_line, ds2):
    # print('sd')
    import time

    from bokeh.events import PointEvent
    from bokeh.models import Title

    tms = [time.time(), None]
    _tt = Title(text=" ",
                align="left", text_font_size='8pt',
                text_font='monospace'
                )
    p.add_layout(_tt, "below")

    def on_mouse_move(event: PointEvent):
        # print('sdf')
        tms[1] = time.time()
        if (tms[1] - tms[0]) > 0.1:
            x = event.x
            y = event.y
            # draw_b_line(ds2, y, pb_line)
            draw_l_line(ds2, x, pl_line, p)
            draw_b_line(ds2, y, pb_line, p)

            #             draw_b_line(dsg, y, pb_line1)
            #             draw_l_line(dsg, x, pl_line1)
            tms[0] = tms[1]

            _s = pd.to_datetime(x, unit='ms').strftime('%y-%m-%d %H:%M %a')
            _t = f'{_s} |{y:8.1f}nm'
            p.title = _t
            # print(_t,p)
            _tt.text = _t

            # print(x)

    p.on_event("mousemove", on_mouse_move)
    # p.on_event('mousemove',lambda x: print(3))


def draw_l_line(ds2, x, pl_line, p):
    _t = np.array(x).astype("datetime64[ms]")
    # print(_t)
    _d = ds2.interp({"time": _t}, kwargs=dict(fill_value="extrapolate"))

    _d1 = np.log10(p.y_range.start) - 9
    _d2 = np.log10(p.y_range.end) - 9

    # print(_d1,_d2)
    _d = _d.loc[{"lDp": slice(_d1, _d2)}]
    # print (_d)
    _x = _d.values
    _y = 10 ** (_d["lDp"].values + 9)
    pl_line.data_source.data = {"x": _x, "y": _y}


def draw_b_line(ds2, y, pb_line, p):
    _t1 = np.array(p.x_range.start).astype("datetime64[ms]")
    _t2 = np.array(p.x_range.end).astype("datetime64[ms]")

    # print(y)

    _d1 = np.log10(y) - 9

    # print(_t)
    _d = ds2.interp({"lDp": _d1}, kwargs=dict(fill_value="extrapolate"))

    # print(_d1,_d2)
    _d = _d.loc[{"time": slice(_t1, _t2)}]
    # print (_d)
    _y = _d.values
    _x = _d["time"].values

    # print(_x,_y)
    pb_line.data_source.data = {"x": _x, "y": _y}
    # pb_line.yaxis.axis_label_text_font_size = '6pt'




def black_format_dic(dic):
    from black import FileMode, format_str

    str_dic = repr(dic)
    res = format_str(str_dic, mode=FileMode())
    return res


def open_dic(file):
    with open(file, "r") as file:
        dic = eval(file.read())
    return dic


def save_dic(*, dic, srs_dic, color_mapper, file):
    # mask_df = source.to_df().to_dict('list')
    color_high = color_mapper.high
    color_low = color_mapper.low
    # xy = xy.tolist()

    for i, s in srs_dic.items():
        dic[i] = s.to_df().to_dict('list')
    dic["color_high"] = color_high
    dic["color_low"] = color_low
    # dic["xy"] = xy

    dic_str = black_format_dic(dic)

    with open(file, "w") as f:
        f.write(dic_str)
    return dic_str


def create_table(source):
    ops = [
        "bc plume",
        "start npf",
        "manual npf",
        "clearsky",
        "cloudy",
        "semi clearsky",
        "dry to wet",
        "wet to dry",
        "airmass change",
        "cloud to sun",
        "lp airmass",
        # "npf start",
        "alto airmass",
        "no data",
        # "clear sky",
        "npf airmass",
        # "semi clear sky",
        "no smps data",
        "sun to cloud",
        # "q plume",
        "change airmass",
        # "clarsky",
        # "bv plume",
        "manual nucleation",
        "alto lp airmass",
        "no lp alto influence",
        "sa plume",
        "start nuc.",
        "no nais data",
        "no rad data",
        "airmass lowland",
        "npf only nuc",
        "only nuc",
        # "no npf clear sky day",
        # "nucleation",
        "same airmass",
        # "posible npf 2",
        # "manual gr",
        # "semi clear sky day",
        # "semi clearky",
    ]
    _cols = [
        bkm.TableColumn(field='txt',editor=bkm.SelectEditor(options = ops)),
        bkm.TableColumn(field='comment')
    ]
    pdt_ta = bkm.DataTable(
        columns=_cols,
        width=300,
        source=source,
        height_policy='min',
        min_height=100,
        max_height=300,
        editable=True
    )
    return pdt_ta


def add_point_tool(p, color='Blue', name='poly', source=None):
    # p.scatter([0, 1], [0, 1])
    pdt_c = color
    if source is None:
        source = bkm.ColumnDataSource({'xs': [], 'ys': [], 'txt': [],'comment':[]})
    pdt_re = p.circle('xs', 'ys', source=source, color=pdt_c,
                      line_width=1, alpha=ALPHA, size=10, line_alpha=1)
    # pdt_re1 = p.circle([], [], color=pdt_c, alpha=.4, size=10)
    pdt_to = bkm.PointDrawTool(renderers=[pdt_re])
    p.add_tools(pdt_to)
    pdt_ta = create_table(source)
    col = bkl.column(bkm.Div(text=name), pdt_ta)
    return col, pdt_re, source


def add_rect_tool(p, color='Blue', name='rect', source=None):
    # p.scatter([0, 1], [0, 1])
    pdt_c = color
    # print(source)
    if source is None:
        source = bkm.ColumnDataSource(
            {'x': [], 'y': [], 'txt': [], 'width': [], 'height': [],'comment':[]})
    pdt_re = p.rect(
        'x', 'y', 'width', 'height',
        source=source, color=pdt_c, line_width=4, alpha=ALPHA)

    pdt_re1 = p.rect(
        'x', 'y', 'width', 'height',
        source=source, line_color=pdt_c, line_width=1, alpha=1, fill_color=None)
    # pdt_re1 = p.circle([], [], color=pdt_c, alpha=.4, size=10)
    pdt_to = bkm.BoxEditTool(renderers=[pdt_re, pdt_re1])
    p.add_tools(pdt_to)
    pdt_ta = create_table(source)
    col = bkl.column(bkm.Div(text=name), pdt_ta)
    return col, pdt_re, source


def add_poly_tool(p, color='Blue', name='poly', patch='patches', source=None):
    # p.scatter([0, 1], [0, 1])
    # p:Figure
    # p.patch
    # print(source)
    pdt_c = color
    if source is None:
        source = bkm.ColumnDataSource({'xs': [], 'ys': [], 'txt': [],'comment':[]})
    pdt_re = getattr(p, patch)(
        'xs', 'ys', source=source, color=pdt_c, line_width=4,
        alpha=ALPHA
    )

    pdt_re_l = getattr(p, patch)(
        'xs', 'ys', source=source, line_color=pdt_c, line_width=1, color=None)

    pdt_re1 = p.circle([], [], color=pdt_c, alpha=ALPHA, size=10)

    _ren = [pdt_re, pdt_re_l]
    pdt_to = bkm.PolyDrawTool(renderers=_ren, vertex_renderer=pdt_re1)

    p.add_tools(pdt_to)
    pdt_ta = create_table(source)
    col = bkl.column(bkm.Div(text=name), pdt_ta)
    return col, pdt_re, source


# for fake_bokeh in [1]:
def bokeh_gr_gui(init_file, *, _open_rad_df_, _open_wind_dir_df, _open_psd_ds, _open_bc_df, _open_wind_spd_df,
                 _open_top_4, day_plus=2, day_minus=0):
    bkp.output_notebook()
    # def _m_plot():
    dic = open_dic(init_file)

    def _open_source(_str):
        df = dic[_str]
        if df == {}:
            df = None
        else:
            if 'comment' not in df:
                df['comment'] = [None] * len(df['txt'])
            # print(df)
            df = bkm.ColumnDataSource(df)
        return df

    pdt_so = _open_source('poly_df')
    mdt_so = _open_source('multi_df')
    pnt_so = _open_source('point_df')
    rec_so = _open_source('rec_df')

    d0 = pd.to_datetime(dic['date'])

    d1 = d0 - pd.Timedelta(24 * day_minus, 'hour')
    d2 = d0 + pd.Timedelta(24 * day_plus, 'hour')

    for fake_open_dfs in [1]:
        # this should return df/psd with columns for each line with a max of 2 cols
        # print(d1,d2)
        # print(df_rad0)

        df_win_dir2 = _open_wind_dir_df(d1, d2)
        df_win_spd3 = _open_wind_spd_df(d1, d2)
        df_4 = _open_top_4(d1, d2)
        ds_psd = _open_psd_ds(d1, d2)
        df_bc1 = _open_bc_df(d1, d2)
        df_rad0 = _open_rad_df_(d1, d2)


    for fake_cons in [1]:
        w = 700
        h = 350
        # h1 = 150
        h2 = 70
        w1 = 100
        pad = 60
        pad_bo = 60
        wl = 400

    for fake_figures in [1]:
        p_cb, p_ce = basic_bokeh_plot(ds_psd)
        color_mapper = p_ce.select_one({'tags': 'CMAP'})
        # print(color_mapper)
        # color_mapper.high = dic['color_high']
        # color_mapper.low = dic['color_low']

        p_to = plot_top_line(df_rad0)
        p_to1 = plot_top_line(df_bc1)
        p_to2 = plot_top_line(df_win_dir2)
        p_to3 = plot_top_line(df_win_spd3)
        p_to4 = plot_top_line(df_4)

        # add_bc_del(p_to)

        p_le = bkp.figure(y_axis_type="log", x_axis_type="log")
        p_le.xaxis.major_label_orientation = np.pi / 4
        p_le.xaxis.major_label_text_font_size = "6pt"
        p_le.xaxis.ticker.desired_num_ticks = 3

        p_bo = bkp.figure(x_axis_type="datetime", y_axis_type="log")
        p_bo.yaxis.major_label_text_font_size = "6pt"
        p_bo.axis.ticker.desired_num_ticks = 3

        # p_to.legend.visible = False

    for fake_interaction in [1]:
        # p_ce.multi_polygons()
        # p1 = p_ce.patches("xs", "ys", line_width=2, alpha=0.4, source=source)
        # draw_tool_p1 = PolyDrawTool(renderers=[p1])
        # p_ce.add_tools(draw_tool_p1)
        cols = bokeh.palettes.Set1[9]
        pdt_col, pdt_re, pdt_so = add_poly_tool(
            p_ce, color=cols[0], name='poly', patch='patches', source=pdt_so)
        mdt_col, mdt_re, mdt_so = add_poly_tool(
            p_ce, color=cols[1], name='multi', patch='multi_line', source=mdt_so)
        pnt_col, pnt_re, pnt_so = add_point_tool(
            p_ce, color=cols[2], name='point', source=pnt_so)
        rec_col, rec_re, rec_so = add_rect_tool(
            p_ce, color=cols[4], name='rect', source=rec_so)

        table_cols = bkl.column(pdt_col, mdt_col, pnt_col, rec_col)

        pet_re = p_ce.circle([], [], color='Red', alpha=.4, size=10)
        pet = bkm.PolyEditTool(renderers=[pdt_re, mdt_re], vertex_renderer=pet_re)
        p_ce.add_tools(pet)
        pass

    for fake_set_crosssections in [1]:
        p_le_line = p_le.line([], [])
        p_bo_line = p_bo.line([], [])
        add_mouse_event(p_ce, p_le_line, p_bo_line, ds_psd)

        path_map = '/home/jupyter-ubuntu/Apeper-flx/tmp_data/flx_chc/'
        d1_ = (d1 - pd.to_datetime(0)) / np.timedelta64(1, 'ms')
        p_ = get_path(path_map, d1_)
        img, xdim, ydim = get_fig(p_)
        p_map = get_map_fig(h2, wl-w1, xdim, ydim)
        rgb = p_map.image_rgba(image=[img], x=[0], y=[0], dw=[xdim], dh=[ydim])
        add_mouse_event_map(path_map, rgb, p_ce)

        path_map1 = '/home/jupyter-ubuntu/Apeper-flx/tmp_data/flx_chc_long/'
        d1_ = (d1 - pd.to_datetime(0)) / np.timedelta64(1, 'ms')
        p_1 = get_path(path_map1, d1_)
        img1, xdim1, ydim1 = get_fig(p_1)
        p_map1 = get_map_fig(h2, wl-w1, xdim1, ydim1)
        rgb1 = p_map1.image_rgba(image=[img1], x=[0], y=[0], dw=[xdim1], dh=[ydim1])
        add_mouse_event_map(path_map1, rgb1, p_ce)

    for fake_tools in [1]:
        p_to.add_tools(p_ce.select_one({"name": "crosshair"}))
        p_to1.add_tools(p_ce.select_one({"name": "crosshair"}))
        p_to2.add_tools(p_ce.select_one({"name": "crosshair"}))
        p_to3.add_tools(p_ce.select_one({"name": "crosshair"}))
        p_to4.add_tools(p_ce.select_one({"name": "crosshair"}))

        p_bo.add_tools(p_ce.select_one({"name": "crosshair"}))
        p_le.add_tools(p_ce.select_one({"name": "crosshairw"}))

        p_ce.toolbar.active_inspect = [
            p_ce.select_one({"name": "crosshair"}),
            p_ce.select_one({"name": "crosshairw"}),
        ]
        set_wheel(p_to1)
        set_wheel(p_to2)
        set_wheel(p_to3)
        set_wheel(p_to4)

        p_to.toolbar_location = None
        p_to1.toolbar_location = None
        p_to2.toolbar_location = None
        p_to3.toolbar_location = None
        p_to4.toolbar_location = None
        p_le.toolbar_location = None
        p_bo.toolbar_location = None

    for fake_set_sizes in [1]:
        p_ce.width = w
        p_ce.height = h
        p_le.width = w1
        p_le.height = h
        p_to.width = w
        p_to.height = h2
        p_to1.width = w
        p_to1.height = h2
        p_to2.width = w
        p_to2.height = h2
        p_to3.width = w
        p_to3.height = h2
        p_to4.width = w
        p_to4.height = h2
        p_bo.width = w
        p_bo.height = h2
        p_cb.height = h

        central_col_figs = [
            p_ce,
            p_to,
            p_to1,
            p_to2,
            p_to3,
            p_to4,
            p_bo,
        ]

        for _p in central_col_figs:
            _p: bkp.Figure
            _p.frame_width = w - 300
            # _p.min_border_left = 100

    for fake_set_pads in [1]:
        p_le.min_border_bottom = pad_bo
        p_ce.min_border_bottom = pad_bo
        p_to.min_border_left = pad
        p_to1.min_border_left = pad
        p_to2.min_border_left = pad
        p_to3.min_border_left = pad
        p_to4.min_border_left = pad
        p_ce.min_border_left = pad
        p_bo.min_border_left = pad
        # p_to.min_border_right = pad

    for fake_set_ranges in [1]:
        p_to.x_range = p_ce.x_range
        p_le.x_range.flipped = True
        p_bo.x_range = p_ce.x_range
        p_to1.x_range = p_ce.x_range
        p_to2.x_range = p_ce.x_range
        p_to3.x_range = p_ce.x_range
        p_to4.x_range = p_ce.x_range
        p_le.y_range = p_ce.y_range

    for fake_set_dic in [1]:
        # if dic['mask_df'] != {}:
        #     source.data = dic['mask_df']

        color_mapper.high = dic['color_high']
        color_mapper.low = dic['color_low']
        p_cb.y_range.end = dic['color_high']
        p_cb.y_range.start = dic['color_low']

    for fake_save in [1]:
        srs = {
            'poly_df' : pdt_so,
            'multi_df': mdt_so,
            'point_df': pnt_so,
            'rec_df'  : rec_so
        }

        # xy = np.array([])
        def _save_dic3(attr, old, new):
            # print('saving')
            save_dic(dic=dic,
                     srs_dic=srs,
                     color_mapper=color_mapper, file=init_file)

        def _save_dic():
            # print('saving')
            save_dic(dic=dic,
                     srs_dic=srs,
                     color_mapper=color_mapper, file=init_file)

        # source.on_change('data', _save_dic3)
        pdt_so.on_change('data', _save_dic3)
        mdt_so.on_change('data', _save_dic3)
        pnt_so.on_change('data', _save_dic3)
        rec_so.on_change('data', _save_dic3)

    # central_column = bkl.column(bkl.row(p_to4), p_to3, p_to2, bkl.row(p_to1), bkl.row(p_to), bkl.row(p_ce, p_cb), p_bo)
    _sp = bkl.Spacer(width=w1)
    _sp1 = bkl.Spacer(width=wl)
    c_row = bkl.row(p_ce, p_cb)
    l_row = bkl.row(bkl.Spacer(width_policy='max'), p_le, width=wl)
    l_row = bkl.row(p_map1, p_le, width=wl)
    c_top = bkl.column(p_to4, p_to3, p_to2, p_to1, p_to)

    central_column = bkl.layout(
        [p_map,_sp, c_top],
        [p_map1,p_le, c_row],
        [_sp1, p_bo],
    )

    g_plot = bkl.row(central_column, table_cols)
    # g_plot = bkl.gridplot(
    #     [
    #         [None, p_to4, None],
    #         [None, p_to3, None],
    #         [None, p_to2, None],
    #         [None, p_to1, None],
    #         [None, p_to, None],
    #         [p_le, p_ce, p_cb],
    #         [None, p_bo, None],
    #     ],
    #     merge_tools=False,
    # )

    for fake_add_touchs in [1]:
        p_to.xaxis.major_label_text_font_size = "0px"
        p_to1.xaxis.major_label_text_font_size = "0px"
        p_to2.xaxis.major_label_text_font_size = "0px"
        p_to3.xaxis.major_label_text_font_size = "0px"
        p_to4.xaxis.major_label_text_font_size = "0px"
        p_bo.xaxis.major_label_text_font_size = "0px"

        # p_to.yaxis.axis_label = 'rad'
        # p_to1.yaxis.axis_label = 'BC'
        # p_to2.yaxis.axis_label = 'WD'
        # p_to3.yaxis.axis_label = 'WS'

    # noinspection PyUnresolvedReferences
    def bkapp(doc: bk.document):
        # doc.add_periodic_callback(_save_dic, 3 * 1e3)
        doc.add_root(g_plot)
        # doc.add_root(central_column)

    bkp.show(bkapp, notebook_url=remote_jupyter_proxy_url)


def get_map_fig(h2, wl, xdim, ydim):
    p_map = bkp.figure(width=wl, height=h2 * 4, x_range=(0, xdim), y_range=(0, ydim))
    # p_map.line([1, 2], [1, 2])
    p_map.match_aspect = True
    set_wheel(p_map)
    p_map.toolbar_location = 'below'
    return p_map


# show(bkapp, notebook_url=remote_jupyter_proxy_url)

def add_mouse_event_map(path, rgb, p):
    # print('sd')
    import time

    from bokeh.events import PointEvent

    tms = [time.time(), None]

    pol = [None, None]
    working = [False]

    def on_mouse_move(event: PointEvent):
        # print('sdf')
        x = event.x
        pol[1] = get_path(path, x)

        tms[1] = time.time()
        b1 = (tms[1] - tms[0]) > 1
        b2 = pol[0] != pol[1]
        b3 = working[0] is False

        if (b1 and b2) and b3:
            working[0] = True
            # t1 = time.time()
            # print(p2)
            pol[0] = pol[1]
            tms[0] = tms[1]
            # noinspection PyBroadException
            try:
                img = get_fig(pol[1])[0]
                rgb.data_source.data['image'] = [img]
            except:
                pass
            working[0] = False
            # t2 = time.time()
            # print(t2-t1)

    p.on_event("mousemove", on_mouse_move)
    # p.on_event('mousemove',lambda x: print(3))


def get_path(path, date_ms):
    # import os
    d = pd.to_datetime(date_ms, unit='ms')
    ds = d.strftime('%Y-%m-%d %H.png')
    p = os.path.join(path, ds)
    return p


def get_fig(p):
    from PIL import Image
    f = Image.open(p)
    f.thumbnail([200] * 2)
    # Open image, and make sure it's RGB*A*
    lena_img = f.convert('RGBA')
    xdim, ydim = lena_img.size
    # print("Dimensions: ({xdim}, {ydim})".format(**locals()))
    # Create an array representation for the image `img`, and an 8-bit "4
    # layer/RGBA" version of it `view`.
    img = np.empty((ydim, xdim), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape((ydim, xdim, 4))
    # Copy the RGBA image into view, flipping it so it comes right-side up
    # with a lower-left origin
    view[:, :, :] = np.flipud(np.asarray(lena_img))
    return img, xdim, ydim
