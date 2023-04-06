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

for fake_imports in [1]:
    from IPython import get_ipython

    # noinspection PyBroadException
    try:
        _magic = get_ipython().run_line_magic
        _magic("load_ext", "autoreload")
        _magic("autoreload", "2")
    except:
        pass

    import datetime as dt
    import glob
    import os
    import pprint
    import sys
    import warnings

    import bnn_tools.basic_funs as bfu
    import bnn_tools.bnn_array
    import bnn_tools.coag_sink as cs
    import bnn_tools.funs as fu
    import matplotlib as mpl
    import matplotlib.colors
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy.optimize
    import seaborn as sns
    import xarray as xr

    warnings.filterwarnings("ignore", category=scipy.optimize.OptimizeWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    import bokeh
    import bokeh.models.tools
    import bokeh.palettes
    from bokeh.layouts import Column, Row, gridplot
    from bokeh.models import ColorBar, ColumnDataSource, CustomJS, LogColorMapper
    from bokeh.models.tools import CrosshairTool, PointDrawTool, PolyEditTool
    from bokeh.models.widgets import DataTable, RangeSlider, TableColumn
    from bokeh.plotting import Figure, figure, output_notebook, show
    from bokeh.models import Range1d, LinearAxis, LogAxis, PolyDrawTool

    from IPython.display import HTML, display

for fake_cons in [1]:
    COLS = bokeh.palettes.Category10[10]


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
    df_bc_chc = xr.open_dataarray("./small_data/BC_chc_v01_giancarlo.nc").to_dataframe()
    return df_bc_chc


def set_wheel(p):
    a = None
    for _t in p.tools:
        if isinstance(_t, bokeh.models.WheelZoomTool):
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
        axis_types = ['linear', 'linear']

    assert len(dfm.columns) <= 2

    for fake_plot1 in [1]:
        p = Figure(x_axis_type="datetime", width=1000, height=400, y_axis_type=axis_types[0])

        col0 = dfm.columns[0]
        p.line(
            dfm.index,
            dfm[col0],
            color=COLS[0],
            # y_range_name=y_range_name
            # legend_label=_a[i],
            # line_width=1,
            # line_dash=_l[i],
        )
        p.y_range.start = dfm[col0].min()
        p.y_range.end = dfm[col0].max()

        p.yaxis.axis_label = col0
        p.yaxis.axis_label_text_color = COLS[0]

    for fake_plot2 in [1]:
        if len(dfm.columns) == 2:
            col1 = dfm.columns[1]
            axs_dic = {'linear': LinearAxis, 'log': LogAxis}
            at2 = axs_dic[axis_types[1]]
            y_range_name = 'foo'
            ax2 = at2(y_range_name=y_range_name)
            _min = dfm[col1].min()
            _max = dfm[col1].max()
            p.extra_y_ranges = {y_range_name: bokeh.models.Range1d(_min,_max)}
            # Adding the second axis to the plot.
            p.add_layout(ax2, 'right')

            p.line(dfm.index, dfm[col1], color=COLS[1], y_range_name=y_range_name)
            ax2.axis_label = col1
            ax2.axis_label_text_color = COLS[1]

    set_wheel(p)
    return p


def plot_bc(bc_df):
    p = figure(x_axis_type="datetime", y_axis_type="log")
    p.line(bc_df.index, bc_df["BC"] + 0.1)
    return p


def plot_wd(dfM):
    p = figure(x_axis_type="datetime")
    p.line(dfM.index, dfM["WD"] / 360 * 12)
    p.line(dfM.index, dfM["WDstdv"] / 360 * 12 * 5, color='red')
    # p.line(dfM.index, dfM["WD"] /360 * 12+12)
    return p


def plot_ws(dfM):
    p = figure(x_axis_type="datetime")
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
    p = Figure(
        x_axis_type="datetime",
        width=1000,
        height=400,
        tools=["pan", "box_zoom", "reset", _w, _c, _cw, "hover"],
        y_axis_type="log",
    )
    p.toolbar.active_scroll = _w
    color_mapper = LogColorMapper(palette="Plasma256", low=vmin, high=vmax)
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
    return cb, p


def add_mouse_event(p, pl_line, pb_line, ds2):
    # print('sd')
    import time

    from bokeh.events import PointEvent

    tms = [time.time(), None]

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


def open_meteo_chc():
    df = pd.read_csv("./small_data/meteo_cumbre.csv")

    df["date"] = pd.to_datetime(df["date"])
    df1 = df.set_index("date")
    return df1


def black_format_dic(dic):
    from black import FileMode, format_str

    str_dic = repr(dic)
    res = format_str(str_dic, mode=FileMode())
    return res


def open_dic(file):
    with open(file, "r") as file:
        dic = eval(file.read())
    return dic


def save_dic(*, dic, source, color_mapper, xy, file):
    mask_df = source.to_df().to_dict('list')
    color_high = color_mapper.high
    color_low = color_mapper.low
    xy = xy.tolist()

    dic["mask_df"] = mask_df
    dic["color_high"] = color_high
    dic["color_low"] = color_low
    dic["xy"] = xy

    dic_str = black_format_dic(dic)

    with open(file, "w") as f:
        f.write(dic_str)
    return dic


for fake_vars in [1]:
    # d1 = "2018-05-25"
    # d2 = "2018-05-26"
    # df = open_rad_file()
    # dfm = df.loc[d1:d2]
    # bc = open_bc_chc_file()[d1:d2]
    # ds = open_ds_psd(d1, d2)
    # dfM = open_meteo_chc()[d1:d2]
    pass


# for fake_bokeh in [1]:
def bokeh_gr_gui(init_file, *, _open_rad_df, _open_wind_dir_df, _open_psd_ds, _open_bc_df, _open_wind_spd_df):
    output_notebook()
    # def _m_plot():
    dic = open_dic(init_file)
    d1 = pd.to_datetime(dic['date'])
    d2 = d1 + pd.Timedelta(36, 'hour')

    for fake_open_dfs in [1]:
        # this should return df/psd with columns for each line with a max of 2 cols
        df_rad0 = _open_rad_df(d1, d2)
        df_win_dir2 = _open_wind_dir_df(d1, d2)
        df_win_spd3 = _open_wind_spd_df(d1, d2)
        ds_psd = _open_psd_ds(d1, d2)
        df_bc1 = _open_bc_df(d1, d2)

    for fake_cons in [1]:
        w = 500
        h = 400
        h1 = 150
        h2 = 75
        w1 = 150
        pad = 60
        pad_bo = 40

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
        # add_bc_del(p_to)

        p_le = figure(y_axis_type="log", x_axis_type="log")
        p_le.xaxis.major_label_orientation = np.pi / 4

        p_bo = figure(x_axis_type="datetime", y_axis_type="log")

        # p_to.legend.visible = False

    for fake_table in [1]:
        source = ColumnDataSource(
            {'xs': [], 'ys': [], 'txt': [], 'yd': [], 'xd': []})

        columns = [
            TableColumn(field="txt", title="txt"),
            TableColumn(field="xs", title="xs"),
            TableColumn(field="ys", title="ys"),
            # TableColumn(field="w", title="w"),
            # TableColumn(field="h", title="h"),
        ]

        data_table = DataTable(
            source=source, columns=columns, width=400, height=600, editable=True
        )

    for fake_interaction in [1]:
        p_ce.multi_polygons()
        p1 = p_ce.patches("xs", "ys", line_width=2, alpha=0.4, source=source)
        draw_tool_p1 = PolyDrawTool(renderers=[p1])
        p_ce.add_tools(draw_tool_p1)




        # def _draw_grs(d1, d2, p_ce, gr, c):
        #     def from_time2msec(dts):
        #         s1 = dts - np.datetime64(0, 'Y')
        #         s2 = s1 / np.timedelta64(1, 'ms')
        #         # o = o.assign_coords({'secs': s2})
        #         return s2
        #
        #     dates = pd.date_range(d1, d2, freq='5H')
        #     mts = from_time2msec(dates)
        #     for mt in mts:
        #         p_circle = p_ce.line([], [], color=c)
        #         plot_gr_guide(mt, gr, p_circle)
        #
        # for gr, c in zip([2, 5, 10], ['red', 'blue', 'green']):
        #     _draw_grs(d1, d2, p_ce, gr, c)

    for fake_set_crosssections in [1]:
        p_le_line = p_le.line([], [])
        p_bo_line = p_bo.line([], [])
        add_mouse_event(p_ce, p_le_line, p_bo_line, ds_psd)

    for fake_tools in [1]:
        p_to.add_tools(p_ce.select_one({"name": "crosshair"}))
        p_to1.add_tools(p_ce.select_one({"name": "crosshair"}))
        p_to2.add_tools(p_ce.select_one({"name": "crosshair"}))
        p_to3.add_tools(p_ce.select_one({"name": "crosshair"}))

        p_bo.add_tools(p_ce.select_one({"name": "crosshair"}))
        p_le.add_tools(p_ce.select_one({"name": "crosshairw"}))

        p_ce.toolbar.active_inspect = [
            p_ce.select_one({"name": "crosshair"}),
            p_ce.select_one({"name": "crosshairw"}),
        ]
        set_wheel(p_to1)
        set_wheel(p_to2)
        set_wheel(p_to3)

        p_to.toolbar_location = None
        p_to1.toolbar_location = None
        p_to2.toolbar_location = None
        p_to3.toolbar_location = None
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
        p_bo.width = w
        p_bo.height = h2

    for fake_set_pads in [1]:
        p_le.min_border_bottom = pad_bo
        p_to.min_border_left = pad
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
        p_le.y_range = p_ce.y_range


    for fake_set_dic in [1]:
        if dic['mask_df'] != {}:
            source.data = dic['mask_df']
        color_mapper.high = dic['color_high']
        color_mapper.low = dic['color_low']
        p_cb.y_range.end = dic['color_high']
        p_cb.y_range.start = dic['color_low']

    for fake_save in [1]:
        xy = np.array([])
        def _save_dic3(attr, old, new):
            # print('saving')
            save_dic(dic=dic, source=source, color_mapper=color_mapper, xy=xy, file=init_file)
        def _save_dic():
            # print('saving')
            save_dic(dic=dic, source=source, color_mapper=color_mapper, xy=xy, file=init_file)

        source.on_change('data', _save_dic3)


    g_plot = gridplot(
        [
            [None, p_to3, None],
            [None, p_to2, None],
            [None, p_to1, None],
            [None, p_to, None],
            [p_le, p_ce, p_cb],
            [None, p_bo, None],
        ],
        merge_tools=False,
    )

    for fake_add_touchs in [1]:
        p_to.xaxis.major_label_text_font_size = "0px"
        p_to1.xaxis.major_label_text_font_size = "0px"
        p_to2.xaxis.major_label_text_font_size = "0px"
        p_to3.xaxis.major_label_text_font_size = "0px"
        p_bo.xaxis.major_label_text_font_size = "0px"

        # p_to.yaxis.axis_label = 'rad'
        # p_to1.yaxis.axis_label = 'BC'
        # p_to2.yaxis.axis_label = 'WD'
        # p_to3.yaxis.axis_label = 'WS'


    def bkapp(doc:bokeh.document):
        doc.add_periodic_callback(_save_dic,3*1e3)
        doc.add_root(Row(g_plot, data_table))

    show(bkapp, notebook_url=remote_jupyter_proxy_url)

# show(bkapp, notebook_url=remote_jupyter_proxy_url)
