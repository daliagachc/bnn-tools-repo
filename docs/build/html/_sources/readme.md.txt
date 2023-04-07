# Intro
The bnn_tools package provides functions for analyzing and plotting 
atmospheric aerosol particle size distributions.

The package has the following structure:
- bnn_tools
    - basic_funs.py  
      - a module in the bnn_tools package that contains functions for 
      analyzing and plotting atmospheric aerosol particle size distribution
    - bnn_array.py  
      - an accessor to some of the functions in basic_funs that can be 
      applied directly to xarray DataArrays/Sets.
    - coag_sink.py
    - funs.py
    - funs_bokeh_flx.py
    - gr_gui.py


## basic_funs.py
this is a summary of the functions in the module
- `format_ticks(ax)`: Format ticks on the x-axis of a matplotlib plot with dates.
- `format_ticks2(ax,M,m)`: Format the x-axis of a matplotlib plot to show dates and ticks.
- `from_time2sec(o)`: Convert time to seconds since 1970-01-01.
- `from_sec2time(o)`: Convert seconds since 1970-01-01 to time.
- `from_Dp2lDp(o)`: Convert particle diameter to log10(particle diameter).
- `from_lDp2Dp(o)`: Convert log10(particle diameter) to particle diameter.
- `from_lDp2dlDp(o)`: Calculate the interval between log10(particle diameters).
- `from_Dp2dDp(o)`: Convert particle diameter to the difference in particle diameters for consecutive bins.
- `set_time(o)`: Set the time as either coordinate or dimension depending on the current form.
- `set_Dp(o)`: Set the Dp as either coordinate or dimension depending on the current form.
- `set_lDp(o)`: Set the lDp as either coordinate or dimension depending on the current form.
- `set_sec(o)`: Set the sec as either coordinate or dimension depending on the current form.
- `plot_psd(o, **kwargs)`: Plot particle size distribution with some optional arguments.
- `get_dN(o, d1, d2)`: Calculate the differential number concentration between d1 and d2 meters.
- `get_N(o, d1, d2)`: Calculate the total number concentration between d1 and d2 meters.
- `resample_ts(o, dt_secs)`: Resamples a time series with a given time step. It assumes that the time stamp is centered aligned. The result is also center aligned.
- `upsample_ts(o, dt)`: Upsamples a time series with a given time step.
- `dp_regrid(da, *, n_subs, log_dy)`: Regrids the diameter distribution of
  an aerosol particle dataset linearly over a logarithmic diameter range.
  The number of subintervals per bin is controlled by `n_subs`--the more the
  better accuracy and more computation power needed--and the size of
  the interval in logarithmic diameter is controlled by `log_dy`.
- `get_exact_N(dc1, Dp_min, Dp_max)`: Counts the exact number of particles in the range Dp_min to Dp_max using linear integration.
- `bokeh_plot_psd(ds,width)`: Plots particle size distribution using Bokeh plotting library.
- `basic_bokeh_plot(ds, width)`: Creates a basic Bokeh plot.
- `get_colorbar(color_mapper, p, vmax, vmin)`: Creates a color bar for a Bokeh plot.
- `remote_jupyter_proxy_url(port)`: Configures Bokeh's show method when a proxy must be configured.

## bnn_array.py
this is a summary of the functions in the module
- `dp_regrid`: Regrids the diameter distribution of an aerosol particle dataset linearly over a logarithmic diameter range.
- `from_time2sec`: Converts time to seconds since 1970-01-01.
- `from_sec2time`: Converts seconds since 1970-01-01 to time.
- `from_Dp2lDp`: Converts particle diameter to log10(particle diameter).
- `from_lDp2Dp`: Converts log10(particle diameter) to particle diameter.
- `from_lDp2dlDp`: Calculates the interval between log10(particle diameters).
- `from_Dp2dDp`: Converts particle diameter to the difference in particle diameters for consecutive bins.
- `set_time`: Sets the time as either coordinate or dimension depending on the current form.
- `set_Dp`: Sets the Dp as either coordinate or dimension depending on the current form.
- `set_lDp`: Sets the lDp as either coordinate or dimension depending on the current form.
- `set_sec`: Sets the sec as either coordinate or dimension depending on the current form.
- `plot_psd`: Plots particle size distribution with some optional arguments.
- `get_dN`: Calculates the differential number concentration between d1 and d2 meters.
- `get_exact_N`: Counts the exact number of particles in the range Dp_min to Dp_max using linear integration.
- `get_N`: Calculates the total number concentration between d1 and d2 meters.
- `resample_ts`: Resamples a time series with a given time step. It assumes that the time stamp is center aligned. The result is also center aligned.
- `upsample_ts`: Upsamples a time series with a given time step.
- `u`: Sets the units metadata attribute of an xarray DataArray object.
- `ln`: Sets the long_name metadata attribute of an xarray DataArray object.

## connection between bnn_array and basic_funs
- `bnn_tools.basic_funs` and `bnn_array` are two modules in the `bnn_tools` package for analyzing atmospheric aerosol particle size distributions.
- They have similar functions such as converting between different units of particle diameter and time, setting coordinates, and plotting particle size distribution.
    - in fact bnn_array is just an accesor to bnn_tools basic funs.
        - so that the functions from basic_funs can directly be aplied to xr.
          DataArrays/sets with the following notation:
            ```py
            array.bnn.function()    
            ```
