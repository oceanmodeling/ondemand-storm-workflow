"""
Dynamic map hindcast implementation 

###############################################################
# Original development from https://github.com/ocefpaf/python_hurricane_gis_map
# # Exploring the NHC GIS Data
#
# This notebook aims to demonstrate how to create a simple interactive GIS map with the National Hurricane Center predictions [1] and CO-OPS [2] observations along the Hurricane's path.
#
#
# 1. http://www.nhc.noaa.gov/gis/
# 2. https://opendap.co-ops.nos.noaa.gov/ioos-dif-sos/
#
#
# NHC codes storms are coded with 8 letter names:
# - 2 char for region `al` &rarr; Atlantic
# - 2 char for number `11` is Irma
# - and 4 char for year, `2017`
#
# Browse http://www.nhc.noaa.gov/gis/archive_wsurge.php?year=2017 to find other hurricanes code.
###############################################################
"""

__author__ = 'Saeed Moghimi'
__copyright__ = 'Copyright 2020, UCAR/NOAA'
__license__ = 'GPL'
__version__ = '1.0'
__email__ = 'moghimis@gmail.com'

import argparse
import logging
import os
import sys
import pathlib
import warnings
from glob import glob
from datetime import datetime, timedelta, timezone
from importlib import resources

import numpy as np
import pandas as pd
import arrow
import f90nml
from bokeh.resources import CDN
from bokeh.plotting import figure
from bokeh.models import Title
from bokeh.embed import file_html
from bokeh.models import Range1d, HoverTool
from branca.element import IFrame
import folium
from folium.plugins import Fullscreen, MarkerCluster, MousePosition
import netCDF4
import matplotlib as mpl
import matplotlib.tri as Tri
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, box
from geopandas import GeoDataFrame
import geopandas as gpd
from pyschism.mesh import Hgrid
import cfunits
from retrying import retry
from searvey import coops

import defn as defn
import hurricane_funcs as hurr_f

_logger = logging.getLogger()
mpl.use('Agg')

warnings.filterwarnings("ignore", category=DeprecationWarning)


def ceil_dt(date=datetime.now(), delta=timedelta(minutes=30)):
    """
    Rounds up the input date based on the `delta` time period tolerance

    Examples
    --------
    now = datetime.now()
    print(now)    
    print(ceil_dt(now,timedelta(minutes=30) ))

    """

    date_min = datetime.min
    if date.tzinfo:
        date_min = date_min.replace(tzinfo=date.tzinfo)
    return date + (date_min - date) % delta

@retry(stop_max_attempt_number=5, wait_fixed=3000)
def get_coops(start, end, sos_name, units, bbox, datum='NAVD', verbose=True):
    """
    function to read COOPS data
    We need to retry in case of failure b/c the server cannot handle
    the high traffic during hurricane season.
    """


    coops_stas = coops.coops_stations_within_region(region=box(*bbox))
    # TODO: NAVD 88?
    coops_data = coops.coops_product_within_region(
        'water_level', region=box(*bbox), start_date=start, end_date=end)
    station_names = [
        coops_stas[coops_stas.index == i].name.values[0]
        for i in coops_data.nos_id.astype(int).values
    ]
    staobs_df = coops_data.assign(
            {'station_name': ('nos_id', station_names)}
        ).reset_coords().drop(
            ['f', 's', 'q', 'nws_id']
        ).to_dataframe().reset_index(
            level='nos_id'
        ).rename(
            columns={
                'x': 'lon',
                'y': 'lat',
                'v': 'ssh',
                't': 'time',
                'nos_id': 'station_code',
            }
        ).astype(
            {'station_code': 'int64'}
        )
    staobs_df.index = staobs_df.index.tz_localize(tz=timezone.utc)

    return staobs_df



def make_plot_1line(obs, label=None):
    # TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"
    TOOLS = 'crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,reset,save,'

    p = figure(
        toolbar_location='above',
        x_axis_type='datetime',
        width=defn.width,
        height=defn.height,
        tools=TOOLS,
    )

    if obs.station_code.isna().sum() == 0:
        station_code = obs.station_code.array[0]
        p.add_layout(
            Title(text=f"Station: {station_code}",
                  text_font_style='italic'),
            'above')

    if obs.station_name.isna().sum() == 0:
        station_name = obs.station_name.array[0]
        p.add_layout(
            Title(text=station_name, text_font_size='10pt'),
            'above')

    p.yaxis.axis_label = label

    obs_val = obs.ssh.to_numpy().squeeze()

    l1 = p.line(
        x=obs.index,
        y=obs_val,
        line_width=5,
        line_cap='round',
        line_join='round',
        legend_label='model',
        color='#0000ff',
        alpha=0.7,
    )

    minx = obs.index.min()
    maxx = obs.index.max()

    p.x_range = Range1d(start=minx, end=maxx)

    p.legend.location = 'top_left'

    p.add_tools(HoverTool(tooltips=[('model', '@y'), ], renderers=[l1], ), )
    return p


def make_plot_2line(obs, model=None, label=None, remove_mean_diff=False, bbox_bias=0.0):
    # TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"
    TOOLS = 'crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,reset,save,'

    p = figure(
        toolbar_location='above',
        x_axis_type='datetime',
        width=defn.width,
        height=defn.height,
        tools=TOOLS,
    )

    if obs.station_code.isna().sum() == 0:
        station_code = obs.station_code.array[0]
        p.add_layout(
            Title(text=f"Station: {station_code}",
                  text_font_style='italic'),
            'above')

    if obs.station_name.isna().sum() == 0:
        station_name = obs.station_name.array[0]
        p.add_layout(
            Title(text=station_name, text_font_size='10pt'),
            'above')

    p.yaxis.axis_label = label

    obs_val = obs.ssh.to_numpy().squeeze()

    l1 = p.line(
        x=obs.index,
        y=obs_val,
        line_width=5,
        line_cap='round',
        line_join='round',
        legend_label='obs.',
        color='#0000ff',
        alpha=0.7,
    )

    if model is not None:
        mod_val = model.ssh.to_numpy().squeeze()

        if ('SSH' in label) and remove_mean_diff:
            mod_val = mod_val + obs_val.mean() - mod_val.mean()

        if ('SSH' in label) and bbox_bias is not None:
            mod_val = mod_val + bbox_bias

        l0 = p.line(
            x=model.index,
            y=mod_val,
            line_width=5,
            line_cap='round',
            line_join='round',
            legend_label='model',
            color='#9900cc',
            alpha=0.7,
        )


        minx = max(model.index.min(), obs.index.min())
        maxx = min(model.index.max(), obs.index.max())

        minx = model.index.min()
        maxx = model.index.max()
    else:
        minx = obs.index.min()
        maxx = obs.index.max()

    p.x_range = Range1d(start=minx, end=maxx)

    p.legend.location = 'top_left'

    p.add_tools(
        HoverTool(tooltips=[('model', '@y'), ], renderers=[l0], ),
        HoverTool(tooltips=[('obs', '@y'), ], renderers=[l1], ),
    )

    return p


#################
def make_marker(p, location, fname, color='green', icon='stats'):
    html = file_html(p, CDN, fname)
    # iframe = IFrame(html , width=defn.width+45+defn.height, height=defn.height+80)
    iframe = IFrame(html, width=defn.width * 1.1, height=defn.height * 1.2)
    # popup = folium.Popup(iframe, max_width=2650+defn.height)
    popup = folium.Popup(iframe)
    iconm = folium.Icon(color=color, icon=icon)
    marker = folium.Marker(location=location, popup=popup, icon=iconm)
    return marker


###############################
def read_max_water_level_file(fgrd='hgrid.gr3', felev='maxelev.gr3', cutoff=True):
    
    hgrid = Hgrid.open(fgrd, crs='EPSG:4326')
    h = -hgrid.values
    bbox = hgrid.get_bbox('EPSG:4326', output_type='bbox')

    elev = Hgrid.open(felev, crs='EPSG:4326')
    mzeta = -elev.values
    D = mzeta

    #Mask dry nodes
    NP = len(mzeta)
    idxs = np.where(h < 0)
    D[idxs] = np.maximum(0, mzeta[idxs]+h[idxs])

    idry = np.zeros(NP)
    idxs = np.where(mzeta+h <= 1e-6)
    idry[idxs] = 1

    MinVal = np.min(mzeta)
    MaxVal = np.max(mzeta)
    NumLevels = 21

    if cutoff:
        MinVal = max(MinVal, 0.0)
        MaxVal = min(MaxVal, 2.4)
        NumLevels = 12
    _logger.info(f'MinVal is {MinVal}')
    _logger.info(f'MaxVal is {MaxVal}')

    step = 0.2  # m
    levels = np.arange(MinVal, MaxVal + step, step=step)
    _logger.info(f'levels is {levels}')

    fig, ax = plt.subplots()
    tri = elev.triangulation
    mask = np.any(np.where(idry[tri.triangles], True, False), axis=1)
    tri.set_mask(mask)

    contour = ax.tricontourf(
            tri,
            mzeta,
            vmin=MinVal,
            vmax=MaxVal,
            levels=levels,
            cmap=defn.my_cmap,
            extend='max')

    return contour, MinVal, MaxVal, levels


#############################################################
def contourf_to_geodataframe(contour_obj):

    """Transform a `matplotlib.contour.ContourSet` to a GeoDataFrame"""

    polygons, colors = [], []
    for i, polygon in enumerate(contour_obj.collections):
        mpoly = []
        for path in polygon.get_paths():
            try:
                path.should_simplify = False
                poly = path.to_polygons()
                # Each polygon should contain an exterior ring + maybe hole(s):
                exterior, holes = [], []
                if len(poly) > 0 and len(poly[0]) > 3:
                    # The first of the list is the exterior ring :
                    exterior = poly[0]
                    # Other(s) are hole(s):
                    if len(poly) > 1:
                        holes = [h for h in poly[1:] if len(h) > 3]
                mpoly.append(Polygon(exterior, holes))
            except:
                _logger.warning('Warning: Geometry error when making polygon #{}'.format(i))
        if len(mpoly) > 1:
            mpoly = MultiPolygon(mpoly)
            polygons.append(mpoly)
            colors.append(polygon.get_facecolor().tolist()[0])
        elif len(mpoly) == 1:
            polygons.append(mpoly[0])
            colors.append(polygon.get_facecolor().tolist()[0])
    return GeoDataFrame(geometry=polygons, data={'RGBA': colors}, crs={'init': 'epsg:4326'})


#################
def convert_to_hex(rgba_color):
    red = str(hex(int(rgba_color[0] * 255)))[2:].capitalize()
    green = str(hex(int(rgba_color[1] * 255)))[2:].capitalize()
    blue = str(hex(int(rgba_color[2] * 255)))[2:].capitalize()

    if blue == '0':
        blue = '00'
    if red == '0':
        red = '00'
    if green == '0':
        green = '00'

    return '#' + red + green + blue


#################
def get_model_station_ssh(sim_date, sta_in_file, sta_out_file, stations_info):
    """Read model ssh"""

    station_dist_tolerance = 0.0001 # degrees

    # Get rid of time zone and convert to string "<YYYY>-<MM>-<DD>T<HH>"
    sim_date_str = sim_date.astimezone(timezone.utc).strftime('%Y-%m-%dT%H')


    #Read model output
    sta_data = np.loadtxt(sta_out_file)
    time_deltas = sta_data[:, 0].ravel().astype('timedelta64[s]')
    sta_date = pd.DatetimeIndex(
            data=np.datetime64(sim_date_str) + time_deltas,
            tz=timezone.utc,
            name="date_time")

    sta_zeta = sta_data[:, 1:]

    _logger.debug(len(sta_zeta[:,1]))
    _logger.debug(type(sta_date))
    _logger.debug(type(sta_zeta))

    df_staout = pd.DataFrame(data=sta_zeta, index=sta_date)
    df_staout_melt = df_staout.melt(
            ignore_index=False,
            value_name="ssh",
            var_name="staout_index")

    df_stain = pd.read_csv(
            sta_in_file,
            sep=' ',
            header=None,
            skiprows=2,
            usecols=[1, 2],
            names=["lon", "lat"])

    df_stasim = df_staout_melt.merge(
            df_stain, left_on='staout_index', right_index=True)

    gdf_stasim = gpd.GeoDataFrame(
            df_stasim,
            geometry=gpd.points_from_xy(df_stasim.lon, df_stasim.lat))

    gdf_sta_info = gpd.GeoDataFrame(
            stations_info,
            geometry=gpd.points_from_xy(
                stations_info.lon, stations_info.lat))

    gdf_staout_w_info = gpd.sjoin_nearest(
            gdf_stasim,
            gdf_sta_info.drop(columns=['lon','lat']),
            lsuffix='staout', rsuffix='real_station',
            max_distance=station_dist_tolerance)

    # Now go back to DF or keep GDF and remove lon lat columns?
    df_staout_w_info = pd.DataFrame(gdf_staout_w_info.drop(
            columns=['geometry']))
    df_staout_w_info = df_staout_w_info.rename(
            columns={'nos_id': 'station_code'}
        ).astype(
            {'station_code': 'int64'}
        )

    # TODO: Reset index or keep date as index?
#    df_staout_w_info['date_time'] = df_staout_w_info.index
#    df_staout_w_info = df_staout_w_info.reset_index(drop=True)


    return df_staout_w_info



################
def get_storm_bbox(cone_gdf_list, pos_gdf_list, bbox_from_track=True):
    # Find the bounding box to search the data.
    last_cone = cone_gdf_list[-1]['geometry'].iloc[0]
    track = LineString([point['geometry'] for point in pos_gdf_list])
    if bbox_from_track:
        track_lons = track.coords.xy[0]
        track_lats = track.coords.xy[1]
        bbox = (
            min(track_lons) - 2, min(track_lats) - 2,
            max(track_lons) + 2, max(track_lats) + 2,
        )
    else:
        bounds = np.array([last_cone.buffer(2).bounds, track.buffer(2).bounds]).reshape(4, 2)
        lons, lats = bounds[:, 0], bounds[:, 1]
        bbox = lons.min(), lats.min(), lons.max(), lats.max()

    return bbox


def get_storm_dates(pos_gdf_list):
    # Ignoring the timezone, like AST (Atlantic Time Standard) b/c
    # those are not a unique identifiers and we cannot disambiguate.

    if 'FLDATELBL' in pos_gdf_list[0].keys():
        start = pos_gdf_list[0]['FLDATELBL']
        end = pos_gdf_list[-1]['FLDATELBL']
        date_format = 'YYYY-MM-DD h:mm A ddd'

    elif 'ADVDATE' in pos_gdf_list[0].keys():
        # older versions (e.g. IKE)
        start = pos_gdf_list[0]['ADVDATE']
        end = pos_gdf_list[-1]['ADVDATE']
        date_format = 'YYMMDD/hhmm'

    else:
        msg = 'Check for correct time stamp and adapt the code !'
        _logger.error(msg)
        raise ValueError(msg)

    beg_date = arrow.get(start, date_format).datetime
    end_date = arrow.get(end, date_format).datetime

    return beg_date, end_date


def get_stations_info(bbox):

    # Not using static local file anymore!
    # We should get the same stations we use for observation
#    df = coops.coops_stations()
    df = coops.coops_stations_within_region(region=box(*bbox))
    stations_info = df.assign(
            lon=df.geometry.apply('x'),
            lat=df.geometry.apply('y'),
        ).reset_index().rename(
            {'name': 'station_name', 'nos_id': 'station_code'}
        )

    # Some stations are duplicate with different NOS ID but the same NWS ID
    stations_info = stations_info.drop_duplicates(subset=['nws_id'])
    stations_info = stations_info[stations_info.nws_id != '']

    
    return stations_info


def get_adjusted_times_for_station_outputs(staout_df_w_info, freq):

    # Round up datetime smaller than 30 minutes
    start_date = ceil_dt(staout_df_w_info.index.min().to_pydatetime())
    end_date = ceil_dt(staout_df_w_info.index.max().to_pydatetime())

    new_index_dates = pd.date_range(
        start=start_date.replace(tzinfo=None),
        end=end_date.replace(tzinfo=None),
        freq=freq,
        tz=start_date.tzinfo
    )

    return new_index_dates


def adjust_stations_time_and_data(time_indexed_df, freq, groupby):

    new_index_dates = get_adjusted_times_for_station_outputs(
            time_indexed_df, freq)

    
    adj_staout_df = pd.concat(
        df.reindex(
            index=new_index_dates, limit=1, method='nearest').drop_duplicates()
        for idx, df in time_indexed_df.groupby(by=groupby))
    adj_staout_df.loc[np.abs(adj_staout_df['ssh']) > 10, 'ssh'] = np.nan
    adj_staout_df = adj_staout_df[adj_staout_df.ssh.notna()]

    return adj_staout_df


def get_esri_url(layer_name):

    pos = 'MapServer/tile/{z}/{y}/{x}'
    base = 'http://services.arcgisonline.com/arcgis/rest/services'
    layer_info = dict(
        Imagery='World_Imagery/MapServer',
        Ocean_Base='Ocean/World_Ocean_Base',
        Topo_Map='World_Topo_Map/MapServer',
        Physical_Map='World_Physical_Map/MapServer',
        Terrain_Base='World_Terrain_Base/MapServer',
        NatGeo_World_Map='NatGeo_World_Map/MapServer',
        Shaded_Relief='World_Shaded_Relief/MapServer',
        Ocean_Reference='Ocean/World_Ocean_Reference',
        Navigation_Charts='Specialty/World_Navigation_Charts',
        Street_Map='World_Street_Map/MapServer'
    )

    layer = layer_info.get(layer_name)
    if layer is None:
        layer = layer_info['Imagery']

    url = f'{base}/{layer}/{pos}'
    return url


def folium_create_base_map(bbox_str, layer_name_list=None):

    # Here is the final result. Explore the map by clicking on
    # the map features plotted!
    bbox_ary = np.fromstring(bbox_str, sep=',')
    lon = 0.5 * (bbox_ary[0] + bbox_ary[2])
    lat = 0.5 * (bbox_ary[1] + bbox_ary[3])

    m = folium.Map(
            location=[lat, lon],
            tiles='OpenStreetMap',
            zoom_start=4, control_scale=True)
    Fullscreen(position='topright', force_separate_button=True).add_to(m)

    if layer_name_list is None:
        return m

    for lyr_nm in layer_name_list:
        url = get_esri_url(lyr_nm)

        lyr = folium.TileLayer(tiles=url, name=lyr_nm, attr='ESRI', overlay=False)
        lyr.add_to(m)

    return m


def folium_add_max_water_level_contour(map_obj, max_water_level_contours_gdf, MinVal, MaxVal):
    ## Get colors in Hex
    colors_elev = []
    for i in range(len(max_water_level_contours_gdf)):
        color = defn.my_cmap(i / len(max_water_level_contours_gdf))
        colors_elev.append(mpl.colors.to_hex(color))

    # assign to geopandas obj
    max_water_level_contours_gdf['RGBA'] = colors_elev

    # plot geopandas obj
    maxele = folium.GeoJson(
        max_water_level_contours_gdf,
        name='Maximum water level [m above MSL]',
        style_function=lambda feature: {
            'fillColor': feature['properties']['RGBA'],
            'color': feature['properties']['RGBA'],
            'weight': 1.0,
            'fillOpacity': 0.6,
            'line_opacity': 0.6,
        },
    )

    maxele.add_to(map_obj)

    # Add colorbar
    color_scale = folium.StepColormap(
        colors_elev,
        # index=color_domain,
        vmin=MinVal,
        vmax=MaxVal,
        caption='Maximum water level [m above MSL]',
    )
    map_obj.add_child(color_scale)


def folium_add_ssh_time_series(map_obj, staout_df_w_info, obs_df=None):
#    marker_cluster_estofs_ssh = MarkerCluster(name='CO-OPS SSH observations')
    marker_cluster_estofs_ssh = MarkerCluster(name='Simulation SSH [m above MSL]')

    _logger.info('      > plot model only')


    by = ["staout_index", "station_code"]
    for (staout_idx, st_code), df in staout_df_w_info.groupby(by=by):
        fname = df.station_code.array[0]
        location = df.lat.array[0], df.lon.array[0]
        if st_code is None or obs_df is None:
            p = make_plot_1line(df, label='SSH [m above MSL]')
        else:
            p = make_plot_2line(
                    obs=obs_df[obs_df.station_code == st_code],
                    remove_mean_diff=True,
                    model=df, label='SSH [m]')
        marker = make_marker(p, location=location, fname=fname)
        marker.add_to(marker_cluster_estofs_ssh)

    marker_cluster_estofs_ssh.add_to(map_obj)


def folium_add_bbox(map_obj, bbox_str):
    ## Plotting bounding box
    bbox_ary = np.fromstring(bbox_str, sep=',')
    p = folium.PolyLine(get_coordinates(bbox_ary), color='#009933', weight=2, opacity=0.6)

    p.add_to(map_obj)


def folium_add_storm_latest_cone(map_obj, cone_gdf_list, pos_gdf_list):
    latest_cone_style = {
            'fillOpacity': 0.1,
            'color': 'red',
            'stroke': 1,
            'weight': 1.5,
            'opacity': 0.8,
        }
    # Latest cone prediction.
    latest = cone_gdf_list[-1]
    ###
    if 'FLDATELBL' in pos_gdf_list[0].keys():  # Newer storms have this information
        names3 = 'Cone prediction as of {}'.format(latest['ADVDATE'].values[0])
    else:
        names3 = 'Cone prediction'
    ###
    folium.GeoJson(
        data=latest.__geo_interface__,
        name=names3,
        style_function=lambda feat: latest_cone_style,
    ).add_to(map_obj)


def folium_add_storm_all_cones(
        map_obj,
        cone_gdf_list,
        pos_gdf_list,
        track_radius,
        storm_name,
        storm_year
        ):
    cone_style = {
            'fillOpacity': 0,
            'color': 'lightblue',
            'stroke': 1,
            'weight': 0.3,
            'opacity': 0.3,
        }
    marker_cluster1 = MarkerCluster(name='NHC cone predictions')
    marker_cluster1.add_to(map_obj)
    if 'FLDATELBL' not in pos_gdf_list[0].keys():  # Newer storms have this information
        names3 = 'Cone prediction'

        # Past cone predictions.
    for cone in cone_gdf_list[:-1]:
        folium.GeoJson(
            data=cone.__geo_interface__, style_function=lambda feat: cone_style,
        ).add_to(marker_cluster1)

    # Latest points prediction.
    for k, row in last_pts.iterrows():

        if 'FLDATELBL' in pos_gdf_list[0].keys():  # Newer storms have this information
            date = row['FLDATELBL']
            hclass = row['TCDVLP']
            popup = '{}<br>{}'.format(date, hclass)
            if 'tropical' in hclass.lower():
                hclass = 'tropical depression'

            color = defn.colors_hurricane_condition[hclass.lower()]
        else:
            popup = '{}<br>{}'.format(storm_name, storm_year)
            color = defn.colors_hurricane_condition['hurricane']

        location = row['LAT'], row['LON']
        folium.CircleMarker(
            location=location,
            radius=track_radius,
            fill=True,
            color=color,
            popup=popup,
        ).add_to(map_obj)


def folium_add_storm_track(
        map_obj,
        pos_gdf_list,
        track_radius,
        storm_name,
        storm_year
        ):
    # marker_cluster3 = MarkerCluster(name='Track')
    # marker_cluster3.add_to(map_obj)
    for point in pos_gdf_list:
        if 'FLDATELBL' in pos_gdf_list[0].keys():  # Newer storms have this information
            date = point['FLDATELBL']
            hclass = point['TCDVLP']
            popup = defn.template_track_popup.format(
                storm_name, date, hclass)

            if 'tropical' in hclass.lower():
                hclass = 'tropical depression'

            color = defn.colors_hurricane_condition[hclass.lower()]
        else:
            popup = '{}<br>{}'.format(storm_name, storm_year)
            color = defn.colors_hurricane_condition['hurricane']

        location = point['LAT'], point['LON']
        folium.CircleMarker(
            location=location, radius=track_radius, fill=True, color=color, popup=popup,
        ).add_to(map_obj)


def get_schism_date(param_file):

    # Use f90nml to read parameters from the input mirror

    # &OPT
    #  start_year = 2000 !int
    #  start_month = 1 !int
    #  start_day = 1 !int
    #  start_hour = 0 !double
    #  utc_start = 8 !double
    # /
    params = f90nml.read(str(param_file))

    opt = params.get('opt', {})

    year = opt.get('start_year', 2000)
    month = opt.get('start_month', 1)
    day = opt.get('start_day', 1)
    hour = int(opt.get('start_hour', 0))
    tz_rel_utc = opt.get('utc_start', 8)

    sim_tz = timezone(timedelta(hours=tz_rel_utc))
    sim_date =  datetime(year, month, day, hour, tzinfo=sim_tz)

    return sim_date


def folium_finalize_map(map_obj, storm_name, storm_year, Date, FCT):
    html = map_obj.get_root().html
    if storm_name and storm_year:
        html.add_child(folium.Element(
            defn.template_storm_info.format(storm_name,storm_year)))

    html.add_child(folium.Element(defn.template_fct_info.format(Date, FCT)))
    html.add_child(folium.Element(defn.disclaimer))

    folium.LayerControl().add_to(map_obj)
    MousePosition().add_to(map_obj)


def main(args):

    schism_dir = args.schismdir
    storm_name = args.name
    storm_year = args.year

    storm_tag = f"{storm_name.upper()}_{storm_year}"
    grid_file = schism_dir / "hgrid.gr3"

    draw_bbox = False
    plot_cones = True
    plot_latest_cone_only = True
    track_radius = 5
    freq = '30min'

    sta_in_file = schism_dir / "station.in"
    if not sta_in_file.exists():
        _logger.warning('Stations input file is not found!')
        sta_in_file = None

    results_dir = schism_dir / "outputs"
    if not results_dir.exists():
        raise ValueError("Simulation results directory not found!")

    _logger.info(f'results_dir: {str(results_dir)}')

    sta_out_file = results_dir / 'staout_1'
    if not sta_out_file.exists():
        _logger.warning('Points time-series file is not found!')
        sta_out_file = None

    felev = results_dir / 'maxelev.gr3'
    if not felev.exists():
        raise FileNotFoundError('Maximum elevation file is not found!')

    param_file = results_dir / 'param.out.nml'
    if not param_file.exists():
        raise FileNotFoundError('Parameter file not found!')


    if not grid_file.exists():
        raise FileNotFoundError('Grid file not found!')


    post_dir = schism_dir / 'viz'
    if not post_dir.exists():
        post_dir.mkdir(exist_ok=True, parents=True)


    no_sta = False
    if sta_out_file is None or sta_in_file is None:
        # Station in file is needed for lat-lon
        no_sta = True


#####################################################
    sim_date = get_schism_date(param_file)
    Date = sim_date.strftime('%Y%m%d')
    FCT = ceil_dt(sim_date, timedelta(hours=6)).hour


#####################################################
    bbox_str = args.bbox_str
    if storm_tag is not None:

        _logger.info(f' > Read NHC information for {storm_name} {storm_year} ... ')
        ts_code, hurr_prod_tag = hurr_f.get_nhc_storm_info(str(storm_year), storm_name)

        # download gis zip files
        hurr_gis_path = hurr_f.download_nhc_gis_files(hurr_prod_tag, post_dir)

        # get advisory cones and track points
        cone_gdf_list, pos_gdf_list, last_pts = hurr_f.read_advisory_cones_info(
                hurr_prod_tag, hurr_gis_path, str(storm_year), ts_code)

        bbox = get_storm_bbox(cone_gdf_list, pos_gdf_list)
        start_date, end_date = get_storm_dates(pos_gdf_list)

        bbox_str = ', '.join(format(v, '.2f') for v in bbox)
        _logger.info(' > bbox: {}\nstart: {}\n  end: {}'.format(
            bbox_str, start_date, end_date))


#####################################################
    obs_df = None
    if not no_sta:

        stations_info = get_stations_info(bbox)
        staout_df_w_info = get_model_station_ssh(
                sim_date, sta_in_file, sta_out_file, stations_info)
        adj_station_df = adjust_stations_time_and_data(
                staout_df_w_info, freq, "staout_index")


        start_dt = adj_station_df.index.min().to_pydatetime()
        end_dt = adj_station_df.index.max().to_pydatetime()

        all_obs_df = get_coops(
            start=start_dt,
            end=end_dt,
            sos_name='water_surface_height_above_reference_datum',
            units=cfunits.Units('meters'),
            datum = 'MSL',
            bbox=bbox,
        )

        # Get observation from stations that have a corresponding
        # model time history output
        obs_df = all_obs_df[all_obs_df.station_code.isin(
            np.unique(adj_station_df.station_code.to_numpy()))]

        # To get smaller html file
#        obs_df = adjust_stations_time_and_data(
#                obs_df, freq, "station_code")


#####################################################
    _logger.info('  > Put together the final map')
    m = folium_create_base_map(bbox_str, layer_name_list=["Imagery"])


#####################################################
    _logger.info('     > Plot max water elev ..')
    contour, MinVal, MaxVal, levels = read_max_water_level_file(fgrd=grid_file, felev=felev)
    max_water_level_contours_gdf = contourf_to_geodataframe(contour)

    folium_add_max_water_level_contour(m, max_water_level_contours_gdf, MinVal, MaxVal)

#####################################################
    if not no_sta:
        _logger.info('     > Plot SSH stations ..')

        folium_add_ssh_time_series(m, adj_station_df, obs_df)

#####################################################
    if draw_bbox:
        folium_add_bbox(m, bbox_str)

#####################################################
    if storm_tag is not None:
        if plot_cones:
            _logger.info('     > Plot NHC cone predictions')

            if plot_latest_cone_only:
                folium_add_storm_latest_cone(m, cone_gdf_list, pos_gdf_list)
            else:
                folium_add_storm_all_cones(
                        m, cone_gdf_list, pos_gdf_list,
                        track_radius,
                        storm_name, storm_year)

        _logger.info('     > Plot points along the final track ..')
        folium_add_storm_track(
                m, pos_gdf_list, track_radius,
                storm_name, storm_year)


#####################################################
    _logger.info('     > Add disclaimer and storm name ..')
    folium_finalize_map(m, storm_name, storm_year, Date, FCT)

    _logger.info('     > Save file ...')


    fname = os.path.join(post_dir, '{}_{}_{}.html'.format(storm_tag, Date, FCT))
    _logger.info(fname)
    m.save(fname)


def entry():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "name", help="name of the storm", type=str)

    parser.add_argument(
        "year", help="year of the storm", type=int)

    parser.add_argument(
        "schismdir", type=pathlib.Path)

    parser.add_argument('--vdatum', default='MSL')
    parser.add_argument(
            '--bbox-str',
            default='-99.0,5.0,-52.8,46.3',
            help='format: lon_min,lat_min,lon_max,lat_max')

    main(parser.parse_args())

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    entry()
