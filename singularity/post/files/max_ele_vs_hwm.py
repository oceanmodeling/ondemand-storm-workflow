import argparse
import logging
import os
import warnings
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import xarray as xr

from cartopy.feature import NaturalEarthFeature
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap, BoundaryNorm
from matplotlib.dates import DateFormatter
from matplotlib_scalebar.scalebar import ScaleBar
from shapely.geometry.point import Point
from pathlib import Path 
from pyproj import CRS
from pyschism.mesh import Hgrid
from stormevents import StormEvent


def get_storm_track_line(storm_track, advisory):
    for idx in storm_track.linestrings[advisory]:
        xy_line = storm_track.linestrings[advisory][idx].xy
    return xy_line


def get_stations_coordinates(dataset):
    coord_x = dataset.longitude.values
    coord_y = dataset.latitude.values
    coord_combined = np.column_stack([coord_x, coord_y])
    return coord_combined


def find_stations_indices(coordinates, hgridfile):
    longitude = hgridfile.x.T
    latitude = hgridfile.y.T
    long_lat = np.column_stack((longitude.ravel(),latitude.ravel()))
    tree = sp.spatial.cKDTree(long_lat)
    dist,idx = tree.query(coordinates,k=1)
    ind = np.column_stack(np.unravel_index(idx,longitude.shape))
    return [i for i in ind]
    
    
def get_schism_max_elevation(schism_output, station_indices_array):
    max_elev = []
    for idx in station_indices_array:
        tmp = float(schism_output[int(idx)].values) 
        max_elev.append(tmp)
    return max_elev


def find_utm_from_lon(x):
    utm_zone = int(((180+x)/6) + 1)
    return utm_zone


def utm_north_south(y):
    if y<0:
        return True
    else:
        return False
    

def get_epsg_code(utm_zone, is_south):

    epsg_dict = CRS.from_dict({'proj': 'utm', 'zone': utm_zone, 'south': is_south}).to_authority()
    return int(epsg_dict[1])


def calc_distance_meter(x, y):
    utm_zone = find_utm_from_lon(x)
    is_south = utm_north_south(y)
    proj_epsg = get_epsg_code(utm_zone,is_south)
    
    points = gpd.GeoSeries(
        [Point(x-0.5, y), 
         Point(x+0.5, y)], 
        crs=4326)  # Geographic WGS 84 - degrees
    
    points = points.to_crs(proj_epsg)  # Projected WGS 84 - meters
    distance_meters = points[0].distance(points[1])
    
    return distance_meters


def plot_scatter(df, save_dir):
    figure, axis = plt.subplots(1, 1)
    # figure.set_size_inches(12, 12/1.6)

    # df.plot.scatter(x='elev_m', y='max_elev', ax=axis, s=13)
    plt.scatter(x=df['elev_m'], y=df['max_elev'], s=13, facecolors='none', edgecolors='b')
     
    xlim = axis.get_xlim()
    ylim = axis.get_ylim()
    ax_min = min(xlim[0],ylim[0])
    ax_max = max(xlim[1],ylim[1])
    axis.set_xlim([ax_min, ax_max])
    axis.set_ylim([ax_min, ax_max])
    
    b, a = np.polyfit(df['elev_m'], df['max_elev'], 1)
    xline = np.linspace(ax_min+1, ax_max-1, num=10)
    axis.plot(xline, a + b * xline, color="k", lw=1);

    axis.text(0.05, 0.9, f"model = {round(a,2)} + {round(b,2)} * obs.",
              horizontalalignment='left', verticalalignment='center',
              transform=axis.transAxes, size=10)
    
    axis.axline((ax_min,ax_min), (ax_max,ax_max), linestyle='--', color='grey')
    axis.set_xlabel('USGS HWMs [m]')
    axis.set_ylabel('SCHISM max. elevation [m]')
    
    axis.set_title(f'{df.eventName.values[0]}')
    
    plt.savefig(os.path.join(save_dir, f'{df.eventName.values[0]}_scatter_HWM_vs_max_elev.png'))

    
def plot_contour_vs_hwm(storm_track, hwm_points, triangles, max_elev, countries_map, save_dir):
    figure, axis = plt.subplots(1, 1)
    figure.set_size_inches(12, 12/1.6)
    # best_track_plot_date = storm_track.start_date.isoformat().translate({ord(i): None for i in '-:'})
    track_line = get_storm_track_line(storm_track, 'BEST')
    
    # Define the colors for positive and negative values
    positive_color = 'red'
    negative_color = 'blue'    

    # Create a colormap with red for positive and blue for negative values
    man_cmap = LinearSegmentedColormap.from_list('red_blue_colormap', ['white', 'blue', 'red'])
                                                 # [negative_color, 'white', positive_color])
    
    step = 0.025  # m
    MinVal = 0.0 
    MaxVal = 8.0 # hwm_points.height_above_gnd.max() #10.0 
    levels = np.arange(MinVal, MaxVal + step, step=step)
    
    hwm_points.plot('elev_m', 
                    vmin = MinVal,
                    vmax = MaxVal, 
                    markersize = 20,
                    marker='s',
                    edgecolor='black',
                    ax=axis, 
                    cmap=man_cmap,
                    zorder=10)
   
    xlim = axis.get_xlim()
    ylim = axis.get_ylim()    
    
    axis.tricontourf(triangles, 
                     max_elev, 
                     cmap = man_cmap, 
                     vmin = MinVal, 
                     vmax = MaxVal, 
                     levels=levels, 
                     extend='both',
                     zorder=0)
# option 1       
    colorbar = plt.colorbar(
        ScalarMappable(
            norm=Normalize(
                vmin=MinVal, #hwm_points['height_above_gnd'].min(),
                vmax=MaxVal, #hwm_points['height_above_gnd'].max(),
            ),
            cmap=man_cmap,
        ),extend='both', shrink=0.5,
    )
# option 2    
    # bounds = [0, 1, 2, 3, 4, 5, 6]
    # norm = BoundaryNorm(bounds, man_cmap.N, extend='both')
    # colorbar = plt.colorbar(ScalarMappable(norm=norm, cmap=man_cmap), shrink=0.5, extend='both') 

    colorbar.update_ticks()
    colorbar.set_label('[m]', rotation=270, labelpad=15)
    
    # axis.plot(*storm_track.linestrings['BEST'][best_track_plot_date].xy, 
    #           c='red', linestyle='dashed', 
    #           label='BEST track', zorder=5)
    axis.plot(*track_line, c='black', linestyle='dashed', label='BEST track', zorder=15)
    
    countries_map.plot(color='lightgrey', ax=axis, zorder=-1)
        
    axis.set_xlim(xlim)
    axis.set_ylim(ylim)

    axis.legend()
    axis.set_title(f'{storm_track.name} {storm_track.year} SCHISM Max Elevevation vs USGS HWM')
    
    # add scalebar
    distance_meters = calc_distance_meter(np.mean(xlim),np.mean(ylim))
    axis.add_artist(ScaleBar(distance_meters, location='lower right')) 
    
    # add north arrow
    axis.text(0.5, 0.02, u"\u25B2\nN",
              horizontalalignment='center', verticalalignment='bottom',
              transform=axis.transAxes, size=15)
    
    plt.savefig(os.path.join(save_dir, f'{storm_track.name}_{storm_track.year}_HWM_vs_max_elev.png'))


###############################
def main(args):

    name_of_storm = args.storm_name.upper()
    year_of_storm = args.storm_year
    hgrid_directory = Path(args.grid_dir)
    output_directory = Path(args.sim_dir)
    save_directory = args.save_dir      
       
    hgrid_file_path = hgrid_directory / 'hgrid.gr3'
    if not hgrid_file_path.exists():
        raise FileNotFoundError(f'{hgrid_file_path} was not found!')
           
    output_path = output_directory / 'out2d_1.nc'
    if not output_path.exists():
        raise FileNotFoundError(f"{output_path} was not found!")

    print(f"{name_of_storm}_{year_of_storm}")
    
    gdf_countries = gpd.GeoSeries(NaturalEarthFeature(category='physical', scale='10m', name='land').geometries(), crs=4326)

    # Read SCHISM outputs and get maximum elevation
    ds_schism = xr.open_dataset(output_path)
    ds_max_elev = ds_schism['elevation'].max('time')
    
    # Read mesh file and generate triangulations
    hgrid_file = Hgrid.open(hgrid_file_path, crs=4326)
    tri = hgrid_file.triangulation
    mesh_poly = hgrid_file.hull().unary_union

    # Load storm and obtain isobars from stormevent
    storm = StormEvent(name_of_storm, year_of_storm)
    storm_best_track = storm.track()
    
    # obtain HWMs, quality control, and select poins with fair or better quality within the mesh domain
    hwm = storm.flood_event.high_water_marks()
    hwm_qc = hwm[hwm.hwm_quality_id<=3] # To select 1:Excellent, 2:Good, and 3: Fair ; (4:Poor, 5:Very Poor, 6:Unknown) 
    hwm_qc = hwm[hwm.verticalDatumName=='NAVD88'] 
    hwm_sel = hwm_qc[hwm_qc.intersects(mesh_poly)] # To keep points within the domain
    
    df_summary = hwm_sel[['latitude','longitude','eventName','hwm_quality_id','elev_ft','height_above_gnd','geometry']]
    df_summary['elev_m'] = df_summary['elev_ft'] * 0.3048 # convert ft to meter
    # df_summary = df_summary.dropna() # remove NaN HWMs 
    df_filter = df_summary.sort_values(by='elev_m', axis=0, ascending=False).drop_duplicates(subset=['latitude','longitude']) # to keep the highest among duplicates
    df_filter = df_filter[df_filter.elev_m>=0.0] # to filter NaN and negative values
    
    stations_coordinates = get_stations_coordinates(df_filter)
    stations_indices = find_stations_indices(stations_coordinates, hgrid_file) 
    
    df_filter['max_elev'] = get_schism_max_elevation(ds_max_elev, stations_indices)

    plot_contour_vs_hwm(storm_best_track, df_filter, tri, ds_max_elev, gdf_countries, save_directory)
    plot_scatter(df_filter, save_directory)
    

        
def entry():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--storm_name", help="name of the storm", type=str)

    parser.add_argument(
        "--storm_year", help="year of the storm", type=int)
    
    parser.add_argument(
        "--grid_dir", help="path to hgrid.gr3 file", type=str) 
    
    parser.add_argument(
        "--sim_dir", help="Path to SCHISM output directory", type=str)
    
    # optional    
    parser.add_argument(
        "--save_dir", help="directory for saving analysis", default=os.getcwd(), type=str)

    main(parser.parse_args())


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # warnings.filterwarnings("ignore", category=DeprecationWarning)
    entry()