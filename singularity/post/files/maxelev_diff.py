import argparse
import logging

import os
import sys
import time
import warnings
from copy import deepcopy
#from datetime import datetime, timedelta
from pathlib import Path 

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from cartopy.feature import NaturalEarthFeature
#from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LogNorm, LinearSegmentedColormap
#from matplotlib.dates import DateFormatter
#from pyproj import CRS

from pyschism.mesh import Hgrid
#from stormevents import StormEvent

logging.basicConfig(level=logging.DEBUG)
_logger = logging.getLogger()

warnings.filterwarnings("ignore", category=DeprecationWarning)

###############################
def _create_idry(hgrid, schism_welev, schism_welev2, bbox_str):

    """Return dry index"""
    
    # Split the string into a list of substrings using comma as the delimiter
    tmp = bbox_str.split(',')

    # Convert the substrings to floats
    lon_min = float(tmp[0])
    lat_min = float(tmp[1])
    lon_max = float(tmp[2])
    lat_max = float(tmp[3])


    #
    h = -hgrid.values
    D1 = -schism_welev.values
    D2 = -schism_welev2.values

    #
    NP = len(D1)
    idry = np.zeros(NP) + 1
    idxs = np.where((D1+h > 0.01) 
                    & (D2+h > 0.01) 
                    & (hgrid.x < lon_max) 
                    & (hgrid.x > lon_min) 
                    & (hgrid.y < lat_max) 
                    & (hgrid.y > lat_min) 
                    & (h < 25.0))
    idry[idxs] = 0     

    return idry

def _plot_maxelev_diff(storm_tag, hgrid, schism_welev, schism_welev2, idry, bbox_str, **kwargs):

    """plot maxelev diff: model2 - model1"""

    #
    DPI = kwargs.get('DPI', None)
    title_string = kwargs.get('title_string', None)
    save_file_name = kwargs.get('save_file_name', None)

    # Figure dpi resolution:
    if DPI is None:       DPI = 300

    # Split the string into a list of substrings using comma as the delimiter
    tmp = bbox_str.split(',')

    # Convert the substrings to floats
    lon_min = float(tmp[0])
    lat_min = float(tmp[1])
    lon_max = float(tmp[2])
    lat_max = float(tmp[3])

    gdf_countries = gpd.GeoSeries(NaturalEarthFeature(category='physical', 
                    scale='10m', name='land').geometries(), crs=4326)

    #
    h = -hgrid.values
    D1 = -schism_welev.values
    D2 = -schism_welev2.values

    # Define the colors for positive and negative values
    positive_color = 'red'
    negative_color = 'blue'

    # Create a colormap with red for positive and blue for negative values
    cmap = LinearSegmentedColormap.from_list('red_blue_colormap', 
        [negative_color, 'white', positive_color])

    # 
    step = 0.025  # m
    MinVal = -1.0 
    MaxVal = 1.0 
    levels = np.arange(MinVal, MaxVal + step, step=step)
    tri = hgrid.triangulation
    mask = np.any(np.where(idry[tri.triangles], True, False), axis=1)
    tri.set_mask(mask)

    figure, axis = plt.subplots(1, 1)
    figure.set_size_inches(12, 12 / 1.61803398875)


    contour = axis.tricontourf(
                tri,
                D2-D1,
                vmin=MinVal,
                vmax=MaxVal,
                levels=levels,
                cmap=cmap,
                extend='both')


    gdf_countries.plot(color='lightgrey', ax=axis, zorder=-1)

    iselect = np.where((idry==0))
    diff = D2[iselect] - D1[iselect]
    diff = diff*100/D1[iselect]
    mean_percentage_increase = np.nanmean(diff)
    _logger.info(f'mean_percentage_increase: {mean_percentage_increase:.2f}%')
    print(mean_percentage_increase)

    axis.annotate(f'Mean percentage increase: {mean_percentage_increase:.2f}%', 
                  (lon_min+0.4, lat_max-0.25), 
                  textcoords="offset points", xytext=(10,0), ha='center',fontsize=6)

    xlim = [lon_min, lon_max] #axis.get_xlim()
    ylim = [lat_min, lat_max]#axis.get_ylim()

    axis.set_xlim(xlim)
    axis.set_ylim(ylim)

    plt.colorbar(contour)

    if title_string is not None:
        axis.set_title(f'{storm_tag} max elev difference (m): {title_string}')
    else: 
        axis.set_title(f'{storm_tag} max elev difference (m): model2 - model1')

    if save_file_name is not None:
        print(f'Save maxelev diif plot in {storm_tag}_{save_file_name}.png') 
        plt.savefig(f'{storm_tag}_{save_file_name}.png', dpi=DPI)

###############################
def main(args):

    schism_dir1 = Path(args.schismdir1)
    schism_dir2 = Path(args.schismdir2)
    storm_name = args.name
    storm_year = args.year

    storm_tag = f"{storm_name.upper()}_{storm_year}"
    grid_file = schism_dir1 / "hgrid.gr3"

    results_dir1 = schism_dir1 / "outputs"
    if not results_dir1.exists():
        raise ValueError("Simulation results directory for model1 not found!")

    felev1 = results_dir1 / 'maxelev.gr3'
    if not felev1.exists():
        raise FileNotFoundError('Maximum elevation file for model1 is not found!')


    results_dir2 = schism_dir2 / "outputs"
    if not results_dir2.exists():
        raise ValueError("Simulation results directory for model2 not found!")

    felev2 = results_dir2 / 'maxelev.gr3'
    if not felev2.exists():
        raise FileNotFoundError('Maximum elevation file for model2 is not found!')


    if not grid_file.exists():
        raise FileNotFoundError('Grid file not found!')

    bbox_str = args.bbox_str

    # read grid info
    if grid_file is not None:
        hgrid = Hgrid.open(grid_file, crs=4326)

    # read model 1 maxelev
    if felev1 is not None:
        schism_welev = Hgrid.open(felev1, crs=4326)

    # read model 2 maxelev
    if felev2 is not None:
       schism_welev2 = Hgrid.open(felev2, crs=4326)

    # create mask
    idry = _create_idry(hgrid, schism_welev, schism_welev2, bbox_str) 

    #create plot
    _plot_maxelev_diff(storm_tag, hgrid, schism_welev, schism_welev2, 
        idry, bbox_str, save_file_name="test")

def entry():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name", help="name of the storm", type=str)

    parser.add_argument(
        "--year", help="year of the storm", type=int)

    parser.add_argument(
        "--schismdir1", type=str)

    parser.add_argument(
        "--schismdir2", type=str)

    parser.add_argument(
            '--bbox-str',
            default='-80.5,32.0,-75.0,36.5',
            help='format: lon_min,lat_min,lon_max,lat_max')

    main(parser.parse_args())

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    entry()

