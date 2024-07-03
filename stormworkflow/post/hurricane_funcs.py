from __future__ import division, print_function

# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Functions for handling nhc data


"""

__author__ = 'Saeed Moghimi'
__copyright__ = 'Copyright 2020, UCAR/NOAA'
__license__ = 'GPL'
__version__ = '1.0'
__email__ = 'moghimis@gmail.com'

import pandas as pd
import geopandas as gpd
import numpy as np
import sys
from glob import glob
import requests
from bs4 import BeautifulSoup

try:
    from urllib.request import urlopen, urlretrieve
except:
    from urllib import urlopen, urlretrieve
import lxml.html

import wget

# from highwatermarks import HighWaterMarks
# from collections import OrderedDict
# import json
import os


##################
def url_lister(url):
    urls = []
    connection = urlopen(url)
    dom = lxml.html.fromstring(connection.read())
    for link in dom.xpath('//a/@href'):
        urls.append(link)
    return urls


#################
def download(url, path, fname):
    sys.stdout.write(fname + '\n')
    if not os.path.isfile(path):
        urlretrieve(url, filename=path, reporthook=progress_hook(sys.stdout))
        sys.stdout.write('\n')
        sys.stdout.flush()


#################
def progress_hook(out):
    """
    Return a progress hook function, suitable for passing to
    urllib.retrieve, that writes to the file object *out*.
    """

    def it(n, bs, ts):
        got = n * bs
        if ts < 0:
            outof = ''
        else:
            # On the last block n*bs can exceed ts, so we clamp it
            # to avoid awkward questions.
            got = min(got, ts)
            outof = '/%d [%d%%]' % (ts, 100 * got // ts)
        out.write('\r  %d%s' % (got, outof))
        out.flush()

    return it


#################
def get_nhc_storm_info(year, name):
    """
    
    """

    print('Read list of hurricanes from NHC based on year')

    if int(year) < 2008:
        print('  ERROR:   GIS Data is not available for storms before 2008 ')
        sys.exit('Exiting .....')

    url = 'http://www.nhc.noaa.gov/gis/archive_wsurge.php?year=' + year

    # r = requests.get(url,headers=headers,verify=False)
    r = requests.get(url, verify=False)

    soup = BeautifulSoup(r.content, 'lxml')

    table = soup.find('table')
    # table = [row.get_text().strip().split(maxsplit=1) for row in table.find_all('tr')]

    tab = []
    for row in table.find_all('tr'):
        tmp = row.get_text().strip().split()
        tab.append([tmp[0], tmp[-1]])

    print(tab)

    df = pd.DataFrame(data=tab[:], columns=['identifier', 'name'], ).set_index('name')

    ###############################

    print('  > based on specific storm go fetch gis files')
    hid = df.to_dict()['identifier'][name.upper()]
    al_code = ('{}' + year).format(hid)
    hurricane_gis_files = '{}_5day'.format(al_code)

    return al_code, hurricane_gis_files


#################
# @retry(stop_max_attempt_number=5, wait_fixed=3000)
def download_nhc_gis_files(hurricane_gis_files, rundir):
    """
    """

    base = os.path.abspath(os.path.join(rundir, 'nhcdata', hurricane_gis_files))

    if len(glob(base + '/*')) < 1:
        nhc = 'http://www.nhc.noaa.gov/gis/forecast/archive/'

        # We don't need the latest file b/c that is redundant to the latest number.
        fnames = [
            fname
            for fname in url_lister(nhc)
            if fname.startswith(hurricane_gis_files) and 'latest' not in fname
        ]

        if not os.path.exists(base):
            os.makedirs(base)

        for fname in fnames:
            path1 = os.path.join(base, fname)
            if not os.path.exists(path1):
                url = '{}/{}'.format(nhc, fname)
                download(url, path1, fname)

    return base
    #################################


# Only needed to run on binder!
# See https://gitter.im/binder-project/binder?at=59bc2498c101bc4e3acfc9f1
os.environ['CPL_ZIP_ENCODING'] = 'UTF-8'


def read_advisory_cones_info(hurricane_gis_files, base, year, code):
    print('  >  Read cones shape file ...')

    cones, points = [], []
    for fname in sorted(glob(os.path.join(base, '{}_*.zip'.format(hurricane_gis_files)))):
        number = os.path.splitext(os.path.split(fname)[-1])[0].split('_')[-1]

        # read cone shapefiles

        if int(year) < 2014:
            # al092008.001_5day_pgn.shp
            divd = '.'
        else:
            divd = '-'

        pgn = gpd.read_file(
            ('/{}' + divd + '{}_5day_pgn.shp').format(code, number),
            vfs='zip://{}'.format(fname),
        )
        cones.append(pgn)

        # read points shapefiles
        pts = gpd.read_file(
            ('/{}' + divd + '{}_5day_pts.shp').format(code, number),
            vfs='zip://{}'.format(fname),
        )
        # Only the first "obsevartion."
        points.append(pts.iloc[0])

    return cones, points, pts


#################
def download_nhc_best_track(year, code):
    """
    
    """

    url = 'http://ftp.nhc.noaa.gov/atcf/archive/{}/'.format(year)
    fname = 'b{}.dat.gz'.format(code)
    base = os.path.abspath(os.path.join(os.path.curdir, 'data', code + '_best_track'))

    if not os.path.exists(base):
        os.makedirs(base)

    path1 = os.path.join(base, fname)
    # download(url, path,fname)
    if not os.path.exists(url + fname):
        wget.download(url + fname, out=base)

    return base


#################
def download_nhc_gis_best_track(year, code):
    """
    
    """

    url = 'http://www.nhc.noaa.gov/gis/best_track/'
    fname = '{}_best_track.zip'.format(code)
    base = os.path.abspath(os.path.join(os.path.curdir, 'data', code + '_best_track'))

    if not os.path.exists(base):
        os.makedirs(base)

    path = os.path.join(base, fname)
    # download(url, path,fname)
    if not os.path.exists(url + fname):
        wget.download(url + fname, out=base)
    return base


#################
def read_gis_best_track(base, code):
    """
    
    """
    print('  >  Read GIS Best_track file ...')

    fname = base + '/{}_best_track.zip'.format(code)

    points = gpd.read_file(('/{}_pts.shp').format(code), vfs='zip://{}'.format(fname))

    radii = gpd.read_file(('/{}_radii.shp').format(code), vfs='zip://{}'.format(fname))

    line = gpd.read_file(('/{}_lin.shp').format(code), vfs='zip://{}'.format(fname))

    return line, points, radii


def get_coordinates(bbox):
    """
    Create bounding box coordinates for the map.  It takes flat or
    nested list/numpy.array and returns 5 points that closes square
    around the borders.

    Examples
    --------
    >>> bbox = [-87.40, 24.25, -74.70, 36.70]
    >>> len(get_coordinates(bbox))
    5

    """
    bbox = np.asanyarray(bbox).ravel()
    if bbox.size == 4:
        bbox = bbox.reshape(2, 2)
        coordinates = []
        coordinates.append([bbox[0][1], bbox[0][0]])
        coordinates.append([bbox[0][1], bbox[1][0]])
        coordinates.append([bbox[1][1], bbox[1][0]])
        coordinates.append([bbox[1][1], bbox[0][0]])
        coordinates.append([bbox[0][1], bbox[0][0]])
    else:
        raise ValueError('Wrong number corners.' '  Expected 4 got {}'.format(bbox.size))
    return coordinates
