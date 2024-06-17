#!/usr/bin/env python

# Import modules
import logging
import os
import pathlib
import argparse
import sys
import warnings

import numpy as np

from fiona.drvsupport import supported_drivers
from shapely.geometry import box, MultiLineString
from shapely.ops import polygonize, unary_union, linemerge
from pyproj import CRS, Transformer
import geopandas as gpd

from ocsmesh import Raster, Geom, Hfun, JigsawDriver, Mesh, utils
from ocsmesh.cli.subset_n_combine import SubsetAndCombine


# Setup modules
# Enable KML driver
#from https://stackoverflow.com/questions/72960340/attributeerror-nonetype-object-has-no-attribute-drvsupport-when-using-fiona
supported_drivers['KML'] = 'rw'
supported_drivers['LIBKML'] = 'rw'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S')


# Helper functions
def get_raster(path, crs=None):
    rast = Raster(path)
    if crs and rast.crs != crs:
        rast.warp(crs)
    return rast


def get_rasters(paths, crs=None):
    rast_list = list()
    for p in paths:
        rast_list.append(get_raster(p, crs))
    return rast_list


def _generate_mesh_boundary_and_write(
        out_dir, mesh_path, mesh_crs='EPSG:4326', threshold=-1000
    ):

    mesh = Mesh.open(str(mesh_path), crs=mesh_crs)

    logger.info('Calculating boundary types...')
    mesh.boundaries.auto_generate(threshold=threshold)

    logger.info('Write interpolated mesh to disk...')
    mesh.write(
        str(out_dir/f'mesh_w_bdry.grd'), format='grd', overwrite=True
    )


def _write_mesh_box(out_dir, mesh_path, mesh_crs='EPSG:4326'):
    mesh = Mesh.open(str(mesh_path), crs=mesh_crs)
    domain_box = box(*mesh.get_multipolygon().bounds)
    gdf_domain_box = gpd.GeoDataFrame(
            geometry=[domain_box], crs=mesh.crs)
    gdf_domain_box.to_file(out_dir/'domain_box')


# Main script
def main(args, clients):

    cmd = args.cmd
    logger.info(f"The mesh command is {cmd}.")

    clients_dict = {c.script_name: c for c in clients}

    storm_name = str(args.name).lower()
    storm_year = str(args.year).lower()

    final_mesh_name = 'hgrid.gr3'
    write_mesh_box = False

    if cmd == 'subset_n_combine':
        final_mesh_name = 'final_mesh.2dm'
        write_mesh_box = True

    elif cmd == 'hurricane_mesh':
        final_mesh_name = 'mesh_no_bdry.2dm'

    if cmd in clients_dict:
        clients_dict[cmd].run(args)
    else:
        raise ValueError(f'Invalid meshing command specified: <{cmd}>')

    #TODO interpolate DEM?
    if write_mesh_box:
        _write_mesh_box(args.out, args.out / final_mesh_name)
    _generate_mesh_boundary_and_write(args.out, args.out / final_mesh_name)


class HurricaneMesher:

    @property
    def script_name(self):
        return 'hurricane_mesh'

    def __init__(self, sub_parser):

        this_parser = sub_parser.add_parser(self.script_name)

        this_parser.add_argument(
            "--lo-dem", nargs='+', help="Path to low resolution DEMS", type=pathlib.Path)
        this_parser.add_argument(
            "--hi-dem", nargs='+', help="Path to high resolution DEMS", type=pathlib.Path)

        this_parser.add_argument(
            "--nprocs", type=int, help="Number of parallel threads to use when "
            "computing geom and hfun.")

        this_parser.add_argument(
            "--geom-nprocs", type=int, help="Number of processors used when "
            "computing the geom, overrides --nprocs argument.")

        this_parser.add_argument(
            "--hfun-nprocs", type=int, help="Number of processors used when "
            "computing the hfun, overrides --nprocs argument.")

        this_parser.add_argument(
            "--hmax", type=float, help="Maximum mesh size.",
            default=20000)

        this_parser.add_argument(
            "--hmin-low", type=float, default=1500,
            help="Minimum mesh size for low resolution region.")
        
        this_parser.add_argument(
            "--rate-low", type=float, default=2e-3,
            help="Expansion rate for low resolution region.")

        this_parser.add_argument(
            "--contours", type=float, nargs=2,
            help="Contour specification applied to whole domain; "
            "contour mesh size needs to be greater that hmin-low",
            metavar="SPEC")

        this_parser.add_argument(
            "--transition-elev", "-e", type=float, default=-200,
            help="Cut off elev for high resolution region")

        this_parser.add_argument(
            "--hmin-high", type=float, default=300,
            help="Minimum mesh size for high resolution region.")

        this_parser.add_argument(
            "--rate-high", type=float, default=1e-3,
            help="Expansion rate for high resolution region")

        this_parser.add_argument(
            "--shapes-dir",
            help="top-level directory that contains shapefiles",
            type=pathlib.Path
        )

        this_parser.add_argument(
            "--windswath",
            help="path to NHC windswath shapefile",
            type=pathlib.Path
        )

        # Similar to the argument for SubsetAndCombine
        this_parser.add_argument(
            "--out",
            help="mesh operation output directory",
            type=pathlib.Path
        )

    def run(self, args):

        nprocs = args.nprocs

        geom_nprocs = nprocs
        if args.geom_nprocs:
            nprocs = args.geom_nprocs
        geom_nprocs = -1 if nprocs == None else nprocs

        hfun_nprocs = nprocs
        if args.hfun_nprocs:
            nprocs = args.hfun_nprocs
        hfun_nprocs = -1 if nprocs == None else nprocs
        
        storm_name = str(args.name).lower()
        storm_year = str(args.year).lower()

        lo_res_dems = args.lo_dem
        hi_res_dems = args.hi_dem
#        hi_tiles = args.hi_tiles
        shp_dir = pathlib.Path(args.shapes_dir)
        hurr_info = args.windswath
        out_dir = args.out

        coarse_geom = shp_dir / 'base_geom'
        fine_geom = shp_dir / 'high_geom'


        all_dem_paths = [*lo_res_dems, *hi_res_dems]
        lo_res_paths = lo_res_dems

        # Specs
        wind_kt = 34
        filter_factor = 3
        max_n_hires_dem = 150


        # Geom (hardcoded based on prepared hurricane meshing spec)
        z_max_lo = 0
        z_max_hi = 10
        z_max = max(z_max_lo, z_max_hi)

        # Hfun
        hmax = args.hmax

        hmin_lo = args.hmin_low
        rate_lo = args.rate_low

        contour_specs_lo = []
        if args.contours is not None:
            for c_elev, m_size in args.contours:
                if hmin_lo > m_size:
                    warnings.warn(
                        "Specified contour must have a mesh size"
                        f" larger than minimum low res size: {hmin_low}")
                contour_specs_lo.append((c_elev, rate_lo, m_size))

        else:
            contour_specs_lo = [
                (-4000, rate_lo, 10000),
                (-1000, rate_lo, 6000),
                (-10, rate_lo, hmin_lo)
            ]

        const_specs_lo = [
            (hmin_lo, 0, z_max)
        ]

        cutoff_hi = args.transition_elev
        hmin_hi = args.hmin_high
        rate_hi = args.rate_high

        contour_specs_hi = [
            (0, rate_hi, hmin_hi)
        ]
        const_specs_hi = [
            (hmin_hi, 0, z_max)
        ]


        # Read inputs
        logger.info("Reading input shapes...")
        gdf_fine = gpd.read_file(fine_geom)
        gdf_coarse = gpd.read_file(coarse_geom)

        logger.info("Reading hurricane info...")
        gdf = gpd.read_file(hurr_info)
        gdf_wind_kt = gdf[gdf.RADII.astype(int) == wind_kt]

        # Simplify high resolution geometry
        logger.info("Simplify high-resolution shape...")
        gdf_fine = gpd.GeoDataFrame(
            geometry=gdf_fine.to_crs("EPSG:3857").simplify(tolerance=hmin_hi / 2).buffer(0).to_crs(gdf_fine.crs),
            crs=gdf_fine.crs)


        # Calculate refinement region
        logger.info(f"Create polygon from {wind_kt}kt windswath polygon...")
        ext_poly = [i for i in polygonize([ext for ext in gdf_wind_kt.exterior])]
        gdf_refine_super_0 = gpd.GeoDataFrame(
            geometry=ext_poly, crs=gdf_wind_kt.crs)

        logger.info("Find upstream...")
        domain_extent = gdf_fine.to_crs(gdf_refine_super_0.crs).total_bounds
        domain_box = box(*domain_extent)
        box_tol = 1/1000 * max(domain_extent[2]- domain_extent[0], domain_extent[3] - domain_extent[1])
        gdf_refine_super_0 = gdf_refine_super_0.intersection(domain_box.buffer(-box_tol))
        gdf_refine_super_0.plot()
        ext_poly = [i for i in gdf_refine_super_0.explode().geometry]

        dmn_ext = [pl.exterior for mp in gdf_fine.geometry for pl in mp]
        wnd_ext = [pl.exterior for pl in ext_poly]

        gdf_dmn_ext = gpd.GeoDataFrame(geometry=dmn_ext, crs=gdf_fine.crs)
        gdf_wnd_ext = gpd.GeoDataFrame(geometry=wnd_ext, crs=gdf_wind_kt.crs)

        gdf_ext_over = gpd.overlay(gdf_dmn_ext, gdf_wnd_ext.to_crs(gdf_dmn_ext.crs), how="union")

        gdf_ext_x = gdf_ext_over[gdf_ext_over.intersects(gdf_wnd_ext.to_crs(gdf_ext_over.crs).unary_union)]

        filter_lines_threshold = np.max(gdf_dmn_ext.length) / filter_factor
        lnstrs = linemerge([lnstr for lnstr in gdf_ext_x.explode().geometry])
        if not isinstance(lnstrs, MultiLineString):
            lnstrs = [lnstrs]
        lnstrs = [lnstr for lnstr in lnstrs if lnstr.length < filter_lines_threshold]
        gdf_hurr_w_upstream = gdf_wnd_ext.to_crs(gdf_ext_x.crs)
        gdf_hurr_w_upstream = gdf_hurr_w_upstream.append(
            gpd.GeoDataFrame(
                geometry=gpd.GeoSeries(lnstrs),
                crs=gdf_ext_x.crs
            ))


        gdf_hurr_w_upstream_poly = gpd.GeoDataFrame(
            geometry=gpd.GeoSeries(polygonize(gdf_hurr_w_upstream.unary_union)),
            crs=gdf_hurr_w_upstream.crs)

        logger.info("Find intersection of domain polygon with impacted area upstream...")
        gdf_refine_super_2 = gpd.overlay(
            gdf_fine, gdf_hurr_w_upstream_poly.to_crs(gdf_fine.crs),
            how='intersection'
        )

        gdf_refine_super_2.to_file(out_dir / 'dmn_hurr_upstream')

        logger.info("Selecting high resolution DEMs...")
        gdf_dem_box = gpd.GeoDataFrame(
            columns=['geometry', 'path'],
            crs=gdf_refine_super_2.crs)
        for path in all_dem_paths:
            bbox = Raster(path).get_bbox(crs=gdf_dem_box.crs)
            gdf_dem_box = gdf_dem_box.append(
                gpd.GeoDataFrame(
                    {'geometry': [bbox],
                     'path': str(path)},
                    crs=gdf_dem_box.crs)
            )
        gdf_dem_box = gdf_dem_box.reset_index()


        # TODO: use sjoin instead?!
        gdf_hi_res_box = gdf_dem_box[gdf_dem_box.geometry.intersects(gdf_refine_super_2.unary_union)].reset_index()
        hi_res_paths = gdf_hi_res_box.path.values.tolist()


        # For refine cut off either use static geom at e.g. 200m depth or instead just use low-res for cut off polygon


        # Or intersect with full geom? (timewise an issue for hfun creation)
        logger.info("Calculate refinement area cutoff...")
        cutoff_dem_paths = [i for i in gdf_hi_res_box.path.values.tolist() if pathlib.Path(i) in lo_res_paths]
        cutoff_geom = Geom(
            get_rasters(cutoff_dem_paths),
            base_shape=gdf_coarse.unary_union,
            base_shape_crs=gdf_coarse.crs,
            zmax=cutoff_hi,
            nprocs=geom_nprocs)
        cutoff_poly = cutoff_geom.get_multipolygon()

        gdf_cutoff = gpd.GeoDataFrame(
            geometry=gpd.GeoSeries(cutoff_poly),
            crs=cutoff_geom.crs)

        gdf_draft_refine = gpd.overlay(gdf_refine_super_2, gdf_cutoff.to_crs(gdf_refine_super_2.crs), how='difference')

        refine_polys = [pl for pl in gdf_draft_refine.unary_union]

        gdf_final_refine = gpd.GeoDataFrame(
            geometry=refine_polys,
            crs=gdf_draft_refine.crs)


        logger.info("Write landfall area to disk...")
        gdf_final_refine.to_file(out_dir/'landfall_refine_area')

        gdf_geom = gpd.overlay(
            gdf_coarse,
            gdf_final_refine.to_crs(gdf_coarse.crs),
            how='union')

        domain_box = box(*gdf_fine.total_bounds)
        gdf_domain_box = gpd.GeoDataFrame(
                geometry=[domain_box], crs=gdf_fine.crs)
        gdf_domain_box.to_file(out_dir/'domain_box')

        geom = Geom(gdf_geom.unary_union, crs=gdf_geom.crs)


        logger.info("Create low-res size function...")
        hfun_lo = Hfun(
            get_rasters(lo_res_paths),
            base_shape=gdf_coarse.unary_union,
            base_shape_crs=gdf_coarse.crs,
            hmin=hmin_lo,
            hmax=hmax,
            nprocs=hfun_nprocs,
            method='fast')

        logger.info("Add refinement spec to low-res size function...")
        for ctr in contour_specs_lo:
            hfun_lo.add_contour(*ctr)
            hfun_lo.add_constant_value(value=ctr[2], lower_bound=ctr[0])

        for const in const_specs_lo:
            hfun_lo.add_constant_value(*const)

        # hfun_lo.add_subtidal_flow_limiter(upper_bound=z_max)
        # hfun_lo.add_subtidal_flow_limiter(hmin=hmin_lo, upper_bound=z_max)


        logger.info("Compute low-res size function...")
        jig_hfun_lo = hfun_lo.msh_t()


        logger.info("Write low-res size function to disk...")
        Mesh(jig_hfun_lo).write(
                str(out_dir/f'hfun_lo_{hmin_hi}.2dm'),
                format='2dm',
                overwrite=True)


        # For interpolation after meshing and use GEBCO for mesh size calculation in refinement area.
        hfun_hi_rast_paths = hi_res_paths
        if len(hi_res_paths) > max_n_hires_dem:
            hfun_hi_rast_paths = lo_res_paths

        logger.info("Create high-res size function...")
        hfun_hi = Hfun(
            get_rasters(hfun_hi_rast_paths),
            base_shape=gdf_final_refine.unary_union,
            base_shape_crs=gdf_final_refine.crs,
            hmin=hmin_hi,
            hmax=hmax,
            nprocs=hfun_nprocs,
            method='fast')

        # Apply low resolution criteria on hires as ewll
        logger.info("Add refinement spec to high-res size function...")
        for ctr in contour_specs_lo:
            hfun_hi.add_contour(*ctr)
            hfun_hi.add_constant_value(value=ctr[2], lower_bound=ctr[0])

        for ctr in contour_specs_hi:
            hfun_hi.add_contour(*ctr)
            hfun_hi.add_constant_value(value=ctr[2], lower_bound=ctr[0])

        for const in const_specs_hi:
            hfun_hi.add_constant_value(*const)

        # hfun_hi.add_subtidal_flow_limiter(upper_bound=z_max)

        logger.info("Compute high-res size function...")
        jig_hfun_hi = hfun_hi.msh_t()

        logger.info("Write high-res size function to disk...")
        Mesh(jig_hfun_hi).write(
            str(out_dir/f'hfun_hi_{hmin_hi}.2dm'),
            format='2dm',
            overwrite=True)


        jig_hfun_lo = Mesh.open(str(out_dir/f'hfun_lo_{hmin_hi}.2dm'), crs="EPSG:4326").msh_t
        jig_hfun_hi = Mesh.open(str(out_dir/f'hfun_hi_{hmin_hi}.2dm'), crs="EPSG:4326").msh_t


        logger.info("Combine size functions...")
        gdf_final_refine = gpd.read_file(out_dir/'landfall_refine_area')

        utils.clip_mesh_by_shape(
            jig_hfun_hi,
            shape=gdf_final_refine.to_crs(jig_hfun_hi.crs).unary_union,
            fit_inside=True,
            in_place=True)

        jig_hfun_final = utils.merge_msh_t(
            jig_hfun_lo, jig_hfun_hi,
            drop_by_bbox=False,
            can_overlap=False,
            check_cross_edges=True)


        logger.info("Write final size function to disk...")
        hfun_mesh = Mesh(jig_hfun_final)
        hfun_mesh.write(
            str(out_dir/f'hfun_comp_{hmin_hi}.2dm'),
            format='2dm',
            overwrite=True)


        hfun = Hfun(hfun_mesh)

        logger.info("Generate mesh...")
        driver = JigsawDriver(geom=geom, hfun=hfun, initial_mesh=True)
        mesh = driver.run()


        utils.reproject(mesh.msh_t, "EPSG:4326")
        mesh.write(
            str(out_dir/f'mesh_raw_{hmin_hi}.2dm'),
            format='2dm',
            overwrite=True)

        mesh = Mesh.open(str(out_dir/f'mesh_raw_{hmin_hi}.2dm'), crs="EPSG:4326")

        dst_crs = "EPSG:4326"
        interp_rast_list = [
            *get_rasters(lo_res_paths, dst_crs),
            *get_rasters(gdf_hi_res_box.path.values, dst_crs)]

        # TODO: Fix the deadlock issue with multiple cores when interpolating
        logger.info("Interpolate DEMs on the generated mesh...")
        mesh.interpolate(interp_rast_list, nprocs=1, method='nearest')

        logger.info("Write raw mesh to disk...")
        mesh.write(
            str(out_dir/f'mesh_{hmin_hi}.2dm'),
            format='2dm',
            overwrite=True)

        # Write the same mesh with a generic name
        mesh.write(
            str(out_dir/f'mesh_no_bdry.2dm'),
            format='2dm',
            overwrite=True)


def cli():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name", help="name of the storm", type=str)
    parser.add_argument(
        "year", help="year of the storm", type=int)

    subparsers = parser.add_subparsers(dest='cmd')
    subset_client = SubsetAndCombine(subparsers)
    hurrmesh_client = HurricaneMesher(subparsers)

    args = parser.parse_args()
    if 'outall' in args:
        args.outall = True

    logger.info(f"Mesh arguments are {args}.")

    main(args, [hurrmesh_client, subset_client])


if __name__ == '__main__':
    cli()

