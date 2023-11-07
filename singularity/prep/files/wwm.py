from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import f90nml
import numpy as np
from pyschism.mesh.base import Elements
from pyschism.mesh.base import Gr3
from pyschism.mesh.gridgr3 import Gr3Field
from pyschism.param.param import Param


REFS = Path('/refs')

def setup_wwm(mesh_file: Path, setup_dir: Path, ensemble: bool):
    '''Output is
        - hgrid_WWM.gr3
        - param.nml
        - wwmbnd.gr3
        - wwminput.nml
    '''

    
    runs_dir = [setup_dir]
    if ensemble:
        spinup_dir = setup_dir/'spinup'
        runs_dir = setup_dir.glob('runs/*')

    schism_grid = Gr3.open(mesh_file, crs=4326)
    wwm_grid = break_quads(schism_grid)
    wwm_bdry = Gr3Field.constant(wwm_grid, 0.0)

    # TODO: Update spinup
    # NOTE: Requires setup of WWM hotfile

    # Update runs
    for run in runs_dir:
        wwm_grid.write(run / 'hgrid_WWM.gr3', format='gr3')
        wwm_bdry.write(run / 'wwmbnd.gr3', format='gr3')

        schism_nml = update_schism_params(run / 'param.nml')
        schism_nml.write(run / 'param.nml', force=True)

        wwm_nml = get_wwm_params(run_name=run.name, schism_nml=schism_nml)
        wwm_nml.write(run / 'wwminput.nml')
        


def break_quads(pyschism_mesh: Gr3) -> Gr3 | Gr3Field:
    # Create new Elements and set it for the Gr3.elements
    quads = pyschism_mesh.quads
    if len(quads) == 0:
        new_mesh = deepcopy(pyschism_mesh)

    else:
        tmp = quads[:,2:]
        tmp = np.insert(tmp, 0, quads[:, 0], axis=1)
        broken = np.vstack((quads[:, :3], tmp))
        trias = pyschism_mesh.triangles
        final_trias = np.vstack((trias, broken))
        # NOTE: Node IDs and indexs are the same as before
        elements = {
            idx+1: list(map(pyschism_mesh.nodes.get_id_by_index, tri))
            for idx, tri in enumerate(final_trias)
        }

        new_mesh = deepcopy(pyschism_mesh)
        new_mesh.elements = Elements(pyschism_mesh.nodes, elements)


    return new_mesh



def get_wwm_params(run_name, schism_nml) -> f90nml.Namelist:
    
    # Get relevant values from SCHISM setup
    begin_time = datetime(
        year=schism_nml['opt']['start_year'],
        month=schism_nml['opt']['start_month'],
        day=schism_nml['opt']['start_day'],
        # TODO: Handle decimal hour
        hour=int(schism_nml['opt']['start_hour']),
    )
    end_time = begin_time + timedelta(days=schism_nml['core']['rnday'])
    delta_t = schism_nml['core']['dt']
    mdc = schism_nml['core']['mdc2']
    msc = schism_nml['core']['msc2']
    nstep_wwm = schism_nml['opt']['nstep_wwm']

    time_fmt = '%Y%m%d.%H%M%S'
    wwm_delta_t = nstep_wwm * delta_t

    # For now just read the example file update relevant names and write
    wwm_params = f90nml.read(REFS/'wwminput.nml')
    wwm_params.uppercase = True

    proc_nml = wwm_params['PROC']
    proc_nml['PROCNAME'] = run_name
    # Time for start the simulation, ex:yyyymmdd. hhmmss
    proc_nml['BEGTC'] = begin_time.strftime(time_fmt)
    # Time step (MUST match dt*nstep_wwm in SCHISM!)
    proc_nml['DELTC'] = wwm_delta_t
    # Unity of time step
    proc_nml['UNITC'] = 'SEC'
    # Time for stop the simulation, ex:yyyymmdd. hhmmss
    proc_nml['ENDTC'] = end_time.strftime(time_fmt)
    # Minimum water depth. THis must be same as h0 in selfe
    proc_nml['DMIN'] = 0.01

    grid_nml = wwm_params['GRID']
    # Number of directional bins
    grid_nml['MDC'] = mdc
    # Number of frequency bins
    grid_nml['MSC'] = msc
    # Name of the grid file. hgrid.gr3 if IGRIDTYPE = 3 (SCHISM)
    grid_nml['FILEGRID'] = 'hgrid_WWM.gr3'
    # Gridtype used.
    grid_nml['IGRIDTYPE'] = 3

    bouc_nml = wwm_params['BOUC']
    # Begin time of the wave boundary file (FILEWAVE) 
    bouc_nml['BEGTC'] = begin_time.strftime(time_fmt)
    # Time step in FILEWAVE 
    bouc_nml['DELTC'] = 1
    # Unit can be HR, MIN, SEC 
    bouc_nml['UNITC'] = 'HR'
    # End time
    bouc_nml['ENDTC'] = end_time.strftime(time_fmt)
    # Boundary file defining boundary conditions and Neumann nodes.
    bouc_nml['FILEBOUND'] = 'wwmbnd.gr3'
    bouc_nml['BEGTC_OUT'] = 20030908.000000 
    bouc_nml['DELTC_OUT'] = 600.000000000000
    bouc_nml['UNITC_OUT'] = 'SEC'
    bouc_nml['ENDTC_OUT'] = 20031008.000000 
    
    hist_nml = wwm_params['HISTORY']
    # Start output time, yyyymmdd. hhmmss;
    # must fit the simulation time otherwise no output.
    # Default is same as PROC%BEGTC
    hist_nml['BEGTC'] = begin_time.strftime(time_fmt)
    # Time step for output; if smaller than simulation time step, the latter is used (output every step for better 1D 2D spectra analysis)
    hist_nml['DELTC'] = 1
    # Unit
    hist_nml['UNITC'] = 'SEC'
    # Stop time output, yyyymmdd. hhmmss
    # Default is same as PROC%ENDC
    hist_nml['ENDTC'] = end_time.strftime(time_fmt)
    # Time scoop (sec) for history files
    hist_nml['DEFINETC'] = 86400
    hist_nml['FILEOUT'] = 'wwm_hist.dat'

    sta_nml = wwm_params['STATION']
    # Start simulation time, yyyymmdd. hhmmss; must fit the simulation time otherwise no output
    # Default is same as PROC%BEGTC
    sta_nml['BEGTC'] = begin_time.strftime(time_fmt)
    # Time step for output; if smaller than simulation time step, the latter is used (output every step for better 1D 2D spectra analysis)
    sta_nml['DELTC'] = wwm_delta_t
    # Unit
    sta_nml['UNITC'] = 'SEC'
    # Stop time simulation, yyyymmdd. hhmmss
    # Default is same as PROC%ENDC
    sta_nml['ENDTC'] = end_time.strftime(time_fmt)
    # Time for definition of station files
    sta_nml['DEFINETC'] = 86400

    # TODO: Add hot file?
    hot_nml = wwm_params['HOTFILE']
    # Write hotfile
    hot_nml['LHOTF'] = False
    #'.nc' suffix will be added 
#    hot_nml['FILEHOT_OUT'] = 'wwm_hot_out'
#    #Starting time of hotfile writing. With ihot!=0 in SCHISM,
#    # this will be whatever the new hotstarted time is (even with ihot=2)
#    hot_nml['BEGTC'] = '20030908.000000'
#    # time between hotfile writes
#    hot_nml['DELTC'] = 86400.
#    # unit used above
#    hot_nml['UNITC'] = 'SEC'
#    # Ending time of hotfile writing (adjust with BEGTC)
#    hot_nml['ENDTC'] = '20031008.000000'
#    # Applies only to netcdf
#    # If T then hotfile contains 2 last records.
#    # If F then hotfile contains N record if N outputs
#    # have been done.
#    # For binary only one record.
#    hot_nml['LCYCLEHOT'] = True
#    # 1: binary hotfile of data as output
#    # 2: netcdf hotfile of data as output (default)
#    hot_nml['HOTSTYLE_OUT'] = 2
#    # 0: hotfile in a single file (binary or netcdf)
#    # MPI_REDUCE is then used and thus youd avoid too freq. output 
#    # 1: hotfiles in separate files, each associated
#    # with one process
#    hot_nml['MULTIPLEOUT'] = 0
#    # (Full) hot file name for input
#    hot_nml['FILEHOT_IN'] = 'wwm_hot_in.nc'
#    # 1: binary hotfile of data as input
#    # 2: netcdf hotfile of data as input (default)
#    hot_nml['HOTSTYLE_IN'] = 2
#    # Position in hotfile (only for netcdf)
#    # for reading
#    hot_nml['IHOTPOS_IN'] = 1
#    # 0: read hotfile from one single file
#    # 1: read hotfile from multiple files (must use same # of CPU?)
#    hot_nml['MULTIPLEIN'] = 0

    return wwm_params


def update_schism_params(path: Path) -> f90nml.Namelist:

    schism_nml = f90nml.read(path)

    core_nml = schism_nml['core']
    core_nml['msc2'] = 24
    core_nml['mdc2'] = 30

    opt_nml = schism_nml['opt']
    opt_nml['icou_elfe_wwm'] = 1
    opt_nml['nstep_wwm'] = 4
    opt_nml['iwbl'] = 0
    opt_nml['hmin_radstress'] = 1.
    # TODO: Revisit for spinup support
    # NOTE: Issue 7#issuecomment-1482848205 oceanmodeling fork
#    opt_nml['nrampwafo'] = 0
    opt_nml['drampwafo'] = 0.
    opt_nml['turbinj'] = 0.15
    opt_nml['turbinjds'] = 1.0
    opt_nml['alphaw'] = 0.5


    # NOTE: Python index is different from the NML index
    schout_nml = schism_nml['schout']

    schout_nml['iof_hydro'] = [1]
    schout_nml['iof_wwm'] = [0 for i in range(17)]

    schout_nml.start_index.update(iof_hydro=[14], iof_wwm=[1])

    #sig. height (m) {sigWaveHeight}   2D
    schout_nml['iof_wwm'][0] = 1
    #Mean average period (sec) - TM01 {meanWavePeriod}  2D
    schout_nml['iof_wwm'][1] = 0
    #Zero down crossing period for comparison with buoy (s) - TM02 {zeroDowncrossPeriod}  2D
    schout_nml['iof_wwm'][2] = 0
    #Average period of wave runup/overtopping - TM10 {TM10}  2D
    schout_nml['iof_wwm'][3] = 0
    #Mean wave number (1/m) {meanWaveNumber}  2D
    schout_nml['iof_wwm'][4] = 0
    #Mean wave length (m) {meanWaveLength}  2D
    schout_nml['iof_wwm'][5] = 0
    #Mean average energy transport direction (degr) - MWD in NDBC? {meanWaveDirection}  2D
    schout_nml['iof_wwm'][6] = 0
    #Mean directional spreading (degr) {meanDirSpreading}  2D
    schout_nml['iof_wwm'][7] = 0
    #Discrete peak period (sec) - Tp {peakPeriod}  2D
    schout_nml['iof_wwm'][8] = 1
    #Continuous peak period based on higher order moments (sec) {continuousPeakPeriod}  2D
    schout_nml['iof_wwm'][9] = 0
    #Peak phase vel. (m/s) {peakPhaseVel}  2D
    schout_nml['iof_wwm'][10] = 0
    #Peak n-factor {peakNFactor}   2D
    schout_nml['iof_wwm'][11] = 0
    #Peak group vel. (m/s) {peakGroupVel}   2D
    schout_nml['iof_wwm'][12] = 0
    #Peak wave number {peakWaveNumber}  2D
    schout_nml['iof_wwm'][13] = 0
    #Peak wave length {peakWaveLength}  2D
    schout_nml['iof_wwm'][14] = 0
    #Peak (dominant) direction (degr) {dominantDirection}  2D
    schout_nml['iof_wwm'][15] = 1
    #Peak directional spreading {peakSpreading}  2D
    schout_nml['iof_wwm'][16] = 0

    return schism_nml
