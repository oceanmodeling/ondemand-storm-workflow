from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

cdict = {
    'red': (
        (0.0, 1, 1),
        (0.05, 1, 1),
        (0.11, 0, 0),
        (0.66, 1, 1),
        (0.89, 1, 1),
        (1, 0.5, 0.5),
    ),
    'green': (
        (0.0, 1, 1),
        (0.05, 1, 1),
        (0.11, 0, 0),
        (0.375, 1, 1),
        (0.64, 1, 1),
        (0.91, 0, 0),
        (1, 0, 0),
    ),
    'blue': ((0.0, 1, 1), (0.05, 1, 1), (0.11, 1, 1), (0.34, 1, 1), (0.65, 0, 0), (1, 0, 0)),
}

jetMinWi = LinearSegmentedColormap('my_colormap', cdict, 256)
my_cmap = plt.cm.jet

# Color code for the point track.
colors_hurricane_condition = {
    'subtropical depression': '#ffff99',
    'tropical depression': '#ffff66',
    'tropical storm': '#ffcc99',
    'subtropical storm': '#ffcc66',
    'hurricane': 'red',
    'major hurricane': 'crimson',
}

width = 750
height = 250

# Constants
noaa_logo = 'https://www.nauticalcharts.noaa.gov/images/noaa-logo-no-ring-70.png'

template_track_popup = """
    <div style="width: 200px; height: 90px" </div>
    <h5> {} condition</h5> <br>
    'Date:      {} <br> 
     Condition: {} <br>
    """

template_storm_info = """
            <div style="position: fixed; 
                        bottom: 50px; left: 5px; width: 140px; height: 45px; 
                        border:2px solid grey; z-index:9999; font-size:14px;background-color: lightgray;opacity: 0.9;
                        ">&nbsp; Storm: {} <br>
                          &nbsp; Year:  {}  &nbsp; <br>
            </div>
            """

template_fct_info = """
            <div style="position: fixed; 
                        bottom: 100px; left: 5px; width: 170px; height: 45px; 
                        border:2px solid grey; z-index:9999; font-size:14px;background-color: lightgray;opacity: 0.9;
                        ">&nbsp; Date:  {}UTC <br>
                          &nbsp; FCT : t{}z  &nbsp; <br>
            </div>
            """

disclaimer = """
                <div style="position: fixed; 
                            bottom: 5px; left: 250px; width: 520px; height: px; 
                            border:2px solid grey; z-index:9999; font-size:12px; background-color: lightblue;opacity: 0.6;
                            ">&nbsp; Hurricane Explorer;  
                            <a href="https://nauticalcharts.noaa.gov/" target="_blank" >         NOAA/NOS/OCS</a> <br>
                              &nbsp; Contact: Saeed.Moghimi@noaa.gov &nbsp; <br>
                              &nbsp; Disclaimer: Experimental product. All configurations and results are pre-decisional.<br>
                </div>
  
                """
