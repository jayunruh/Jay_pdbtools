# Jay_pdbtools

This is a collection of python tools to read protein data bank (pdb) files into a pandas dataframe and perform manipulations on them.  General dependencies include pandas, numpy, and scipy.  Here is a description of the modules included:

*jpdbtools2*: codes to read and write pdb files and perform measurements including alignment, surface, and surface distance measurements.

*jpdbtools*: mirrors the functions in the above file but adds plotly and py3D mol methods for 3D visualization.

*mmcif_reader*: implements Bio.PDB (biopython) functions to read mmcif and pdb files into pandas dataframes.

*calc_mlp_jru*: implementation of the ChimeraX (https://www.cgl.ucsf.edu/chimerax/) lipophilcity calculation for measurements in 3D space.

*hbond_tools*: functions to predict hydrogen bonding from donor/acceptor distances and angles (requires hydrogens be added prior to running).
