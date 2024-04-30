"""
Run TC downscaling model

Author: Jonathan Lin

Modified: Craig Arthur
 - enable multiple iterations and run plotting routines on completion
"""

import os
import shutil
import namelist
import sys
from scripts import generate_land_masks #, plotLMI, plotTracks
from util import compute

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == '__main__':
    f_base = '%s/%s/' % (namelist.output_directory, namelist.exp_name)
    os.makedirs(f_base, exist_ok = True)
    print('Saving model output to %s' % f_base)
    shutil.copyfile('./namelist.py', '%s/namelist.py' % f_base)

    generate_land_masks.generate_land_masks()
    #compute.compute_downscaling_inputs()

    print('Running tracks for basin %s...' % sys.argv[1])
    if len(sys.argv) < 3:
        compute.run_downscaling(sys.argv[1], namelist.data_ts)
    else:
        for n in range(int(sys.argv[2])):
            compute.run_downscaling(sys.argv[1], namelist.data_ts)

        #plotLMI.plotLMI()
        #plotTracks.plotTracks(ntracks=namelist.tracks_per_year * int(sys.argv[2]))