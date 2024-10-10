import os
import sys

import pandas as pd
import random
cwd = os.path.dirname(os.path.realpath("."))
parent = os.path.dirname(cwd)
sys.path.append(cwd)
import namelist
def get_samples():
    """
    Get a sample of annual TC counts.
    The source file (`tccount.samples.csv`) was generated using a posterior predictive
    sampling routine from Bayesian MCMC fitting of a linear trend model to TC
    frequency in the Australian region.

    This used 42 years of TC activity (1981-2022), so returns 42 years of
    annual event numbers
    """
    src_directory = os.path.dirname(os.path.abspath(__file__))
    samplefile = os.path.join(src_directory, 'data', f'tccount.samples.csv')
    df = pd.read_csv(samplefile, index_col='draw')
    draw = random.randrange(20000)
    numtcs = df.loc[draw][['year', 'tcs']]
    return numtcs