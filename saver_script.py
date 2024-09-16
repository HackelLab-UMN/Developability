import time
import submodels_module as mb
import load_format_data
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import sys
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import sampling_modules as sm
import plot_modules as pm
# compiled optimizer
import matplotlib as mpl
mpl.use('Agg')
from contextlib import contextmanager




df=pd.read_pickle(pm.make_file_name(dir_name='Nb_sequences_1000_Nbsteps_10_Nb_loops_1000',file_description='times',fileformat='pkl'))

times=sm.convert2numpy(df=df,field='0th loop')

print(np.sum(times,axis=0))
print(df)
#pm.plot_hist(dir_name='Nb_sequences_1000_Nbsteps_10_Nb_loops_1000',j=1000,i=10,seq=df)

