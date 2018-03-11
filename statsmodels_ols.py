import subprocess as sp
import numpy as np
import statsmodels.api as sm

sp.call( 'cls', shell = True )

spector_data = sm.datasets.spector.load()


