import cPickle as pickle
import numpy as np
from Counter import Counter
from scipy import stats
from scipy import sparse

"""
Receives two files, each of which is a dump of a type_arguments.PredPairRepository.
Each such repository contains a matrix of representations of pairs 

We apply CCA and compute a transformation over the two vectors. This transformation
allows us to extract the latent representation z (see Bach and Jordan, 2006) for each pair.
"""


