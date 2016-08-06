import cPickle as pickle
import scipy
import pdb, itertools, driver, time, sys
import bisect
import random, time, math
import numpy as np
import numpy.matlib
from scipy.misc import logsumexp
random.seed(2)

START_NODE = 0
END_NODE = 1
FIRST_REAL_NODE = 2
iter_index = 0

#####################################################
# SCIPY METHODS
#####################################################


def scipy_minus_gradient(w,all_vector_graphs,all_correct_rows,all_batches):
    """
    Returns the minus gradient of the log likelihood.
    vector_graphs is a list of all the vectors in the training data.
    correct_rows are the row indices of the correct edges.
    batches is the list of indices corresponding to each vertex.
    """
    g = None
    for vector_graphs,correct_rows,batches in zip(all_vector_graphs,all_correct_rows,all_batches):
        first_term = vector_graphs[correct_rows,:].sum(axis=0)
        all_scores = vector_graphs * w
        all_probs = []
        for batch in batches:
            batch_scores = all_scores[batch]
            S = logsumexp(batch_scores)
            all_probs.append(np.exp(batch_scores - S))
        all_probs = numpy.hstack(all_probs)
        second_term = all_probs * vector_graphs
        if g is None:
            g = second_term - first_term
        else:
            g = g + second_term - first_term
    g = numpy.ndarray.flatten(numpy.asarray(g))
    print('Gradient norm '+str(iter_index)+':'+str(scipy.linalg.norm(g)))
    global iter_index
    iter_index += 1
    return g

def scipy_minus_objective(w,all_vector_graphs,all_correct_rows,all_batches):
    """
    Returns the minus log-likelihood of the data.
    vector_graphs is a matrix of all the vectors in the training data.
    correct_rows are the row indices of the correct edges.
    batches is the list of indices corresponding to each vertex.
    """
    obj = 0.0
    for vector_graphs,correct_rows,batches in zip(all_vector_graphs,all_correct_rows,all_batches):
        all_scores =  vector_graphs * w
        sum_log_Z = 0.0
        for batch in batches:
            batch_scores = all_scores[batch]
            sum_log_Z += logsumexp(batch_scores) #np.log(np.exp(batch_scores).sum())
        obj += all_scores[correct_rows].sum() - sum_log_Z
    print('Objective:'+str(obj))
    return -1.0 * obj

def exp_and_normalize_vec(v):
    S = logsumexp(v)
    return np.exp(v - S), S



#####################################################
# The decoder
#####################################################
            
class MBRDecoder:
    
    def __init__(self,num_features,vec_feature_names,model_pickle=None):
        self._num_features = num_features
        self._weights = numpy.matlib.zeros(shape=(1,num_features))
        self._vec_feature_names = vec_feature_names
        self._model_pickle = model_pickle
    
    def fit_mbr(self,vector_graphs,perms):
        feature_vectors = [G._vecs for G in vector_graphs]
        
        for g in feature_vectors:
            g[g == -1000000] = -100
        
        batches = [[[e_ind for e_ind,e in enumerate(G.edges()) if e[0] == v] \
                        for v in G.vertices() if v != END_NODE] for G in vector_graphs]

        correct_rows = []
        for G,perm in zip(vector_graphs,perms):
            correct_rows.append([index for index,e in enumerate(G.edges()) if e in perm])
        
        x0 = self._weights
        w = scipy.optimize.fmin_l_bfgs_b(scipy_minus_objective, x0, fprime=scipy_minus_gradient, 
                                     args=(feature_vectors,correct_rows,batches),retall=True,
                                     gtol=0.01) #,maxiter=1000)
        #_bfgs
        print('BFGS Converged.')
        sys.stdout.flush()
        self._weights = w[0]

        if self._model_pickle:
            self._interm_averaged_model = self._weights
            f_pickle = open(self._model_pickle+'_bfgs','w')
            pickle.dump(self,f_pickle,pickle.HIGHEST_PROTOCOL)
            f_pickle.close()





if __name__ == '__main__':
    f_pickle = open(sys.argv[1])
    training_samples,training_labels,test_samples,test_labels,vec_feature_names = pickle.load(f_pickle)
    f_pickle.close()
    training_samples = training_samples[:100]
    test_samples = test_samples[:100]
    
    d = MBRDecoder(len(vec_feature_names),vec_feature_names,'mbr_debug.model')
    d.fit_mbr(training_samples,training_labels)

