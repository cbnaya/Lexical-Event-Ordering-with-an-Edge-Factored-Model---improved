from gurobipy import *
import cPickle as pickle
import pdb, itertools, driver, time
import bisect
import random, time, math
import numpy as np
import numpy.matlib
import scipy.optimize
from scipy.misc import logsumexp
random.seed(2)

START_NODE = 0
END_NODE = 1
FIRST_REAL_NODE = 2


def find_min_hamiltonian_path(G,weights,probs_instead_of_weights=False):
    """
    finds the minimal hamiltonian path in G between the start_node
    and the end_node. returns the set of edges participating in the
    path.
    """

    # Create a new model
    m = Model("hamiltonian_cycle")
   
    # Create variables
    x_vars = {}
    u_vars = {}
    for var1 in G.vertices():
        for var2 in G.vertices():
            if var1 != var2:
                x_vars[(var1,var2)] = m.addVar(vtype='B', name="x_"+str(var1)+'_'+str(var2))
        u_vars[var1] = m.addVar(vtype=GRB.INTEGER, name="u_"+str(var1))
    m.update()
    
    for var in G.vertices():
        if var != START_NODE:
            cur_incoming = LinExpr([(1.0,v) for k,v in x_vars.items() if (k[1] == var)])
            #print(cur_incoming)
            m.addConstr(cur_incoming,GRB.EQUAL,1.0)
        
        if var != END_NODE:
            cur_outgoing = LinExpr([(1.0,v) for k,v in x_vars.items() if (k[0] == var)])
            #print(cur_outgoing)
            m.addConstr(cur_outgoing,GRB.EQUAL,1.0)
    
    for var1 in G.vertices():
        for var2 in G.vertices():
            if var1 != var2:
                c = LinExpr([(1.0,u_vars[var1]),(-1.0,u_vars[var2]),(G.num_vertices(),x_vars[(var1,var2)])])
                #print(c)
                m.addConstr(c,GRB.LESS_EQUAL,G.num_vertices()-1)
    
    # Set objective
    #try:
    edge_weights = G.get_edge_weights(weights)
    if probs_instead_of_weights:
        all_probs = []
        for v in G.vertices():
            if v != END_NODE:
                batch_scores = [(e,w) for e,w in edge_weights if e[0] == v]
                S = logsumexp([x[1] for x in batch_scores])
                batch_scores = [(e,np.exp(w-S)) for e,w in batch_scores]
                all_probs.extend(batch_scores)
        edge_weights = all_probs
    objective = LinExpr([(weight,x_vars[edge]) for edge,weight in edge_weights])
    #except TypeError:
    #    return None

    m.setObjective(objective,GRB.MINIMIZE)    
    code = m.optimize()
    
    try:
        return [k for k,v in x_vars.items() if v.x > 0.98]
    except GurobiError:
        return None

def shuffle(L1,L2):
    "Returns a random permutation of L1 and of L2 (the same permutation)"
    if len(L1) != len(L2):
        raise Exception('Incompatible arguments')
    perm = random.sample(range(len(L1)),len(L1))
    L1 = [L1[ind] for ind in perm]
    L2 = [L2[ind] for ind in perm]
    return L1,L2

#class LocalClassifier:
#    def __init__(self,num_features,num_iters,eta,averaged,vec_feature_names):
#    def save(self,filename):
#    def fit(self,vector_graphs,perms)
#    def predict(self,event_graph)
#def fit()
#def predict()
#def test_on_data(
#def test_on_data_from_file(


            
class StructuredPerceptron:
    
    def __init__(self,num_features,num_iters,eta,averaged,vec_feature_names,calc_train_accuracy=False,\
                     greedy_inference=False,model_pickle=None,time_limit=5,\
                     probs_instead_of_weights=False,sigma=0.1):
        self._num_features = num_features
        self._weights = numpy.matlib.zeros(shape=(1,num_features))
        self._num_iters = num_iters
        self._averaged = averaged
        self._eta = eta
        self._vec_feature_names = vec_feature_names
        self._calc_train_acc = calc_train_accuracy
        self._greedy_inference = greedy_inference
        self._probs_insteadOf_weights = probs_instead_of_weights
        self._model_pickle = model_pickle
        self._interm_averaged_model = numpy.matlib.zeros(shape=(1,num_features))
        self._interm_averaged_model_num_instances = 0
        self._sigma = sigma
        gurobipy.setParam("TimeLimit",time_limit)
        gurobipy.setParam(GRB.Param.MIPGap,0.01)
        gurobipy.setParam("OutputFlag",0)
        gurobipy.setParam("Threads",3)    
        gurobipy.setParam("MIPFocus",1)
    
    def save(self,filename):
        f = open(filename,'wb')
        pickle.dump(self,f)
        f.close()

    def set_time_limit(self,time_limit):
        gurobipy.setParam("TimeLimit",time_limit)
    
    def fit(self,vector_graphs,perms):
        """
        vector_graph is a list of graphs where each edge corresponds to a vector (numpy vectors, could be sparse).
        perms is a list of permutations. Each permutation is a collection of edges in the graph.
        """
        vector_graphs, perms = shuffle(vector_graphs,perms)
        learning_curve_pairs = [(0,0) for x in vector_graphs]
        learning_curve_triples = [(0,0) for x in vector_graphs]
        learning_curve_quads = [(0,0) for x in vector_graphs]
        sum_weight_vectors = np.zeros(shape=self._weights.shape)
        exact_matches = [0 for x in vector_graphs]
        seen_instances = [0 for x in vector_graphs]
        total_instances_seen = 0
        save_gap = 5000 # 5000 # the maximum number of instances between pickles
        for ind in range(self._num_iters):
            instance_index = 0
            error_instances = []
            for G,correct_perm in zip(vector_graphs,perms):
                if self._greedy_inference:
                    predicted_perm = self.greedy_inference(G)
                else:
                    predicted_perm = find_min_hamiltonian_path(G,self._weights)
                    if predicted_perm == None:
                        continue
                self._weights = self._weights + \
                    (self._eta * ( G.get_sum_path(correct_perm) - G.get_sum_path(predicted_perm) )).todense()
                try:
                    predicted_perm_c = convert_edges_perm(predicted_perm)
                    correct_perm_c = convert_edges_perm(correct_perm)
                    learning_curve_pairs[instance_index] = \
                        num_agreeing_tuples(predicted_perm_c,correct_perm_c,2)
                    learning_curve_triples[instance_index] = \
                        num_agreeing_tuples(predicted_perm_c,correct_perm_c,3)
                    learning_curve_quads[instance_index] = \
                        num_agreeing_tuples(predicted_perm_c,correct_perm_c,4)
                    seen_instances[instance_index] = 1
                    exact_matches[instance_index] = (1 if set(predicted_perm) == set(correct_perm) else 0)                    
               
                    self.print_learning_curve(ind,instance_index,learning_curve_pairs,learning_curve_triples,\
                                                  learning_curve_quads,1.0*sum(exact_matches)/sum(seen_instances))
                    sum_weight_vectors = sum_weight_vectors + self._weights
                except Exception:
                    error_instances.append(instance_index)
                    print('Incorrect decoding in training. Instance '+str(ind)+' '+str(instance_index))
                instance_index += 1
                total_instances_seen += 1
                if len(perms) > save_gap and self._model_pickle and total_instances_seen % save_gap == 0:
                    self._interm_averaged_model = sum_weight_vectors / total_instances_seen
                    self._interm_averaged_model_num_instances = total_instances_seen
                    f_pickle = open(self._model_pickle+'_'+str(total_instances_seen)+'inst','w')
                    pickle.dump(self,f_pickle,pickle.HIGHEST_PROTOCOL)
                    f_pickle.close()
            if self._calc_train_acc:
                print('Training accuracy iteration #'+str(ind)+':'+str(self.test_on_data(vector_graphs,perms)))
            if self._model_pickle:
                self._interm_averaged_model = sum_weight_vectors / ((ind+1) * len(perms))
                self._interm_averaged_model_num_instances = (ind+1) * len(perms)
                f_pickle = open(self._model_pickle+'_iter'+str(ind),'w')
                pickle.dump(self,f_pickle,pickle.HIGHEST_PROTOCOL)
                f_pickle.close()
        if self._averaged:
            self._weights = sum_weight_vectors / (self._num_iters * len(perms))
            self._interm_averaged_model = self._weights
            self._interm_averaged_model_num_instances = (self._num_iters * len(perms))
            if self._calc_train_acc:
                print('Training accuracy (after averaging):'+str(self.test_on_data(vector_graphs,perms)))
        return error_instances

    def fit_mbr(self,vector_graphs,perms):
        feature_vectors = [G._vecs for G in vector_graphs]
        
        batches = [[[e_ind for e_ind,e in enumerate(G.edges()) if e[0] == v] \
                        for v in G.vertices() if v != END_NODE] for G in vector_graphs]

        correct_rows = []
        for G,perm in zip(vector_graphs,perms):
            correct_rows.append([index for index,e in enumerate(G.edges()) if e in perm])
        
        x0 = self._weights
        w = scipy.optimize.fmin_l_bfgs_b(scipy_minus_objective, x0, fprime=scipy_minus_gradient, 
                                     args=(feature_vectors,correct_rows,batches,self._sigma),pgtol=0.01)
        print('LBFGS Converged.')
        sys.stdout.flush()
        self._weights = numpy.reshape(w[0],(1,self._num_features))

        if self._model_pickle:
            self._interm_averaged_model = self._weights
            f_pickle = open(self._model_pickle+'_bfgs','w')
            pickle.dump(self,f_pickle,pickle.HIGHEST_PROTOCOL)
            f_pickle.close()
            
    def print_vec(self,v,prefix):
        sys.stdout.write(prefix+'\n')
        for ind in range(v.shape[1]):
            if v[0,ind] != 0:
                sys.stdout.write(str(self._vec_feature_names[ind])+':'+str(v[0,ind])+' ')
        sys.stdout.write('\n')
    
    def print_learning_curve(self,iter_ind,instance_ind,all_pair_swaps,all_triple_swaps,all_quad_swaps,exact_match):
        macro_avg_pair_swaps = 1.0 * sum([p[0] for p in all_pair_swaps]) / sum([p[1] for p in all_pair_swaps])
        macro_avg_triple_swaps = 1.0 * sum([p[0] for p in all_triple_swaps]) / sum([p[1] for p in all_triple_swaps])
        macro_avg_quad_swaps = 1.0 * sum([p[0] for p in all_quad_swaps]) / sum([p[1] for p in all_quad_swaps])
        print(' '.join(['Learning curve:',str(iter_ind),str(instance_ind),str(macro_avg_pair_swaps),\
                            str(macro_avg_triple_swaps),str(macro_avg_quad_swaps),str(exact_match)]))

    """
    def old_local_fit(self,vector_graphs,perms):

        same as fit, but does the fitting not using a structured perceptron, but 
        using a regular perceptron.
        vector_graphs, perms = shuffle(vector_graphs,perms)
        sum_weight_vectors = np.zeros(shape=self._weights.shape)
        for ind in range(self._num_iters):
            instance_index = 0
            for G,correct_perm in zip(vector_graphs,perms):
                edge_weights = dict([(e,-w) for e,w in G.get_edge_weights(self._weights)])
                #labeled_edges = [((e[1],e[0]),-1) for e in G.edges()] + \
                #    [(e,1) for e in predicted_perm]
                for e in correct_perm:
                    if edge_weights[e] <= 0: # if incorrectly negative
                        self._weights = self._weights + self._eta * G.get_vec(e)
                    inv_e = (e[1],e[0])
                    if inv_e in edge_weights and edge_weights[inv_e] >= 0: # if incorrectly negative
                        self._weights = self._weights - self._eta * G.get_vec(inv_e)
                print(instance_index,len(self._vec_feature_names),self._weights.size,np.linalg.norm(self._weights))
                sum_weight_vectors = sum_weight_vectors + self._weights
                instance_index += 1
        if self._averaged:
            self._weights = sum_weight_vectors / (self._num_iters * len(perms))
            if self._calc_train_acc:
                print('Training accuracy (after averaging):'+ str(self.test_on_data(vector_graphs,perms)) )
    """
    
    def local_fit(self,vector_graphs,perms):
        """
        same as fit, but treats the problem as a binary problem of distinguishing between
        correct and incorrect edges. 
        """
        vector_graphs, perms = shuffle(vector_graphs,perms)
        sum_weight_vectors = np.zeros(shape=self._weights.shape)
        for ind in range(self._num_iters):
            instance_index = 0
            for G,correct_perm in zip(vector_graphs,perms):
                predicted_perm = self.predict(G)
                if predicted_perm == None:
                    continue
                for e in G.edges():
                    if e in correct_perm and e not in predicted_perm:
                        self._weights = self._weights + self._eta * G.get_vec(e).transpose()
                    if e not in correct_perm and e in predicted_perm:
                        self._weights = self._weights - self._eta * G.get_vec(e).transpose()
                print('Instance #'+str(instance_index)+' '+str(self._weights.size,np.linalg.norm(self._weights)))
                sum_weight_vectors = sum_weight_vectors + self._weights
                instance_index += 1
        if self._averaged:
            self._weights = sum_weight_vectors / (self._num_iters * len(perms))
            if self._calc_train_acc:
                print('Training accuracy (after averaging):'+ str(self.test_on_data(vector_graphs,perms)) )
    
    def predict(self,event_graph):
        """
        Returns the predicted permutation for the event graph.
        """
        if self._greedy_inference:
            return self.greedy_inference(event_graph)
        elif self._probs_insteadOf_weights:
            return find_min_hamiltonian_path(event_graph,self._weights,True)
        else:
            return find_min_hamiltonian_path(event_graph,self._weights)
    
    def greedy_inference(self,event_graph):
        edge_weights = event_graph.get_edge_weights(self._weights)
        if self._probs_insteadOf_weights:
            edge_weights = weights_to_probs(event_graph,edge_weights)
        predicted_perm = []
        non_visited_nodes = set([v for v in event_graph.vertices() if v not in [START_NODE,END_NODE]])
        cur_node = START_NODE
        while len(non_visited_nodes) > 0:
            possible_edges = [(cur_node,v) for v in non_visited_nodes]
            L = [(e,w) for e,w in edge_weights if e in possible_edges]
            np.random.shuffle(L)
            e_max = min(L,key=lambda x:x[1])[0]
            predicted_perm.append(e_max)
            non_visited_nodes.remove(e_max[1])
            cur_node = e_max[1]
        predicted_perm.append((cur_node,END_NODE))
        return predicted_perm

    def greedy_inference2(self,event_graph):
        """
        Greedy inference, but a different heuristic. It first selects the heaviest edge in the graph,
        then the second etc.
        """
        edge_weights = event_graph.get_edge_weights(self._weights)
        if self._probs_insteadOf_weights:
            edge_weights = weights_to_probs(event_graph,edge_weights)
        predicted_perm = []
        vertex_out = set(event_graph.vertices())
        vertex_in = set(event_graph.vertices())
        while len(vertex_out) > 1 and len(vertex_in) > 1:
            possible_edges_weights = [(e,w) for e,w in edge_weights \
                                          if e[0] in vertex_out and e[1] in vertex_in]
            np.random.shuffle(possible_edges_weights)
            e_max = max(possible_edges_weights,key=lambda x:x[1])[0]
            vertex_out.remove(e_max[0])
            vertex_in.remove(e_max[1])
            predicted_perm.append(e_max)
        predicted_perm.extend([(START_NODE,v) for v in vertex_in])
        predicted_perm.extend([(END_NODE,v) for v in vertex_in])
        return predicted_perm
    
    def objective(self,vector_graphs,perms):
        pass
    
    def test_on_data(self,vector_graphs,correct_perms,output_file=None):
        total_instances = 0
        exact_match = 0
        all_pair_swaps = []
        all_triple_swaps = []
        all_quad_swaps = []
        errs = []
        for G,correct_perm in zip(vector_graphs,correct_perms):
            predicted_perm = self.predict(G)
            if predicted_perm is None:
                predicted_perm = self.greedy_inference(G)
                print('Test instance '+str(total_instances)+': ILP failed. Reverting to greedy method.')
            print('Test instance '+str(total_instances)+' processed')
            if output_file:
                output_file.write('Instance #'+str(total_instances)+'\n'+ \
                                      str(predicted_perm)+'\n'+str(correct_perm)+'\n========\n')
            total_instances += 1
            try:
                if set(predicted_perm) == set(correct_perm):
                    exact_match += 1
                predicted_perm_c = convert_edges_perm(predicted_perm)
                correct_perm_c = convert_edges_perm(correct_perm)
                pair_swaps = num_agreeing_tuples(predicted_perm_c,correct_perm_c,2)
                triple_swaps = num_agreeing_tuples(predicted_perm_c,correct_perm_c,3)
                quad_swaps = num_agreeing_tuples(predicted_perm_c,correct_perm_c,4)
                all_pair_swaps.append(pair_swaps)
                all_triple_swaps.append(triple_swaps)
                all_quad_swaps.append(quad_swaps)
                self.print_learning_curve('Test',total_instances,all_pair_swaps,all_triple_swaps,\
                                              all_quad_swaps,1.0*exact_match/total_instances)
            except Exception as e:
                errs.append(total_instances)

        macro_avg_pair_swaps = 1.0 * sum([p[0] for p in all_pair_swaps]) / sum([p[1] for p in all_pair_swaps])
        macro_avg_triple_swaps = 1.0 * sum([p[0] for p in all_triple_swaps]) / sum([p[1] for p in all_triple_swaps])
        macro_avg_quad_swaps = 1.0 * sum([p[0] for p in all_quad_swaps]) / sum([p[1] for p in all_quad_swaps])
        #micro_avg_pair_swaps = 1.0 * sum(kandall_tau) / len(kandall_tau)
        
        return 1.0 * exact_match / total_instances, macro_avg_pair_swaps, macro_avg_triple_swaps, macro_avg_quad_swaps, errs


    
    """
    def test_on_data_from_file(self,feat_extractor,test_file):
        total_instances = 0
        exact_match = 0
        kandall_tau = []
        swap_pairs = []
        for recipe,correct_perm in driver.read_recipe(test_file):
            G = feat_extractor.extract_features(recipe)
            predicted_perm = self.predict(G)
            total_instances += 1
            if predicted_perm == correct_perm:
                exact_match += 1
            tau, pair_swaps = get_permutation_scores(predicted_perm,correct_perm)
            kandall_tau.append(tau)
            swap_pairs.append(pair_swaps)

        macro_avg_pair_swaps = 1.0 * sum([p[0] for p in swap_pairs]) / sum([p[1] for p in swap_pairs])
        micro_avg_pair_swaps = 1.0 * sum(kandall_tau) / len(kandall_tau)

        return 1.0 * exact_match / total_instances, macro_avg_pair_swaps, micro_avg_pair_swaps
    """

def weights_to_probs(G,edge_weights):
    """
    Receives a graph and a list of pairs of edges and weights. Normalizes the score of each node to 1.
    """
    new_edge_weights = []
    for v in G.vertices():
        outgoing_edges = [e for e in G.edges() if e[0] == v]
        outgoing_edge_weights = [(e,math.exp(w)) for e,w in edge_weights if e in outgoing_edges]
        S = sum([x[1] for x in outgoing_edge_weights])
        new_edge_weights.extend([(e,w/S) for e,w in outgoing_edge_weights])
    return new_edge_weights

def get_permutation_scores(predicted_perm,correct_perm):
    #micro = []
    macro = []
    for k in [2,3,4]:
        tuple_swaps = num_agreeing_tuples(convert_edges_perm(predicted_perm),convert_edges_perm(correct_perm),k)
        macro.append(tuple_swaps)
        #num_swaps(convert_edges_perm(predicted_perm),convert_edges_perm(correct_perm))
        #if tuple_swaps[1] > 0:
        #    micro.append(1.0 * pair_swaps[0] / pair_swaps[1])
        #else:
        #    micro.append(None)
    return macro


def convert_edges_perm(edges):
    """
    receives a list of pairs (edges) that form a permutation. 
    returns a list of the nodes according to the permutation (starting with START_NODE).
    """
    L = dict(edges)
    output = [START_NODE]
    while output[-1] != END_NODE:
        output.append(L[output[-1]])
    if len(edges) + 1 != len(output):
        raise Exception()
    return output


def num_swaps(perm1,perm2):
    """
    Determines the number of swaps between perm1 and perm2.
    Each of the lists is a list of unique numbers.
    The first and the last element are assumed to be fixed and are excluded.
    """
    if len(perm1) != len(perm2) or len(set(perm1)) != len(perm1) or len(set(perm2)) != len(perm2):
        raise Exception("Incompatible lists")
    perm1 = perm1[1:-1]
    perm2 = perm2[1:-1]
    agree = 0
    disagree = 0
    for pair1,pair2 in itertools.combinations(zip(perm1,perm2),2):
        if (pair1[0] > pair2[0] and pair1[1] > pair2[1]) or (pair1[0] < pair2[0] and pair1[1] < pair2[1]):
            agree += 1
        else:
            disagree += 1
    return agree, agree+disagree

def num_agreeing_tuples(perm1,perm2,k):
    """
    same as num_swaps, but for k-tuples.
    """
    if len(perm1) != len(perm2) or len(set(perm1)) != len(perm1) or len(set(perm2)) != len(perm2):
        raise Exception("Incompatible lists")
    perm1 = perm1[1:-1]
    perm2 = perm2[1:-1]
    agree = 0
    disagree = 0
    for index_tuple in itertools.combinations(range(len(perm1)),k):
        tuple1 = [perm1[ind] for ind in index_tuple]
        tuple2 = [perm2[ind] for ind in index_tuple]
        ordered_tuple1 = sorted(tuple1)
        ordered_tuple2 = sorted(tuple2)
        ordinals1 = [bisect.bisect_left(ordered_tuple1,x) for x in tuple1]
        ordinals2 = [bisect.bisect_left(ordered_tuple2,x) for x in tuple2]
        if ordinals1 == ordinals2:
            agree += 1
        else:
            disagree += 1
    return agree,agree+disagree


#####################################################
# SCIPY METHODS
#####################################################

def scipy_minus_gradient(w,all_vector_graphs,all_correct_rows,\
                             all_batches,sigma=None):
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
    g = numpy.ndarray.flatten(numpy.asarray(g)) / len(all_vector_graphs)
    if sigma != None:
        g = g + sigma * w
    print('Gradient norm:'+str(scipy.linalg.norm(g)))
    return g

def scipy_minus_objective(w,all_vector_graphs,all_correct_rows,\
                              all_batches,sigma=None):
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
    obj = obj / len(all_vector_graphs)
    if sigma != None:
        obj += - 0.5 * sigma * (w * w).sum()
    print('Objective:'+str(obj))
    return -1.0 * obj

def exp_and_normalize_vec(v):
    S = logsumexp(v)
    return np.exp(v - S), S



if __name__ == '__main__':
    print(num_agreeing_tuples([],[],4))
    S = 0
    S2 = 0 
    for s in range(200):
        random.seed(s)
        L1 = random.sample(range(20),20)
        L2 = random.sample(range(20),20)
        S += num_agreeing_tuples(L1,L2,4)[0]
        S2 += num_agreeing_tuples(L1,L2,4)[1]
    print(S, S2)
    
