__author__ = 'Zohar'
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
gurobipy.setParam(GRB.Param.MIPGap,0.01)
gurobipy.setParam("OutputFlag",0)
gurobipy.setParam("Threads",3)
gurobipy.setParam("MIPFocus",1)


START_NODE = 0
END_NODE = 1
FIRST_REAL_NODE = 2


def permute(L):
    return random.sample(L,len(L))
    #return L

'''
find min hamiltonaian path and transitive tournament
'''
def find_path_extended(G,weights,probs_instead_of_weights=False):

    # Create a new model
    m = Model("hamiltonian_cycle and acyclic tournament")

    # Create variables
    # edges
    x_vars = {}
    # nodes
    u_vars = {}
    # new edges
    z_vars = {}
    for var1 in permute(G.vertices()):
        for var2 in permute(G.vertices()):
            if var1 != var2:
                x_vars[(var1,var2)] = m.addVar(vtype='B', name="x_"+str(var1)+'_'+str(var2))
                z_vars[(var1,var2)] = m.addVar(vtype='B', name="z_"+str(var1)+'_'+str(var2))

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
    m.update()
    for var1 in G.vertices():
        for var2 in G.vertices():
            if var1 != var2:
                c = LinExpr([(1.0,u_vars[var1]),(-1.0,u_vars[var2]),(G.num_vertices(),x_vars[(var1,var2)])])

                m.addConstr(c,GRB.LESS_EQUAL,G.num_vertices()-1)

    for var1,var2 in itertools.combinations(G.vertices(),2):
        one_direction = LinExpr([(1.0, z_vars[(var1, var2)]),(1.0, z_vars[(var2, var1)])])
        m.addConstr(one_direction, GRB.EQUAL, 1)

        connect_vars = LinExpr([(1.0, x_vars[(var1, var2)]),(-1.0, z_vars[(var1, var2)])])
        m.addConstr(connect_vars, GRB.LESS_EQUAL, 0)
        connect_vars_inverse = LinExpr([(1.0, x_vars[(var2, var1)]),(-1.0, z_vars[(var2, var1)])])
        m.addConstr(connect_vars_inverse, GRB.LESS_EQUAL, 0)

    #rule out cycles with size 3
    for var1, var2, var3 in itertools.combinations(G.vertices(), 3):
        if var3 != var2 and var3 != var1 and var1 != var2:
            cycle = LinExpr([(1.0, z_vars[(var1, var2)]),(1.0, z_vars[(var2, var3)]), (1.0, z_vars[(var3, var1)])])
            inverse_cycle = LinExpr([(1.0, z_vars[(var2, var1)]),(1.0, z_vars[(var1, var3)]), (1.0, z_vars[(var3, var2)])])
            m.addConstr(cycle, GRB.LESS_EQUAL, 2)
            m.addConstr(inverse_cycle, GRB.LESS_EQUAL, 2)

    m.update()
    # Set objective
    #try:
    edge_weights = permute(G.get_edge_weights_extended(weights))
    if probs_instead_of_weights:
        all_probs = []
        for v in G.vertices():
            if v != END_NODE:
                batch_scores = [(e,w) for e,w in edge_weights if e[0] == v]
                S = logsumexp([x[1] for x in batch_scores])
                batch_scores = [(e,np.exp(w-S)) for e,w in batch_scores]
                all_probs.extend(batch_scores)
        edge_weights = all_probs
    objective = LinExpr([(weight_1,x_vars[edge]) for edge,weight_1, weight_2 in edge_weights] + [(weight_2,z_vars[edge]) for edge,weight_1, weight_2 in edge_weights])
    #except TypeError:
    #    return None

    m.setObjective(objective,GRB.MINIMIZE)
    m.update()
    #print("num of vars: " + str(m.numVars))
    #print("num of constraints: " + str(m.numConstrs))

    code = m.optimize()

    try:
        x=x_vars.items()
        z=z_vars.items()
        return ([k for k,v in x_vars.items() if v.x > 0.98],[k for k,v in z_vars.items() if v.x > 0.98])
    except GurobiError as e:
        #print ("gurobiError " + e.message)
        return None


def shuffle(L1,L2, L3):
    "Returns a random permutation of L1 and of L2 (the same permutation)"
    if len(L1) != len(L2) or len(L3) != len(L1):
        raise Exception('Incompatible arguments')
    perm = random.sample(range(len(L1)),len(L1))
    L1 = [L1[ind] for ind in perm]
    L2 = [L2[ind] for ind in perm]
    L3 = [L3[ind] for ind in perm]
    return L1,L2, L3




class StructuredPerceptronExtended:

    def __init__(self,num_features,num_iters,eta,averaged,vec_feature_names,calc_train_accuracy=False,\
                     greedy_inference=False,model_pickle=None,time_limit=5,\
                     probs_instead_of_weights=False,sigma=0.1):
        self._num_features = num_features
        self._weights = numpy.matlib.zeros(shape=(2,num_features))
        self._num_iters = num_iters
        self._averaged = averaged
        self._eta = eta
        self._vec_feature_names = vec_feature_names
        self._calc_train_acc = calc_train_accuracy
        self._greedy_inference = greedy_inference
        self._probs_insteadOf_weights = probs_instead_of_weights
        self._model_pickle = model_pickle
        self._interm_averaged_model = numpy.matlib.zeros(shape=(2,num_features))
        self._interm_averaged_model_num_instances = 0
        self._sigma = sigma
        self._gradient_iter = 0
        self._obj_iter = 0
        gurobipy.setParam("TimeLimit",time_limit)

    def save(self,filename):
        f = open(filename,'wb')
        pickle.dump(self,f)
        f.close()

    def set_time_limit(self,time_limit):
        print(str(time_limit)+' time limit set')
        gurobipy.setParam("TimeLimit",time_limit)

    def fit(self,vector_graphs,perms, perms_tournament):
        """
        vector_graph is a list of graphs where each edge corresponds to a vector (numpy vectors, could be sparse).
        perms is a list of permutations. Each permutation is a collection of edges in the graph.
        """
        if self._interm_averaged_model_num_instances > 0:
            # starting from a previously saved model
            sum_weight_vectors = self._interm_averaged_model * self._interm_averaged_model_num_instances
            done_instances = self._interm_averaged_model_num_instances
            print('x')
        else:
            done_instances = 0
            sum_weight_vectors = np.zeros(shape=self._weights.shape)

        vector_graphs, perms, perms_tournament = shuffle(vector_graphs,perms, perms_tournament)
        learning_curve_pairs = [(0,0) for x in vector_graphs]
        learning_curve_triples = [(0,0) for x in vector_graphs]
        learning_curve_quads = [(0,0) for x in vector_graphs]
        exact_matches = [0 for x in vector_graphs]
        seen_instances = [0 for x in vector_graphs]
        total_instances_seen = 0
        save_gap = 5000 #000 # 5000 # the maximum number of instances between pickles
        for ind in range(self._num_iters):
            instance_index = 0
            error_instances = []
            for G,correct_perm, correct_perm_tournament in zip(vector_graphs,perms, perms_tournament):
                if total_instances_seen >= done_instances:
                    if self._greedy_inference:
                        #This option is not used
                        predicted_perm = self.greedy_inference(G)
                    else:
                        prediction = find_path_extended(G,self._weights)

                        if prediction == None:
                            continue
                        predicted_perm, predicted_perm_tournament = prediction
                        predicted_perm_tournament.remove((0,1))
                    self._weights[0] = self._weights[0] + \
                        (self._eta * ( G.get_sum_path(correct_perm) - G.get_sum_path(predicted_perm)  )).todense()

                    self._weights[1] = self._weights[1] + \
                                    (self._eta * ( G.get_sum_path(correct_perm_tournament) - G.get_sum_path(predicted_perm_tournament))).todense()
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
            if self._model_pickle:
                f_pickle = open(self._model_pickle+'_averaged','w')
                pickle.dump(self,f_pickle,pickle.HIGHEST_PROTOCOL)
                f_pickle.close()
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
                                     args=(feature_vectors,correct_rows,batches,self._sigma,self),pgtol=0.01)
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
            return find_path_extended(event_graph,self._weights,True)
        else:
            return find_path_extended(event_graph,self._weights)

    def greedy_inference(self,event_graph):
        edge_weights = event_graph.get_edge_weights_extended(self._weights)
        if self._probs_insteadOf_weights:
            edge_weights = weights_to_probs(event_graph,edge_weights)
        predicted_perm = []
        predicted_perm_tournament = []
        non_visited_nodes = set([v for v in event_graph.vertices() if v not in [START_NODE,END_NODE]])
        cur_node = START_NODE
        while len(non_visited_nodes) > 0:
            possible_edges = [(cur_node,v) for v in non_visited_nodes]
            L = [(e,w1) for e,w1,w2 in edge_weights if e in possible_edges]
            np.random.shuffle(L)
            e_max = min(L,key=lambda x:x[1])[0]
            predicted_perm.append(e_max)
            non_visited_nodes.remove(e_max[1])
            cur_node = e_max[1]
        predicted_perm.append((cur_node,END_NODE))

        non_visited_nodes = set([v for v in event_graph.vertices() if v not in [START_NODE, END_NODE]])
        for v in non_visited_nodes:
            predicted_perm_tournament.append((START_NODE, v))
            predicted_perm_tournament.append((v, END_NODE))

        possible_edges = [(x,y) for x,y in itertools.permutations(non_visited_nodes, 2)]
        options = [(x,y) for x,y in itertools.combinations(non_visited_nodes, 2)]

        L = [(e, w2) for e, w1, w2 in edge_weights if e in possible_edges]
        for x,y in options:
            flag = False
            if (x,y) not in predicted_perm_tournament and (y,x) not in predicted_perm_tournament:
                option1 = [w2 for e,w2 in L if e==(x,y)][0]
                option2 = [w2 for e,w2 in L if e==(y,x)][0]
                if option1 < option2:
                    for v in non_visited_nodes:
                        if ((x,v) in predicted_perm_tournament and (v,y) in predicted_perm_tournament) or\
                            ((y, v) in predicted_perm_tournament and (v, x) in predicted_perm_tournament):
                            flag = True
                            break
                    if flag:
                        predicted_perm_tournament.append(option2)
                    else:
                        predicted_perm_tournament.append(option1)


        return predicted_perm, predicted_perm_tournament

    def binary_classification(self,test_samples,test_labels):
        """
        Computes how often is the correct path more highly ranked than a random one.
        """
        num_correct = 0
        num_samples = 0
        for sample,perm in zip(test_samples,test_labels):
            if len(sample.vertices()) == 3:
                continue
            minus_edge_weights = sample.get_edge_weights(self._weights)
            S_correct = 0.0
            S_all = 0.0
            for e,w in minus_edge_weights:
                if e == (START_NODE,END_NODE):
                    continue
                if e in perm:
                    S_correct += -1 * w
                S_all += -1 * w
            S_avg = S_all / (len(sample.vertices()) - 2)
            if S_correct > S_avg:
                num_correct += 1
            elif S_correct == S_avg:
                num_correct += 0.5
            num_samples += 1
            if num_samples % 50 == 0:
                print(num_samples)
        return 1.0 * num_correct / num_samples

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
            prediction = self.predict(G)
            if prediction is None:
                predicted_perm, predicted_perm_tournament = self.greedy_inference(G)
                print('Test instance '+str(total_instances)+': ILP failed. Reverting to greedy method.')
            else:
                predicted_perm, predicted_perm_tournament = prediction
            print('Test instance '+str(total_instances)+' processed')
            if output_file:
                output_file.write('Instance #'+str(total_instances)+'\n'+ \
                                      str(predicted_perm)+'\n'+str(correct_perm)+'\n========\n')
            total_instances += 1
            if total_instances % 100 == 0 and output_file:
                output_file.flush()
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
                             all_batches,sigma=None,perceptron=None):
    """
    Returns the minus gradient of the log likelihood.
    vector_graphs is a list of all the vectors in the training data.
    correct_rows are the row indices of the correct edges.
    batches is the list of indices corresponding to each vertex.
    """
    if perceptron:
        perceptron._gradient_iter += 1
    g = None
    index = 0
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
        index += 1
        if index % 100 == 0:
            print('Gradient '+str(index)+' processed')
    g = numpy.ndarray.flatten(numpy.asarray(g)) / len(all_vector_graphs)
    if sigma != None:
        g = g + sigma * w
    print('Gradient norm:'+str(scipy.linalg.norm(g)))
    sys.stdout.flush()
    if perceptron and perceptron._model_pickle:
        if perceptron._gradient_iter % 5 == 0:
            perceptron._weights = numpy.reshape(w,(1,perceptron._num_features))
            perceptron.save(perceptron._model_pickle+'_'+str(perceptron._gradient_iter))
    return g

def scipy_minus_objective(w,all_vector_graphs,all_correct_rows,\
                              all_batches,sigma=None,perceptron=None):
    """
    Returns the minus log-likelihood of the data.
    vector_graphs is a matrix of all the vectors in the training data.
    correct_rows are the row indices of the correct edges.
    batches is the list of indices corresponding to each vertex.
    """
    if perceptron:
        perceptron._obj_iter += 1
    obj = 0.0
    index = 0
    for vector_graphs,correct_rows,batches in zip(all_vector_graphs,all_correct_rows,all_batches):
        all_scores =  vector_graphs * w
        sum_log_Z = 0.0
        for batch in batches:
            batch_scores = all_scores[batch]
            sum_log_Z += logsumexp(batch_scores) #np.log(np.exp(batch_scores).sum())
        obj += all_scores[correct_rows].sum() - sum_log_Z
        index += 1
        if index % 100 == 0:
            print('Objective '+str(index)+' processed')
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


