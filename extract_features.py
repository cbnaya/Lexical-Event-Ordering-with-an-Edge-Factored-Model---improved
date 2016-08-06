import preprocess_data, decoder
import random
import sequencer
import analyze_events
import gurobipy
from scipy import stats
from scipy import sparse
import train_lda, itertools
import numpy as np
import pdb

# ['MEAN_ORDINAL','PMI','PMI_INV','TRANSITION_MOD','PRED_LEX12','PRED1_OBJ2',\
#     'PRED2_OBJ1','OBJ1_OBJ2','BROWN_CL50_PRED1','BROWN_CL50_PRED2','BROWN_CL50_PRED12',\
#     'BROWN_CL50_ARG1','BROWN_CL50_ARG2','BROWN_CL50_ARG12']

LEX_FEATURES = ['PRED_LEX12','PRED1_OBJ2','PRED2_OBJ1','OBJ1_OBJ2'] #,'PRED_OBJ12','SECONDARY']
FREQ_FEATURES = ['MEAN_ORDINAL','PMI','PMI_INV','TRANSITION_MOD','TRANSITION_MOD_INV']
BROWN_FEATURES = ['BROWN_CL50_PRED1','BROWN_CL50_PRED2','BROWN_CL50_PRED12', \
                      'BROWN_CL50_ARG1','BROWN_CL50_ARG2','BROWN_CL50_ARG12']
DEFAULT_FEATURES = LEX_FEATURES + FREQ_FEATURES + BROWN_FEATURES
#DEFAULT_FEATURES = LEX_FEATURES

UNKNOWN_WORD = 'UNKNOWN_WORD'
UNKNOWN_CLUSTER_COMBINATION = 'UNKNOWN_CLUSTER_COMBINATION'
BROWN_CLUSTERS50_FILENAME = 'resources/brown_clusters50_training_data'
EPSILON_LOG_PROB = 0.001



class FeatureExtractor:

    def __init__(self,training_file,num_topics_arg_lda,num_topics_event_lda,transition_max_path_len,\
                     freq_thresh_linear,high_pmi_thresh_linear,low_pmi_thresh_linear,freq_thresh_linker,\
                     high_pmi_thresh_linker,low_pmi_thresh_linker,min_string_feat_freq,features=DEFAULT_FEATURES):
        """
        Loads the transition matrices and LDA models.
        """
        self._transition_max_path_len = transition_max_path_len
        self._training_file = training_file
        self._verb_list = preprocess_data.get_target_verb_list(preprocess_data.TARGET_VERB_FILE)
        if 'TRANSITION' in DEFAULT_FEATURES:
            self._transition_matrices, self._seqs = \
                analyze_events.get_transition_matrix(training_file,self._verb_list,transition_max_path_len,\
                                                         freq_thresh_linear,high_pmi_thresh_linear,\
                                                         low_pmi_thresh_linear,freq_thresh_linker,\
                                                         high_pmi_thresh_linker,low_pmi_thresh_linker)
        if 'TRANSITION_MOD' in DEFAULT_FEATURES or 'PMI' in DEFAULT_FEATURES or 'PMI_INV' in DEFAULT_FEATURES or \
                'MEAN_ORDINAL' in DEFAULT_FEATURES or 'FREQ' in DEFAULT_FEATURES or \
                'TRANSITION_MOD_INV' in DEFAULT_FEATURES:
            self._transition_stats = analyze_events.get_transition_stats(training_file,self._verb_list)
        
        self._num_topics_arg_lda = num_topics_arg_lda
        self._num_topics_event_lda = num_topics_event_lda
        self._feature_names = ['FREE_COEF']
        self._feature_types = ['float']
        self._vec_feature_names = sequencer.sequencer()
        self._features = features
        self._freq_thresh_linear = freq_thresh_linear
        self._freq_thresh_linker = freq_thresh_linker
        
        if ('EVENT_LDA1' in features or 'EVENT_LDA2' in features) and (num_topics_event_lda > 0):
            self._event_lda = train_lda.train_event_lda(training_file,num_topics_event_lda,training_file+\
                                                            '.event_lda_model')

        self._word_to_brown_cluster = read_brown_clusters(BROWN_CLUSTERS50_FILENAME)

        for feature in features:
            if feature == 'TRANSITION':
                new_features = [('TRANSITION',linker,ind) for linker,ind in self._transition_matrices.keys()]
                self._feature_names.extend(new_features)
                self._feature_types.extend(['float' for x in new_features])

            if feature in ['TRANSITION_MOD', 'PMI', 'PMI_INV', 'TRANSITION_MOD_INV']:
                new_features = [(feature,linker) for linker in self._transition_stats.all_reltypes()]
                self._feature_names.extend(new_features)
                self._feature_types.extend(['float' for x in new_features])
            
            if feature in ['MEAN_ORDINAL', 'FREQ']:
                self._feature_names.extend([feature+'1',feature+'2'])
                self._feature_types.extend(['float','float'])
            
            if feature in ['PRED_LEX12','PRED1_OBJ2','PRED2_OBJ1','OBJ1_OBJ2','PRED_OBJ12','SECONDARY']:
                self._feature_names.append(feature)
                self._feature_types.append('string')
            
            #if feature == 'ARG_LDA':
            #    self._arg_lda = train_lda.train_lda(training_file,num_topics_arg_lda,training_file+'.lda_model')
            #    self._feature_names.append('ARG_LDA')
            #    self._feature_types.append('vecFloat_'+str(num_topics_arg_lda))
            
            if feature == 'EVENT_LDA1' or feature == 'EVENT_LDA2' and num_topics_event_lda > 0:
                new_features = [(feature,ind) for ind in range(num_topics_event_lda)]
                self._feature_names.extend(new_features)
                self._feature_types.extend(['float' for x in new_features])

            if feature == 'EVENT_LDA12' and num_topics_event_lda > 0:
                new_features = [(feature,ind1,ind2) for ind1 in range(num_topics_event_lda) \
                                    for ind2 in range(num_topics_event_lda)]
                self._feature_names.extend(new_features)
                self._feature_types.extend(['float' for x in new_features])
            
            if feature.startswith('BROWN_CL50_'):
                self._feature_names.append(feature)
                self._feature_types.append('string')
        
        [self._vec_feature_names.get(fname) for fname,ftype in zip(self._feature_names,self._feature_types) \
             if ftype == 'float']
        
        self._feature_counts = {}
        self._finish_feature_list = False
        self._min_string_feat_freq = int(min_string_feat_freq)


    """
    def all_reltypes(self):
    def get_pair_counts(self,p1,p2,rel_type=LINEAR):
    def get_pmi_for_pairs(self,freq_thresh,rel_type=LINEAR):
    def unigram_prob(self,pred,rel_type=LINEAR):
    """

    
    """
    def get_feature_val(self,v,feat_name,feat_type,ev1,recipe2):
        if feat_name[0][0] == 'TRANSITION':
            #v[('TRANSITION',feat_name[0])] = 
            self._transition_matrices[(feat_name[1],feat_name[2])] \
                [self._seq.inv_get(recipe1.str_form()),self._seq.inv_get(recipe2.str_form())]
        elif feat_name == 'ARG_LDA':
            #args = train_lda.get_args(recipe1)
            #return self._arg_lda[]
    """         

    def vec_feature_names(self):
        return self._vec_feature_names._indices_to_words

    """
    def assign_vector(self,feature_vec,G,e1,e2):
        Receives a feature vector, an event graph and two event identifiers (vertices e1,e2).
        Assigns the matrix in G the correct values
        row_ind = G._edges.index((e1,e2))
        for fname,ftype in zip(self._feature_names,self._feature_types):
            if ftype == 'float':
                G._vecs[row_ind,self._vec_feature_names.get(fname)] = feature_vec[fname]
            elif ftype.startswith('vecFloat_'):
                for ind,val in enumerate(feature_vec[fname]):
                    G._vecs[row_ind,self._vec_feature_names.get(fname+'_'+str(ind))] = feature_vec[fname][ind]
            if ftype == 'string':
                G._vecs[row_ind,self._vec_feature_names.get(fname+'___'+feature_vec[fname])] = 1.0
    """            

    def create_features(self,recipe):
        """
        creates the feature dictionary.
        """
        if self._finish_feature_list:
            return
        G = EventGraph(recipe.keys())
        for e1,e2 in G.edges():
            for fname,ftype in zip(self._feature_names,self._feature_types):
                pred1 = recipe[e1].pred().lower()
                pred2 = recipe[e2].pred().lower()
                str_form1 = recipe[e1].str_form().lower()
                str_form2 = recipe[e2].str_form().lower()
                pred_cluster1 = self._word_to_brown_cluster.get(pred1,UNKNOWN_WORD)
                pred_cluster2 = self._word_to_brown_cluster.get(pred2,UNKNOWN_WORD)
                arg_clusters1 = [self._word_to_brown_cluster.get(x.lower(),UNKNOWN_WORD) \
                                      for x in recipe[e1].args()]
                arg_clusters2 = [self._word_to_brown_cluster.get(x.lower(),UNKNOWN_WORD) \
                                      for x in recipe[e2].args()]
                obj1 = recipe[e1].obj().lower()
                obj2 = recipe[e2].obj().lower()
                if fname == 'PRED1_OBJ2':
                    self._feature_counts[(fname,pred1,obj2)] = self._feature_counts.get((fname,pred1,obj2),0) + 1
                elif fname == 'PRED2_OBJ1':
                    self._feature_counts[(fname,pred2,obj1)] = self._feature_counts.get((fname,pred2,obj1),0) + 1
                elif fname == 'OBJ1_OBJ2':
                    self._feature_counts[(fname,obj1,obj2)] = self._feature_counts.get((fname,obj1,obj2),0) + 1
                elif fname == 'PRED_LEX12':
                    if pred1 in self._verb_list and pred2 in self._verb_list:
                        self._feature_counts[(fname,pred1,pred2)] = \
                            self._feature_counts.get((fname,pred1,pred2),0) + 1
                elif fname == 'PRED_OBJ12':
                    self._feature_counts[(fname,str_form1,str_form2)] = \
                        self._feature_counts.get((fname,str_form1,str_form2),0) + 1
                elif fname == 'SECONDARY':
                    secondary1 = recipe[e1].secondary_pred().lower()
                    secondary2 = recipe[e2].secondary_pred().lower()
                    self._feature_counts[(fname,secondary1,secondary2)] = \
                        self._feature_counts.get((fname,secondary1,secondary2),0) + 1
                elif fname == 'BROWN_CL50_PRED1':
                    self._feature_counts[(fname,pred_cluster1)] = self._feature_counts.get((fname,pred_cluster1),0) + 1
                elif fname == 'BROWN_CL50_PRED2':
                    self._feature_counts[(fname,pred_cluster2)] = self._feature_counts.get((fname,pred_cluster2),0) + 1
                elif fname == 'BROWN_CL50_PRED12':
                    self._feature_counts[(fname,pred_cluster1,pred_cluster2)] = \
                        self._feature_counts.get((fname,pred_cluster1,pred_cluster2),0) + 1
                elif fname == 'BROWN_CL50_ARG1':
                    for arg in arg_clusters1:
                        self._feature_counts[(fname,arg)] = self._feature_counts.get((fname,arg),0) + 1
                elif fname == 'BROWN_CL50_ARG2':
                    for arg in arg_clusters2:
                        self._feature_counts[(fname,arg)] = self._feature_counts.get((fname,arg),0) + 1
                elif fname == 'BROWN_CL50_ARG12':
                    for arg1 in arg_clusters1:
                        for arg2 in arg_clusters2:
                            self._feature_counts[(fname,arg1,arg2)] = self._feature_counts.get((fname,arg1,arg2),0) + 1

    def finish_feature_list(self,L=None):
        """
        This method is called when the feature list has stabilized
        """
        if L == None:
            [self._vec_feature_names.get(feat) for feat,freq in self._feature_counts.items() if freq >= self._min_string_feat_freq]
        else:
            [self._vec_feature_names.get(feat) for feat in L]
        self._vec_feature_names.get(('PRED_LEX12',UNKNOWN_WORD))
        self._vec_feature_names.get(('PRED1_OBJ2',UNKNOWN_WORD))
        self._vec_feature_names.get(('PRED2_OBJ1',UNKNOWN_WORD))
        self._vec_feature_names.get(('OBJ1_OBJ2',UNKNOWN_WORD))
        #self._vec_feature_names.get(('PRED_OBJ12',UNKNOWN_WORD))
        #self._vec_feature_names.get(('SECONDARY',UNKNOWN_WORD))
        
        self._vec_feature_names.get(('BROWN_CL50_PRED1',UNKNOWN_CLUSTER_COMBINATION))
        self._vec_feature_names.get(('BROWN_CL50_PRED2',UNKNOWN_CLUSTER_COMBINATION))
        self._vec_feature_names.get(('BROWN_CL50_PRED12',UNKNOWN_CLUSTER_COMBINATION))
        self._vec_feature_names.get(('BROWN_CL50_ARG1',UNKNOWN_CLUSTER_COMBINATION))
        self._vec_feature_names.get(('BROWN_CL50_ARG2',UNKNOWN_CLUSTER_COMBINATION))
        self._vec_feature_names.get(('BROWN_CL50_ARG12',UNKNOWN_CLUSTER_COMBINATION))
        print('Total number of features:'+str(self._vec_feature_names.get_max()))
        
        #print(self._vec_feature_names._indices_to_words)
        self._finish_feature_list = True
    
    def extract_features(self,recipe):
        """
        Given a recipe, i.e., a collection of events (a dictionary mapping an identifer to a
        TextEvent), returns an EventGraph with the vectors of features.
        """
        if not self._finish_feature_list:
            self.finish_feature_list()
        G = EventGraph(recipe.keys(),self._vec_feature_names.get_max())
        for e1,e2 in G.edges():
            row_ind = G._edges.index((e1,e2))
            if 'EVENT_LDA1' in self._features:
                bag_of_words = [w.lower() for w in recipe[e1].all_words()]
                topic_distribution = self._event_lda[self._event_lda.id2word.doc2bow(bag_of_words)]
                event_lda_vals1 = convert_to_feature_vector(topic_distribution,self._num_topics_event_lda)
            if 'EVENT_LDA2' in self._features:
                bag_of_words = [w.lower() for w in recipe[e2].all_words()]
                topic_distribution = self._event_lda[self._event_lda.id2word.doc2bow(bag_of_words)]
                event_lda_vals2 = \
                    convert_to_feature_vector(topic_distribution,self._num_topics_event_lda)
            pred1 = recipe[e1].pred().lower()
            pred2 = recipe[e2].pred().lower()
            str_form1 = recipe[e1].str_form().lower()
            str_form2 = recipe[e2].str_form().lower()
            str_form_sec_arg1 = recipe[e1].str_form2().lower()
            str_form_sec_arg2 = recipe[e2].str_form2().lower()
            secondary1 = recipe[e1].secondary_pred().lower()
            secondary2 = recipe[e2].secondary_pred().lower()
            obj1 = recipe[e1].obj().lower()
            obj2 = recipe[e2].obj().lower()
            pred_cluster1 = self._word_to_brown_cluster.get(pred1,UNKNOWN_WORD)
            pred_cluster2 = self._word_to_brown_cluster.get(pred2,UNKNOWN_WORD)
            arg_clusters1 = [self._word_to_brown_cluster.get(x.lower(),UNKNOWN_WORD) \
                                 for x in recipe[e1].args()]
            arg_clusters2 = [self._word_to_brown_cluster.get(x.lower(),UNKNOWN_WORD) \
                                 for x in recipe[e2].args()]
            for fname,ftype in zip(self._feature_names,self._feature_types):
                if fname == 'FREE_COEF':
                    G._vecs[row_ind,self._vec_feature_names.get(fname)] = 1.0
                elif fname == 'PRED1_OBJ2':
                    fnum = self._vec_feature_names.get((fname,pred1,obj2),update=False)
                    if fnum is None:
                        fnum = self._vec_feature_names.get((fname,UNKNOWN_WORD),update=False)
                    G._vecs[row_ind,fnum] = 1.0
                elif fname == 'PRED2_OBJ1':
                    fnum = self._vec_feature_names.get((fname,pred2,obj1),update=False)
                    if fnum is None:
                        fnum = self._vec_feature_names.get((fname,UNKNOWN_WORD),update=False)
                    G._vecs[row_ind,fnum] = 1.0
                elif fname == 'OBJ1_OBJ2':
                    fnum = self._vec_feature_names.get((fname,obj1,obj2),update=False)
                    if fnum is None:
                        fnum = self._vec_feature_names.get((fname,UNKNOWN_WORD),update=False)
                    G._vecs[row_ind,fnum] = 1.0
                elif fname == 'PRED_OBJ12':
                    fnum = self._vec_feature_names.get((fname,str_form1,str_form2),update=False)
                    if fnum is None:
                        fnum = self._vec_feature_names.get((fname,UNKNOWN_WORD),update=False)
                    G._vecs[row_ind,fnum] = 1.0
                elif fname == 'SECONDARY':
                    fnum = self._vec_feature_names.get((fname,secondary1,secondary2),update=False)
                    if fnum is None:
                        fnum = self._vec_feature_names.get((fname,UNKNOWN_WORD),update=False)
                    G._vecs[row_ind,fnum] = 1.0
                elif fname == 'PRED_LEX12':
                    fnum = self._vec_feature_names.get((fname,pred1,pred2),update=False)
                    if fnum is None:
                        fnum = self._vec_feature_names.get((fname,UNKNOWN_WORD),update=False)
                    G._vecs[row_ind,fnum] = 1.0
                elif fname == 'BROWN_CL50_PRED1':
                    fnum = self._vec_feature_names.get((fname,pred_cluster1),update=False)
                    if fnum is None:
                        fnum = self._vec_feature_names.get((fname,UNKNOWN_CLUSTER_COMBINATION),update=False)
                    G._vecs[row_ind,fnum] = 1.0
                elif fname == 'BROWN_CL50_PRED2':
                    fnum = self._vec_feature_names.get((fname,pred_cluster2),update=False)
                    if fnum is None:
                        fnum = self._vec_feature_names.get((fname,UNKNOWN_CLUSTER_COMBINATION),update=False)
                    G._vecs[row_ind,fnum] = 1.0
                elif fname == 'BROWN_CL50_PRED12':
                    fnum = self._vec_feature_names.get((fname,pred_cluster1,pred_cluster2),update=False)
                    if fnum is None:
                        fnum = self._vec_feature_names.get((fname,UNKNOWN_CLUSTER_COMBINATION),update=False)
                    G._vecs[row_ind,fnum] = 1.0
                elif fname == 'BROWN_CL50_ARG1':
                    for arg in arg_clusters1:
                        fnum = self._vec_feature_names.get((fname,arg),update=False)
                        if fnum is None:
                            fnum = self._vec_feature_names.get((fname,UNKNOWN_CLUSTER_COMBINATION),update=False)
                        G._vecs[row_ind,fnum] = 1.0
                elif fname == 'BROWN_CL50_ARG2':
                    for arg in arg_clusters2:
                        fnum = self._vec_feature_names.get((fname,arg),update=False)
                        if fnum is None:
                            fnum = self._vec_feature_names.get((fname,UNKNOWN_CLUSTER_COMBINATION),update=False)
                        G._vecs[row_ind,fnum] = 1.0
                elif fname == 'BROWN_CL50_ARG12':
                    for arg1 in arg_clusters1:
                        for arg2 in arg_clusters2:
                            fnum = self._vec_feature_names.get((fname,arg1,arg2),update=False)
                            if fnum is None:
                                fnum = self._vec_feature_names.get((fname,UNKNOWN_CLUSTER_COMBINATION),\
                                                                       update=False)
                            G._vecs[row_ind,fnum] = 1.0
                elif fname == 'MEAN_ORDINAL1':
                    G._vecs[row_ind,self._vec_feature_names.get(fname)] = \
                        self._transition_stats.get_mean_ordinal(str_form1)
                elif fname == 'MEAN_ORDINAL2':
                    G._vecs[row_ind,self._vec_feature_names.get(fname)] = \
                        self._transition_stats.get_mean_ordinal(str_form2)
                elif fname == 'FREQ1':
                    G._vecs[row_ind,self._vec_feature_names.get(fname)] = \
                        np.log(self._transition_stats.unigram_prob(str_form1,no_normalize=True)+0.1)
                elif fname == 'FREQ2':
                    G._vecs[row_ind,self._vec_feature_names.get(fname)] = \
                        np.log(self._transition_stats.unigram_prob(str_form2,no_normalize=True)+0.1)
                elif fname[0] == 'TRANSITION':
                    relevant_matrix = self._transition_matrices[(fname[1],fname[2])]
                    row_seq = self._seqs[fname[1]]
                    ind1 = row_seq.get(recipe[e1].str_form(),update=False)
                    ind2 = row_seq.get(recipe[e2].str_form(),update=False)
                    if ind1 != None and ind2 != None:
                        val = relevant_matrix[ind1,ind2]
                    else:
                        val = 0.0
                    G._vecs[row_ind,self._vec_feature_names.get(fname)] = np.log(val+EPSILON_LOG_PROB)
                elif fname[0] == 'TRANSITION_MOD':
                    if fname[1].endswith('PRED'):
                        val = self._transition_stats.get_pair_counts(pred1,pred2,fname[1])
                    elif fname[1].endswith('SEC'):
                        val = self._transition_stats.get_pair_counts(secondary1,secondary2,fname[1])
                    else:
                        val = self._transition_stats.get_pair_counts(str_form1,str_form2,fname[1])
                    G._vecs[row_ind,self._vec_feature_names.get(fname)] = np.log(val+EPSILON_LOG_PROB)
                elif fname[0] == 'TRANSITION_MOD_INV':
                    if fname[1].endswith('PRED'):
                        val = self._transition_stats.get_pair_counts(pred2,pred1,fname[1])
                    elif fname[1].endswith('SEC'):
                        val = self._transition_stats.get_pair_counts(secondary2,secondary1,fname[1])
                    else:
                        val = self._transition_stats.get_pair_counts(str_form2,str_form1,fname[1])
                    G._vecs[row_ind,self._vec_feature_names.get(fname)] = np.log(val+EPSILON_LOG_PROB)
                elif fname[0] == 'PMI':
                    if fname[1] == analyze_events.LINEAR:
                        G._vecs[row_ind,self._vec_feature_names.get(fname)] = \
                            self._transition_stats.get_pmi(str_form1,str_form2,fname[1],self._freq_thresh_linear)
                    else:
                        G._vecs[row_ind,self._vec_feature_names.get(fname)] = \
                            self._transition_stats.get_pmi(str_form1,str_form2,fname[1],self._freq_thresh_linker)
                elif fname[0] == 'PMI_INV':
                   if fname[1] == analyze_events.LINEAR:
                       G._vecs[row_ind,self._vec_feature_names.get(fname)] = \
                           self._transition_stats.get_pmi(str_form2,str_form1,fname[1],self._freq_thresh_linear)
                   else:
                       G._vecs[row_ind,self._vec_feature_names.get(fname)] = \
                           self._transition_stats.get_pmi(str_form2,str_form1,fname[1],self._freq_thresh_linker) 
                elif fname[0] == 'EVENT_LDA1':
                    G._vecs[row_ind,self._vec_feature_names.get(fname)] = event_lda_vals1[fname[1]]
                elif fname[0] == 'EVENT_LDA2':
                    G._vecs[row_ind,self._vec_feature_names.get(fname)] = event_lda_vals2[fname[1]]
                elif fname[0] == 'EVENT_LDA12':
                    G._vecs[row_ind,self._vec_feature_names.get(fname)] = \
                        event_lda_vals1[fname[1]] * event_lda_vals2[fname[2]]
        G.finish()
        return G
    
    """
    def multiply_features(self,G,fname1,fname2):
    
        gets all the features that start with fname1 and fname2.
        adds new rows to G with all their combinations

        indices1 = self._vec_feature_names.get_all(fname1) # returns pairs of features and numbers starting with fname1
        indices2 = self._vec_feature_names.get_all(fname2)
        for ind1,fname1 in indices1:
            for ind2,fname2 in indices2:
                new_ind = self._vec_feature_names.get((fname1,fname2),update=False)
                G._vecs[:,new_ind] = np.multiply(G._vecs[:,ind1],G._vecs[:,ind2])
    """

def convert_to_feature_vector(topics,num_topics):
    v = [0 for ind in range(num_topics)]
    for num,val in topics:
        v[num] = val
    return v

def read_brown_clusters(filename):
    f = open(filename)
    D = {}
    for line in f:
        fields = line.strip().split('\t')
        if len(fields) > 0:
            D[fields[1]] = fields[0]
    f.close()
    return D

class EventGraph:
    """
    Maps each pair of events to a vector of features.
    """
    
    def __init__(self,keys,num_features=1):
        self._vertices = keys
        self._edges = [(k1,k2) for k1 in keys for k2 in keys if k1 != k2 \
                           and k2 != decoder.START_NODE and k1 != decoder.END_NODE and \
            #TODO: Maybe a problem?
                           (k1 != decoder.START_NODE or k2 != decoder.END_NODE)]
        self._vecs = sparse.dok_matrix((len(self._edges),num_features)) # each row corresponds to an edge. each column to a feature
        
    def assign_vec(self,k1,k2,vec):
        self._vecs[self._edges.index((k1,k2)),:] = vec

    def get_vec(self,edge):
        return self._vecs[self._edges.index(edge),:]
    
    def edges(self):
        return self._edges
        
    def vertices(self):
        return self._vertices

    def num_vertices(self):
        return len(self._vertices)

    def get_edge_weights(self,weights):
        #if self._vecs.shape[1] > weights.size:
        #    self._vecs = self._vecs[:,:weights.size] # re-sizing to get rid of excessive features
        edge_weights = [x[0] for x in (-1.0 * self._vecs * weights.transpose()).tolist()]
        return zip(self._edges,edge_weights)

    def get_edge_weights_extended(self, weights):
        # if self._vecs.shape[1] > weights.size:
        #    self._vecs = self._vecs[:,:weights.size] # re-sizing to get rid of excessive features
        edge_weights_1 = [x[0] for x in (-1.0 * self._vecs * weights[0].transpose()).tolist()]
        edge_weights_2 = [x[0] for x in (-1.0 * self._vecs * weights[1].transpose()).tolist()]
        return zip(self._edges, edge_weights_1, edge_weights_2)

    #weights.get((v1,v2),gurobipy.GRB.INFINITY)
    #def update_weights(self,weight_vec):
    #    for edge,vec in self._G.items():
    #        self._weights[edge] = -1 * vec.dot(weight_vec)
    #        pdb.set_trace()
    
    def get_sum_path(self,perm):
        "Returns the sum of the feature vector over the path"
        row_inds = [self._edges.index(edge) for edge in perm]
        return sum(self._vecs[row_inds,:])

    def finish(self):
        self._vecs = sparse.csr_matrix(self._vecs)

    def scramble_vertices(self,perm):
        V = [v for v in self._vertices if v not in [decoder.START_NODE,decoder.END_NODE]]
        reorder_V = [decoder.START_NODE,decoder.END_NODE] + random.sample(V,len(V))
        V = [decoder.START_NODE,decoder.END_NODE] + V
        vertex_perm = dict(zip(V,reorder_V))
        self._vertices = reorder_V
        self._edges = [(vertex_perm[x],vertex_perm[y]) for x,y in self._edges]
        return [(vertex_perm[x],vertex_perm[y]) for x,y in perm]

    def discard_features(self,feature_indices):
        """
        Leave only features with the corresponding indices.
        """
        self._vecs = self._vecs[:,feature_indices]
        

"""
class FeatureVecConverter:
    FeatureVectors keeps the values of the features for each instance
    and is able to convert them into a matrix if needed.
    
    def __init__(self):
        self._feature_names = []
    
    def add_feature(self,feat_name):
        "introduce a new feature name"
        if feat_name in self._feature_names:
            self._feature_names.append(feat_name)

    def num_features(self):
        return len(self._feature_names)

    def convert_to_vector(self,d):
        d maps feature names to values.
        the method returns a vector after all the proper multiplications, binning etc.
        have been performed.
        pass


"""
