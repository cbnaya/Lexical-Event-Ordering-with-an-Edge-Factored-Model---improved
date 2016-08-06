import decoder, sys, time
import decoder_extended
import extract_features
import pdb
import os
import itertools
import cPickle as pickle
from optparse import OptionParser
#from analyze_events import TextEvent
import analyze_events
import scipy.stats

DEBUG = True
"""
class RecipeReader:

    def __init__(self,filename):
        self._file = open(filename)

    def __iter__(self):
        return self

    def next(self):
        events = {}
        perm = []
        last_index = decoder.START_NODE
        index = decoder.FIRST_REAL_NODE - 1
        for line in self._file:
            line = line.strip()
            print(line)
            if line.startswith('EVENT'):
                ev = TextEvent(line)
                events[index] = ev
                index += 1
                perm.append((last_index,index))
            elif line == "=========":
                perm.append((index,decoder.END_NODE))
                yield events, perm
                events = {}
                perm = []
                last_index = decoder.START_NODE
                index = decoder.FIRST_REAL_NODE - 1
"""

def read_recipe(filename,max_recipes=None):
    events = {}
    perm = []
    last_index = decoder.START_NODE
    index = decoder.FIRST_REAL_NODE
    file_handle = open(filename)
    recipe_index = 0
    for line in file_handle:
        if max_recipes and recipe_index >= max_recipes:
            break            
        line = line.strip()
        if line.startswith('EVENT'):
            ev = analyze_events.TextEvent(line)
            events[index] = ev
            perm.append((last_index,index))
            last_index = index
            index += 1
        elif line == "=========" and events != {}:
            perm.append((index-1,decoder.END_NODE))
            events[decoder.START_NODE] = analyze_events.TextEvent(analyze_events.START_RECIPE)
            events[decoder.END_NODE] = analyze_events.TextEvent(analyze_events.END_RECIPE)
            recipe_index += 1
            yield events, perm
            events = {}
            perm = []
            last_index = decoder.START_NODE
            index = decoder.FIRST_REAL_NODE
    file_handle.close()

def scramble_vertices(Gs,perms):
    """
    scrambles the vertices of the graphs in G, perform a similar scrambling on perms.
    """
    if len(Gs) != len(perms):
        raise Exception('Incompatible lists')
    new_perms = []
    index = 0
    for G,perm in zip(Gs,perms):
        new_perm = G.scramble_vertices(perm)
        new_perms.append(new_perm)        
        index += 1
        if index % 100 == 0:
            print(str(index) + ' instances scrambled')
    return Gs, new_perms
        

def extract_all_features(stats_training_file,weights_training_file,test_file,options,num_topics_arg_lda,\
                             transition_max_path_len,num_iters=3,learning_rate=1,averaged=True):
    if options.load_feat_extract:
        print('Loading feature extractor')
        f_feat_extract = open(options.feat_extract_pickle)
        feat_extractor = pickle.load(f_feat_extract)
        f_feat_extract.close()        
        print('Feature extractor loaded')
    else:
        freq_thresh_linear = options.freq_thresh_linear
        freq_thresh_linker = options.freq_thresh_linker
        high_pmi_thresh_linear = options.freq_thresh_linker
        low_pmi_thresh_linear = options.low_pmi_thresh_linear
        high_pmi_thresh_linker = options.high_pmi_thresh_linker
        low_pmi_thresh_linker = options.low_pmi_thresh_linker
        feat_extractor = extract_features.FeatureExtractor(stats_training_file,\
                                num_topics_arg_lda,options.num_topics_event_lda,transition_max_path_len,\
                                freq_thresh_linear,high_pmi_thresh_linear,low_pmi_thresh_linear,\
                                freq_thresh_linker,high_pmi_thresh_linker,low_pmi_thresh_linker,\
                                                               options.min_string_feat_freq)
        print('Feature extractor initialized')
        sys.stdout.flush()
        
        for recipe,perm in read_recipe(weights_training_file,options.max_train_inst):
            feat_extractor.create_features(recipe)
        print('Done creating feature dictionary')
        sys.stdout.flush()
        
        if options.feat_extract_pickle:
            f_feat_extract_out = open(options.feat_extract_pickle,'w')
            pickle.dump(feat_extractor,f_feat_extract_out)
            f_feat_extract_out.close()
            print('Feature extractor pickled')
            sys.stdout.flush()
    
    training_samples = []
    training_labels = []
    training_labels_tournament = []
    index = 0
    
    t = time.time()
    for recipe,perm in read_recipe(weights_training_file,options.max_train_inst):
        if index < options.skip_train_inst:
            index += 1
            continue
        else:
            G = feat_extractor.extract_features(recipe)
            training_samples.append(G)
            training_labels.append(perm)

            edges = []
            graph = convert_to_graph(perm)
            for v in graph:
                edges += dfs(graph, v)

            training_labels_tournament.append(edges)

            index += 1
            if index % 100 == 0:
                print(str(index)+' training recipes loaded')
                print('Average time to extract 100 samples:'+ str((time.time() - t) / 100))
                t = time.time()
                sys.stdout.flush()

    test_samples = []
    test_labels = []
    test_labels_tournament = []
    index = 0
    t = time.time()
    for recipe,perm in read_recipe(test_file,options.max_test_inst):
        if index < options.skip_test_inst:
            index += 1
            continue
        else:
            G = feat_extractor.extract_features(recipe)
            test_samples.append(G)
            test_labels.append(perm)

            edges = []
            graph = convert_to_graph(perm)
            for v in graph:
                edges += dfs(graph, v)

            test_labels_tournament.append(edges)

            index += 1
            if index % 100 == 0:
                print(str(index)+' test recipes loaded')
                print('Average time to extract 100 samples:'+ str((time.time() - t) / 100))
                t = time.time()
                sys.stdout.flush()

    vec_feature_names = feat_extractor.vec_feature_names()
    #print(vec_feature_names)
    
    return training_samples,training_labels, training_labels_tournament ,test_samples,test_labels, test_labels_tournament ,vec_feature_names

def convert_to_graph(perms):
    graph = {}
    for (x,y) in perms:
        graph[x] = y
    return graph

def dfs(graph, start):
    visited, stack = set(), [start]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            if vertex in graph and graph[vertex] not in visited:
                stack.append(graph[vertex])
    edges = []
    for v in visited:
        if v != start and (start != decoder.START_NODE or v != decoder.END_NODE):
            edges.append((start, v))
    return edges

def cmd_line_parser():
    "Returns the command line parser"
    usage = "usage: %prog [options]\n"

    opt_parser = OptionParser(usage=usage)
    opt_parser.add_option("--trn_stats", action="store", type="string", dest="stats_training_file",
                          help="the training file for gathering the statistics")
    opt_parser.add_option("--trn_weights", action="store", type="string", dest="weights_training_file",
                          help="the training file for getting the feature weights")
    opt_parser.add_option("-t", action="store", type="string", dest="test_file",
                          help="the test file")
    opt_parser.add_option("--arg_lda_topics", action="store", type="int", dest="num_topics_arg_lda",
                          help="the number of topics for training the LDA")
    opt_parser.add_option("--event_lda_topics", action="store", type="int", dest="num_topics_event_lda",default=5,
                          help="the number of topics for training predicate LDA")
    opt_parser.add_option("--max_path_len", action="store", type="int", dest="transition_max_path_len",default=5,
                          help="the maximum length of the path between two events")
    opt_parser.add_option("--num_iters", action="store", type="int", dest="num_iters",default=3,
                          help="the maximum length of the path between two events")
    opt_parser.add_option("--learning_rate", action="store", type="float", dest="learning_rate",default=0.1,
                          help="learning rate for the perceptron")
    opt_parser.add_option("--avg", action="store_true", dest="averaged",
                          help="whether to average the weights or not")
    opt_parser.add_option("--ft", action="store", type="int", dest="freq_thresh_linear",default=0,
                          help="the frequency threshold for the linear order")
    opt_parser.add_option("--ftk", action="store", type="int", dest="freq_thresh_linker",default=0,
                          help="the frequency threshold for the linkers")
    opt_parser.add_option("--hpt", action="store", type="float", dest="high_pmi_thresh_linear",default=3.0,
                          help="the high PMI threshold for the linear order")
    opt_parser.add_option("--lpt", action="store", type="float", dest="low_pmi_thresh_linear",default=-2.0,
                          help="the low PMI threshold for the linear order")
    opt_parser.add_option("--sigma", action="store", type="float", dest="sigma",default=None,
                          help="relevant for MBER: the sigma for the l2 normalization.")
    opt_parser.add_option("--hptk", action="store", type="float", dest="high_pmi_thresh_linker",default=3.0,
                          help="the high PMI threshold for linkers")
    opt_parser.add_option("--lptk", action="store", type="float", dest="low_pmi_thresh_linker",default=-2.0,
                          help="the low PMI threshold for linkers")
    opt_parser.add_option("--load_features", action="store_true", dest="load_features",
                          help="loads the features instead of extracting them")
    opt_parser.add_option("--pfv", action="store", type="string", dest="pickle_vector_graphs",
                          help="pickle the vector graphs")
    
    opt_parser.add_option("--pfv_test", action="store", type="string", dest="pickle_vectors_test",default=None,
                          help="load the feature vectors from the first and the second files")
    
    opt_parser.add_option("--model_pickle", action="store", type="string", dest="model_pickle",
                          help="pickle the model itself")
    opt_parser.add_option("--load_model", action="store", dest="load_model",default=None,
                          help="if toggled, loads a trained model instead of training it")
    opt_parser.add_option("--min_string_feat_freq", action="store",type="string",dest="min_string_feat_freq",default=1)
    opt_parser.add_option("--feat_extract_pickle", action="store",type="string",dest="feat_extract_pickle",default=None)
    opt_parser.add_option("--load_feat_extract", action="store_true",dest="load_feat_extract",default=False,
                          help="if toggled, it loads the feature extractor instead of creating it from data;"+ \
                              " no features can be added from this point on")
    opt_parser.add_option("--skip_training", action="store_true",dest="skip_training",default=False,
                          help="whether to skip the training completely")

    opt_parser.add_option("--skip_train_inst", action="store",type="int",dest="skip_train_inst",default=0,
                          help="the number of train instances to skip")
    opt_parser.add_option("--skip_test_inst", action="store",type="int",dest="skip_test_inst",default=0,
                          help="the number of test instances to skip")
    opt_parser.add_option("--max_train_inst", action="store",type="int",dest="max_train_inst",default=10000000,
                          help="the maximal index of train instance to take")
    opt_parser.add_option("--max_test_inst", action="store",type="int",dest="max_test_inst",default=10000000,
                          help="the maximal index of test instance to take")

    opt_parser.add_option("-b", action="store",type="int",dest="baseline",default=0,
                          help="by default is 0 (the regular model); 1: greedy baseline; 2: local training, " + \
                              "greedy test; 4: MBER")
    opt_parser.add_option("-r", action="store",dest="results_file",default=None,
                          help="the file containing the results")
    opt_parser.add_option("--calc_train_acc", action="store_true",dest="calc_train_acc",default=False,
                          help="whether to compute the training acc")
    opt_parser.add_option("-d", action="store_true",dest="debug_mode",default=False,
                          help="debug mode")
    opt_parser.add_option("--tl", action="store",type="float",default=5.0,dest="time_limit",
                          help="the time limit (in seconds) for running the ILP")
    opt_parser.add_option("--si", action="store",type="int",default=None,dest="seen_instances_loaded_model",
                          help="the number of seen instances by the loaded model if applicable")
    opt_parser.add_option("--binary_class", action="store_true",default=False,dest="binary_class",
                          help="the ability to distinguish between a randomly permuted and a correctly ordered recipe")
    opt_parser.add_option("--tl_test", action="store",type="float",default=None,dest="time_limit_test",
                          help="the time limit (in seconds) for running the ILP in inference (otherwise same as in training)")
    opt_parser.add_option("--extract_and_die",default=False,action="store_true",dest="extract_and_die")
    opt_parser.add_option("--fs", action="store",dest="feat_set",type="int",default=0,
                          help="the feature set to be used: 0 for the full set; 1 for only freq features;" + \
                              " 2 for freq+lexical; 3 for only lexical")
    opt_parser.add_option("--extended", action="store_true", dest="extended_precepteron", default=False,
                          help="use extended precepteron")

    return opt_parser


def run_recipe_ordering(stats_training_file,weights_training_file,test_file,options,num_topics_arg_lda=5,\
                            transition_max_path_len=5,\
                            num_iters=3,learning_rate=1,averaged=True):
    if options.load_features: # and os.path.isfile(options.pickle_vector_graphs):
        f_pickle = open(options.pickle_vector_graphs)
        training_samples,training_labels, training_labels_tournament, test_samples,test_labels, \
            test_labels_tournament, vec_feature_names = pickle.load(f_pickle)
        f_pickle.close()

        if options.pickle_vectors_test:
            f_pickle = open(options.pickle_vectors_test)
            temp,temp2,test_samples,test_labels,vec_feature_names2 = pickle.load(f_pickle)
            f_pickle.close()
            
        start_ind = options.skip_train_inst
        end_ind = options.max_train_inst
        training_samples = training_samples[start_ind:(end_ind+1)]
        training_labels = training_labels[start_ind:(end_ind)+1]
        training_labels_tournament  = training_labels_tournament[start_ind:(end_ind)+1]
        
        start_ind_test = options.skip_test_inst
        end_ind_test = options.max_test_inst
        test_samples = test_samples[start_ind_test:(end_ind_test+1)]
        test_labels = test_labels[start_ind_test:(end_ind_test)+1]
        test_labels_tournament = test_labels_tournament[start_ind_test:(end_ind_test)+1]

        #if options.debug_mode:
        #    lex_feats = [ind for ind,n in enumerate(vec_feature_names) if n[0] == 'PRED_LEX12']
        #    vec_feature_names = [n for ind,n in enumerate(vec_feature_names) if n[0] == 'PRED_LEX12']
        #    for sample in training_samples + test_samples:
        #        sample.discard_features(lex_feats)
        if options.feat_set != 0:
            lex_feats = [ind for ind,n in enumerate(vec_feature_names) if n[0] in extract_features.LEX_FEATURES]
            freq_feats = [ind for ind,n in enumerate(vec_feature_names) if n[0] in extract_features.FREQ_FEATURES]
            brown_feats = [ind for ind,n in enumerate(vec_feature_names) if n[0] in extract_features.BROWN_FEATURES]
            sys.stdout.flush()
            if options.feat_set == 1:
                leave_inds = freq_feats
            elif options.feat_set == 2:                
                leave_inds = sorted(lex_feats + freq_feats)
            elif options.feat_set == 3:                
                leave_inds = lex_feats
            else:
                sys.stderr.write('Error in defining the feature set. Aborting.')
                sys.exit(-1)  
            vec_feature_names = [vec_feature_names[ind] for ind in leave_inds]
            for ind,sample in enumerate(training_samples + test_samples):
                sample.discard_features(leave_inds)

        print('Total number of features:'+str(len(vec_feature_names)))
        #print(vec_feature_names)
    else:
        training_samples,training_labels, training_labels_tournament, test_samples,test_labels, test_labels_tournament,\
        vec_feature_names = \
            extract_all_features(stats_training_file,weights_training_file,test_file,options,num_topics_arg_lda,\
                                     transition_max_path_len,\
                                     num_iters,learning_rate,averaged)
        if options.pickle_vector_graphs:
            f_pickle = open(options.pickle_vector_graphs,'w')
            pickle.dump((training_samples,training_labels,test_samples,test_labels,vec_feature_names),\
                            f_pickle,pickle.HIGHEST_PROTOCOL)
            f_pickle.close()
            print('Training & test features pickled')

    print('Number of training instances:'+str(len(training_samples)))
    print('Number of test instances:'+str(len(test_samples)))
    sys.stdout.flush()
    
    if options.extract_and_die:
        sys.exit(-1)

    # HACK. convert all == -1000000.0 to -100.0
    for g in training_samples + test_samples:
        g._vecs[g._vecs == -1000000] = -100    

    """
    if options.scramble:
        training_samples, training_labels = scramble_vertices(training_samples, training_labels)
        print(training_labels)
        test_samples,test_labels = scramble_vertices(test_samples,test_labels)
        """
    errs = []

    # training
    if options.load_model:
        f_pickle = open(options.load_model)
        perceptron = pickle.load(f_pickle)
        perceptron._model_pickle = perceptron._model_pickle + '_new'
        f_pickle.close()
        perceptron.set_time_limit(options.time_limit)
        if options.seen_instances_loaded_model:
            perceptron._interm_averaged_model_num_instances = options.seen_instances_loaded_model
    else:
        if options.extended_precepteron:
            perceptron = decoder_extended.StructuredPerceptronExtended(len(vec_feature_names), \
                                                      num_iters, learning_rate, averaged, vec_feature_names, \
                                                      calc_train_accuracy=options.calc_train_acc, \
                                                      greedy_inference=False, model_pickle=options.model_pickle, \
                                                      time_limit=options.time_limit)
        elif options.baseline in [0,3]:
            perceptron = decoder.StructuredPerceptron(len(vec_feature_names),\
                                                          num_iters,learning_rate,averaged,vec_feature_names,\
                                                          calc_train_accuracy=options.calc_train_acc,\
                                                          greedy_inference=False,model_pickle=options.model_pickle,\
                                                          time_limit=options.time_limit)
        elif options.baseline in [1,2]:
            perceptron = decoder.StructuredPerceptron(len(vec_feature_names),\
                                                          num_iters,learning_rate,averaged,vec_feature_names,\
                                                          calc_train_accuracy=options.calc_train_acc,\
                                                          greedy_inference=True,model_pickle=options.model_pickle)
        elif options.baseline in [4]:
            perceptron = decoder.StructuredPerceptron(len(vec_feature_names),\
                                                          0,0,False,vec_feature_names,\
                                                          greedy_inference=False,\
                                                          model_pickle=options.model_pickle,\
                                                          probs_instead_of_weights=True)
        elif options.baseline in [5]:
            perceptron = decoder.StructuredPerceptron(len(vec_feature_names),\
                                                          0,0,False,vec_feature_names,\
                                                          greedy_inference=True,\
                                                          model_pickle=options.model_pickle,\
                                                          probs_instead_of_weights=False)
    print('Beginning training')
    
    if not options.skip_training:
        '''
        training_samples = training_samples[:4000]
        training_labels = training_labels[:4000]
        training_labels_tournament = training_labels_tournament[:4000]
        '''

        if options.extended_precepteron:
            perceptron.fit(training_samples, training_labels, training_labels_tournament)
        elif options.baseline in [0,1]:
            #pass
            perceptron.fit(training_samples,training_labels)
        elif options.baseline in [2,3]:
            perceptron.local_fit(training_samples,training_labels)
        elif options.baseline in [4,5]:
            perceptron.fit_mbr(training_samples,training_labels)
    
    
    # options.time_limit_test
    # predict on the test data and compute the accuracy
    if options.results_file:
        output_file = open(options.results_file,'w')
    else:
        output_file = None
    
    if options.time_limit_test:
        perceptron.set_time_limit(options.time_limit_test)

    if options.binary_class:
        # see if a random path does better than the correct one.
        acc = perceptron.binary_classification(test_samples,test_labels)
        print('Binary classification accuracy:'+str(acc))
    else:
        exact_match, macro_avg_pair_swaps, macro_avg_triple_swaps, macro_avg_quad_swaps, errs = \
            perceptron.test_on_data(test_samples,test_labels,output_file=output_file)
    
        print('Exact matches:'+str(exact_match))
        print('Macro-average pair swaps:'+str(macro_avg_pair_swaps))
        print('Macro-average triple swaps:'+str(macro_avg_triple_swaps))
        print('Macro-average quad swaps:'+str(macro_avg_quad_swaps))
        print('Errors in test set decoding:'+str(errs))
        print('\t'.join(["%0.3f" % i for i in \
                             [macro_avg_pair_swaps,macro_avg_triple_swaps,macro_avg_quad_swaps,exact_match]]))
    
if __name__ == '__main__':
    parser = cmd_line_parser()
    (options,args) = parser.parse_args(sys.argv)
    print(options.weights_training_file)
    run_recipe_ordering(options.stats_training_file,options.weights_training_file,\
                        options.test_file,options,options.num_topics_arg_lda,options.transition_max_path_len,\
                        options.num_iters,options.learning_rate,options.averaged)


