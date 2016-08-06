import decoder, sys, time
import extract_features
import pdb
import os
import itertools
import cPickle as pickle
from optparse import OptionParser
import driver
import scipy.stats
import pdb


def cmd_line_parser():
    "Returns the command line parser"
    usage = "usage: %prog [options]\n"

    opt_parser = OptionParser(usage=usage)
    opt_parser.add_option("-r", action="store", type="string", dest="text_recipes",\
                              help="the file with the features.")
    opt_parser.add_option("--fe", action="store", type="string", dest="feat_extractor",\
                              help="the feature extractor")
    opt_parser.add_option("-m", action="store", type="string", dest="model_pickle",\
                              help="the pickled model for scoring edges")
    opt_parser.add_option("--fs", action="store", type="int", dest="feat_set",\
                              help="the pickled model for scoring edges")
    opt_parser.add_option("-o", action="store", type="string", dest="output",\
                              help="the output file")

    return opt_parser

def run_qual_analysis(options):
    
    # load feature extractor
    f_feat_extract = open(options.feat_extractor)
    feat_extractor = pickle.load(f_feat_extract)
    f_feat_extract.close()        
    
    # load model
    f_model = open(options.model_pickle)
    model = pickle.load(f_model)
    f_model.close()
    
    if options.feat_set != 0:
        vec_feature_names = feat_extractor._vec_feature_names._indices_to_words
        lex_feats = [ind for ind,n in enumerate(vec_feature_names) if n[0] in extract_features.LEX_FEATURES]
        freq_feats = [ind for ind,n in enumerate(vec_feature_names) if n[0] in extract_features.FREQ_FEATURES]
        brown_feats = [ind for ind,n in enumerate(vec_feature_names) if n[0] in extract_features.BROWN_FEATURES]
        
        if options.feat_set == 1:
            leave_inds = freq_feats
        elif options.feat_set == 2:                
            leave_inds = sorted(lex_feats + freq_feats)
        elif options.feat_set == 3:                
            leave_inds = lex_feats
        else:
            sys.stderr.write('Error in defining the feature set. Aborting.')
            sys.exit(-1)  

    weights = {}
    counts = {}
    for recipe,perm in driver.read_recipe(options.text_recipes):
        G = feat_extractor.extract_features(recipe)
        G.discard_features(leave_inds)
        edge_weights = G.get_edge_weights(-1 * model._weights)
        for e,w in edge_weights:
            L = [w,e[0],e[1],(e in perm), recipe[e[0]],recipe[e[1]],recipe[e[0]].str_form(), recipe[e[1]].str_form()]
            print('\t'.join([str(x) for x in L]))
            weights[(recipe[e[0]].str_form(), recipe[e[1]].str_form())] = weights.get((recipe[e[0]].str_form(), recipe[e[1]].str_form()),0) + w
            counts[(recipe[e[0]].str_form(), recipe[e[1]].str_form())] = counts.get((recipe[e[0]].str_form(), recipe[e[1]].str_form()),0) + 1

    weights = [(k,1.0*v/counts[k]) for k,v in weights.items()]
    f = open(options.output,'w')
    pickle.dump(weights,f)
    f.close()

if __name__ == '__main__':
    parser = cmd_line_parser()
    (options,args) = parser.parse_args(sys.argv)
    run_qual_analysis(options)


