"""
Execute the event extraction pipeline for a particular file.
"""
import os, preprocess_data, pdb, sys, random, extract_events
from datetime import datetime

def parse_file(in_filename):
    """
    parses a file with the stanford parser, returns both trees and stanford dependencies as a list of 
    pairs of strings, each pair string corresponding to a pair of a tree and stanford dependencies.
    it actually returns a list of such lists, where the lists are delimited by the special token END_RECIPE.
    """
    parser_out = os.popen("java -mx150m -cp \"stanford_parser/*:.\" edu.stanford.nlp.parser.lexparser.LexicalizedParser " + \
                              "-maxLength 40 -outputFormat 'oneline,typedDependencies' edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz "+\
                              in_filename).readlines()
    output = []
    cur_recipe = []
    tree_parse = None
    sd_parse = ''
    state = 0
    for line in parser_out:
        line = line.strip()
        if 'END_RECIPE' in line:
            output.append(cur_recipe)
            cur_recipe = []
        elif line.startswith('Parsing'):
            continue
        elif line.startswith('(ROOT'):
            tree_parse = line
        elif line == '':
            if tree_parse != None and sd_parse != '':
                cur_recipe.append((tree_parse,sd_parse))
                tree_parse = None
                sd_parse = ''
        else:
            sd_parse = sd_parse + line + '\n'
    return [x for x in output if x != []]

def run_discourse_marker_classifier(trees_list):
    """
    trees is a list of lists in which there are tuples of which the first is a tree string.
    the script replaces it with the classifier output string.
    """
    new_trees_list = []
    for entry in trees_list:
        temp_filename = 'temp_trees'+str(random.random())
        f = open(temp_filename,'w')
        for pair in entry:
            f.write(pair[0]+'\n')
        f.close()
        classifier_out = os.popen("perl addDiscourse/addDiscourse.pl --parses "+temp_filename).readlines()
        new_entry = []
        for pair,line in zip(entry,classifier_out):
            new_entry.append((line.strip(),pair[1]))
        new_trees_list.append(new_entry)
        os.remove(temp_filename)
    return new_trees_list
            
        
    
def run_pipeline(in_filename,out_filename):
    preprocess_data.run_preprocessing(in_filename,in_filename+'.preprocessed')
    list_of_tree_sd_string_pairs = parse_file(in_filename+'.preprocessed')
    list_of_tree_sd_string_pairs = run_discourse_marker_classifier(list_of_tree_sd_string_pairs)
    
    # reads the stanford deps, the ptree and write it to f_out
    f_out = open(out_filename,'w')
    f_out_parses = open(out_filename+'.parses','w')
    for recipe in list_of_tree_sd_string_pairs:
        for tree_sd_string_pair in recipe:
            f_out_parses.write(tree_sd_string_pair[0]+'\n'+tree_sd_string_pair[1]+'\n')
            ptree = extract_events.read_tree(tree_sd_string_pair[0])
            standep = extract_events.read_sds_from_string(tree_sd_string_pair[1])
            extract_events.write_events_linkages(standep,ptree,f_out)
        f_out.write('=========\n')
        f_out_parses.write('=========\n')
    f_out.close()
    f_out_parses.close()
    


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: run_pipeline.py <input file>')
        sys.exit(-1)
    run_pipeline(sys.argv[1],sys.argv[1]+'.events')

