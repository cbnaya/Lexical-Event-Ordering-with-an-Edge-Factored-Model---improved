'''
Created on 26 Sep 2013

@author: oabend
'''
import copy, math, re, pickle, SparseVectorsUtil
from optparse import OptionParser

LHS = 0
RHS = 1

tab_regexp = re.compile('\\s*\\t\\s*')
space_regexp = re.compile('\\s+')





def cmd_line_parser():
    "Returns the command line parser"
    usage = "usage: %prog [options]\n"
    opt_parser = OptionParser(usage=usage)
    opt_parser.add_option("--entailment_file", "-e", dest="entailment_file", action="store", type="string", \
                          help="the list of predicates to extract")
    opt_parser.add_option("--db", "-d", dest="database", action="store", type="string", \
                          help="the database to extract the entries from")
    opt_parser.add_option("--pickle_file", "-p", dest="pickle_file", action="store", type="string", \
                          help="replaces the db file. this is a pickled file with the db")
    opt_parser.add_option("--filename", "-f", dest="filename", action="store", type="string", \
                          help="the filename where the triples are")
    opt_parser.add_option("--output_file", "-o", dest="output", action="store", type="string", \
                          help="the output file where the scores for each entailment are")
    opt_parser.add_option("--output_pickle_file", dest="output_pickle", action="store", type="string", \
                          help="the pickle file where the db file will be stored")
    return opt_parser


def get_predicate_pairs(filename):
    lhs_preds = []
    rhs_preds = []
    f = open(filename)
    for line in f:
        fields = tab_regexp.split(line.strip())
        lhs_preds.append(fields[0])
        rhs_preds.append(fields[1])
    return lhs_preds, rhs_preds

def read_pred_stats_from_file(pred_list, reverb_filename):
    """
    Input: a list of predicates which will serve as the keys of the output dictionary, and an inventory of triples in Reverb style
    Output: a dict of vectors (dicts), each vector contains a dictionary of its slot fillers
    """
    output = dict()
    f = open(reverb_filename)
    for line in f:
        fields = tab_regexp.split(line.strip())
        if fields[5] in pred_list:
            cur_entry = output.get(fields[5],(dict(),dict()))
            cur_entry[LHS][fields[4]] = cur_entry[LHS].get(fields[4],0) + int(fields[7])
            cur_entry[RHS][fields[6]] = cur_entry[RHS].get(fields[6],0) + int(fields[7])
            output[fields[5]] = cur_entry
    return output
    

def main():
    SIM_MEASURE = SparseVectorsUtil.cosine_vectors
    opt_parser = cmd_line_parser()
    (options, args) = opt_parser.parse_args()
    if len(args) > 0:
        opt_parser.error("all arguments must be flagged")
    if options.output == None:
        opt_parser.error("output file must be specified ")

    pred_pairs = get_predicate_pairs(options.entailment_file)
    
    if options.filename != 'None' :
        stats_db = read_pred_stats_from_file(pred_pairs[0] + pred_pairs[1], options.filename)
        if options.output_pickle_file != None:
            f_out = open(options.output_pickle_file, 'wb')
            pickle.dump(stats_db, f_out)
            f_out.close()
    elif options.pickle_file != None:
        stats_db = pickle.load(open(options.pickle_file, 'rb'))
    
    f_out = open(options.output, 'w')
    for lhs_p,rhs_p in zip(pred_pairs[0], pred_pairs[1]):
        try:
            dirt_similarity = math.sqrt(SIM_MEASURE(stats_db[lhs_p][LHS], stats_db[rhs_p][LHS]) * SIM_MEASURE(stats_db[lhs_p][RHS], stats_db[rhs_p][RHS]))
        except KeyError:
            dirt_similarity = 0
        f_out.write(str(dirt_similarity)+'\n')
    f_out.close()


if __name__ == '__main__':
    main()



