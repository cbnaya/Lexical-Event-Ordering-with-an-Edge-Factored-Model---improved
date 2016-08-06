'''
Created on 30 Sep 2013

@author: oabend

Adds a frequency column to a set of dependency files (as in the MALT output)
'''
import sys, re, pickle
from optparse import OptionParser
tab_regexp = re.compile('\\s*\\t\\s*')


def get_unigram(filename, pickle_output): 
    f_in = open(filename)
    unigram = dict()
    for line in f_in:
        words = tab_regexp.split(line.strip())
        if len(words) > 1:
            key = words[1].lower()
            unigram[key] = unigram.get(key, 0) + 1
    f_in.close()
    f_out = open(pickle_output, 'wb')
    pickle.dump(unigram, f_out)
    f_out.close()

def add_freq_column(in_filename, out_filename, pickled_unigram):
    """
    Takes a pickled unigram file, and adds the frequency according to a unigram to an extra column.
    """
    f_out = open(out_filename, 'w')
    f_in = open(in_filename)
    unigram = pickle.load(open(pickled_unigram))
    
    for line in f_in:
        words = tab_regexp.split(line.strip())
        if len(words) > 1:
            key = words[1].lower()
            f_out.write(line.strip()+'\t'+str(unigram.get(key,0))+'\n')
        else:
            f_out.write(line)
    f_in.close()
    f_out.close()
  
def cmd_line_parser():
    "Returns the command line parser"
    usage = "usage: %prog [options]\n"
    opt_parser = OptionParser(usage=usage)
    opt_parser.add_option("-w", action="store_true", dest="write", default=False,
                          help="use unigram instead of producing it")
    return opt_parser



if __name__ == '__main__':
    opt_parser = cmd_line_parser()
    (options, args) = opt_parser.parse_args()
    if len(sys.argv) == 3 and options.write:
        get_unigram(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4 and not options.write:
        add_freq_column(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Usage: add_frequency_column <input dependency file> <pickle output dependency file>")
        print("OR")
        print("Usage: add_frequency_column <input dependency file> <output dependency file> <unigram pickle> -w")
        sys.exit(-1)


