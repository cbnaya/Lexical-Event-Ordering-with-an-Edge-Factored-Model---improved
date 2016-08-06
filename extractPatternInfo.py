'''
Created on 25 Sep 2013

@author: oabend
'''
import extractDistributions, sys

def file_to_list(filename):
    "Receives a file name and returns a list where each entry is a list of the words in a given line"
    f = open(filename)
    L = []
    for line in f:
        L.append(line.strip().split(' '))
    return L


def main(args):
    extractor = extractDistributions.NgramsExtractor('/disk/data2/oabend/take_trigrams')
    ngramFilter = extractDistributions.NgramFilter(words = file_to_list(args[0]))
    extractor.print_to_file_by_filter(ngramFilter, args[1])
    

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: extractPatternInfo <list of complex predicates> <output file>')
        sys.exit(-1)
    main(sys.argv[1:])

