"""
Computes the probability of a pair of a pair of verbs given a linker.
"""
import extractDistributions, sys


def main(target_linker_words,output_fn):
    extractor = \
        extractDistributions.NgramsExtractor('/afs/inf.ed.ac.uk/group/corpora/large/google-syntactic-ngrams/extended-triarcs')
    ngramFilter = extractDistributions.QuadarcFilter(target_linker_words)
    extractor.print_to_file_by_filter(ngramFilter, \
                                          '/afs/inf.ed.ac.uk/user/o/oabend/local_disk/linkage_ngrams/'+output_fn, True)
    

if __name__ == '__main__':
    if len(sys.argv) > 4 or len(sys.argv) == 1:
        print('Usage: get_linker_statistics  <output> [<target linker word>]')
        sys.exit(-1)
    elif len(sys.argv) == 3:
        main([sys.argv[2]],sys.argv[1])
    elif len(sys.argv) == 2:
        linkers = ['however', 'but', 'although', 'because', 'after', 'before', 'when', 'if', 'since', 'until', 'while']
        main(linkers,sys.argv[1])




