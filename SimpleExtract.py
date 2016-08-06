'''
Created on 23 Sep 2013

@author: oabend
'''
import extractDistributions

def main():
    extractor = \
        extractDistributions.NgramsExtractor('/afs/inf.ed.ac.uk/group/corpora/large/google-syntactic-ngrams/extended-triarcs')
    ngramFilter = extractDistributions.QuadarcFilter()
    extractor.print_to_file_by_filter(ngramFilter, '/afs/inf.ed.ac.uk/user/o/oabend/local_disk/xxx', True)
    

if __name__ == '__main__':
    main()




