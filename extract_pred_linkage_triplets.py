"""
Extracts triplets of the format (verb,verb,linker).
"""
import extractDistributions, sys, collections


def extract_triplets(ngram_file):
    extractor = \
        extractDistributions.NgramsExtractor(files_directory='.',filenames=[ngram_file],gzip=False)
    ngram = extractor.next()
    joint_distribution = collections.Counter()
    counter = 0
    while ngram != None and counter < 10000000:
        counter += 1
        if ngram != extractDistributions.EMPTY:
            output = ngram.verb_verb_linker_tuples('if')
            for triplet in output:
                joint_distribution[triplet] += int(ngram.freq())
        ngram = extractor.next()

    print(joint_distribution)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: get_linker_statistics <ngram file>')
        sys.exit(-1)
    extract_triplets(sys.argv[1])


