'''
Created on 27 Sep 2013
@author: oabend
'''
import re, sys, nltk

tab_regexp = re.compile('\\s*\\t\\s*')
space_regexp = re.compile('\\s+')

def convert_reverb_to_dep(reverb_filename, dep_filename,unigram_filename):
    if unigram_filename:
        unigram = pickle.load(open(unigram_filename,'rb'))
    else:
        unigram = {}
    f = open(reverb_filename)
    f_out = open(dep_filename, 'w')
    for line in f:
        line = line.lower()
        fields = tab_regexp.split(line.strip())
        parts_words = []
        for ind,field in enumerate(fields[1:4]):
            words = space_regexp.split(field)
            parts_words.extend([(ind,w) for w in words])
        tags = nltk.pos_tag([x[1] for x in parts_words])
        freqs = [unigram.get(x[1],0) for x in parts_words]
        for pw,t,freq in zip(parts_words,tags,freqs):
            f_out.write('\t'.join([pw[1],pw[0],t,str(freq)])+'\n')
    f_out.close()
    f.close()

if __name__ == '__main__':
    if len(sys.argv) == 4:
        convert_reverb_to_dep(sys.argv[1], sys.argv[2],sys.argv[3])
    elif len(sys.argv) == 3:
        convert_reverb_to_dep(sys.argv[1], sys.argv[2],None)
    else:
        print('Usage: convert_reverb_to_dep <reverb filename> <output dependency-style filename>' + \
                  '<pickled unigram>')
        sys.exit(-1)

        


