'''
Created on 27 Sep 2013
@author: oabend
'''
import re, sys, nltk

tab_regexp = re.compile('\\s*\\t\\s*')
space_regexp = re.compile('\\s+')

def convert_reverb_to_dep(reverb_filename, unigram_filename=None):
    if unigram_filename:
        unigram = pickle.load(open(unigram_filename,'rb'))
    else:
        unigram = {}
    f = open(reverb_filename)

    for s_ind,line in enumerate(f,1):
        line = line
        fields = tab_regexp.split(line.strip())
        parts_words = []
        for ind,field in enumerate(fields[1:4]):
            words = space_regexp.split(field)
            parts_words.extend([(ind,w) for w in words])
        #tags = nltk.pos_tag([x[1] for x in parts_words])
        #freqs = [unigram.get(x[1],0) for x in parts_words]
        for w_ind,pw in enumerate(parts_words,1):
            print(str(s_ind)+'.'+str(w_ind)+'\t'+str(pw[1])+'\t'+str(pw[0]))
        print('')
    f.close()

if __name__ == '__main__':
    if len(sys.argv) == 2:
        convert_reverb_to_dep(sys.argv[1]) #, sys.argv[2], sys.argv[3])
    else:
        print('Usage: convert_reverb_to_dep <reverb filename>')
        sys.exit(-1)




