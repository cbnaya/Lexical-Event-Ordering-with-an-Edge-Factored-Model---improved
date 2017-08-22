"""
Add the word 'you' in the beginning of every sentence.
"""
import sys, itertools
TARGET_VERB_FILE = 'resources/verb_list.txt'

def run_preprocessing(in_file,out_file,verb_list=None):
    if verb_list == None:
        verb_list = get_target_verb_list(TARGET_VERB_FILE)
    f_in = open(in_file)
    f_out = open(out_file,'w')
    cur_text = ''
    text = []
    for line in itertools.chain(f_in,['----']):
        line = line.strip()
        if line == '' or ':' in line:
            continue
        elif line == '' or line.startswith('MMMMM') or \
                line.startswith('----'):
            if cur_text != '':
                sentences = cur_text.strip().split('.')
                for sent in sentences:
                    words = sent.strip().split()
                    if len(words) > 0 and words[0].lower() in verb_list:
                        words.insert(0,'you')
                        words[1] = words[1].lower()
                    if len(words) > 0:
                        f_out.write(' '.join(words)+'. ')
                f_out.write('\n')
                cur_text = ''
            if line.startswith('MMMMM') or line.startswith('----'):
                f_out.write('\n END_RECIPE . \n')
        elif line[0].isalpha():
            cur_text = cur_text + ' ' + line
    f_in.close()
    f_out.close()

def get_target_verb_list(filename):
    "returns a list of verbs that we add 'you' before"
    f = open(filename)
    L = [x.strip().lower() for x in f.readlines()]
    f.close()
    return L

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: preprocess_data.py <input file> <output file>')
        sys.exit(-1)
    run_preprocessing(sys.argv[1],sys.argv[2],get_target_verb_list(TARGET_VERB_FILE))
