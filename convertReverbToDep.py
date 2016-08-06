'''
Created on 27 Sep 2013

@author: oabend
'''
import re, sys

DEFAULT_POS = 'POS'
tab_regexp = re.compile('\\s*\\t\\s*')
space_regexp = re.compile('\\s+')

def convert_reverb_to_dep(reverb_filename, dep_filename):
    f = open(reverb_filename)
    f_out = open(dep_filename, 'w')
    for line in f:
        fields = tab_regexp.split(line.strip())
        index = 1
        for x in fields[1:4]:
            words = space_regexp.split(x)
            for w in words:
                f_out.write(str(index) + '\t' + w + '\t_\t' + DEFAULT_POS + '\t' + DEFAULT_POS + '\t' + '_' + '\n')
                index += 1
        f_out.write('\n')
    f_out.close()
    f.close()


def convert_reverb_to_plain_text(reverb_filename, dep_filename):
    f = open(reverb_filename)
    f_out = open(dep_filename, 'w')
    for line in f:
        fields = tab_regexp.split(line.strip())
        for x in fields[1:4]:
            words = space_regexp.split(x)
            for w in words:
                f_out.write(w + ' ')
        f_out.write('\n')
    f_out.close()
    f.close()

def convert_orens_to_plain_text(reverb_filename, dep_filename):
    f = open(reverb_filename)
    f_out = open(dep_filename, 'w')
    for line in f:
        fields = tab_regexp.split(line.strip())
        for x in fields[2:4]:
            words = space_regexp.split(x)
            for w in words:
                f_out.write(w + ' ')
            f_out.write('.\n')
    f_out.close()
    f.close()
    
def convert_plain_text_to_dep(plain_text_file, dep_filename):
    f = open(plain_text_file)
    f_out = open(dep_filename, 'w')
    for line in f:
        words = space_regexp.split(line.strip())
        index = 1
        for w in words:
            word_pos = w.split('_')
            f_out.write(str(index) + '\t' + '_'.join(word_pos[:-1]) + '\t_\t' + word_pos[-1] + '\t' + word_pos[-1] + '\t' + '_' + '\n')
            index += 1
        f_out.write('\n')    
    f_out.close()

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: convert_reverb_to_dep <reverb filename> <output dependency filename> <option number>')
        sys.exit(-1)
    if sys.argv[3] == '0':
        convert_reverb_to_dep(sys.argv[1], sys.argv[2])
    elif sys.argv[3] == '1':
        convert_reverb_to_plain_text(sys.argv[1], sys.argv[2])
    elif sys.argv[3] == '2':
        convert_plain_text_to_dep(sys.argv[1], sys.argv[2])
    elif sys.argv[3] == '3':    
        convert_orens_to_plain_text(sys.argv[1], sys.argv[2])

