'''
Created on 19 Sep 2013

@author: oabend
'''
import os, gzip, re

ARCS_DIR = '/afs/inf.ed.ac.uk/group/corpora/large/google-syntactic-ngrams/arcs'
VERB_POS = re.compile('^VB')
SUBJ_LABEL = 'nsubj'
OBJ_LABEL = 'dobj' 
EMPTY = '---EMPTY---'

AUX_VERBS = ['\'m', '\'s', '\'re', '\'', 'am','is', 'are', 'was', 'were', 'be', 'been', 'being', \
                 'have', 'has', 'had', 'do', 'does', 'did', \
                 'become', 'becomes', 'became', 'becoming', 'turn', 'turned', 'turning', \
                 'turns', 'get', 'got', 'gotten', 'getting', 'gets']

mult_spaces = r = re.compile('\t|  +')

################################
# UTILITY FUNCTIONS
################################
def lcs(a, b):
    "Returns the length of the longest common subsequence of a and b"
    lengths = [[0 for j in range(len(b)+1)] for i in range(len(a)+1)]
    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i+1][j+1] = lengths[i][j] + 1
            else:
                lengths[i+1][j+1] = \
                    max(lengths[i+1][j], lengths[i][j+1])
    return lengths[-1][-1] 

def contains(small, big):
    return (lcs(small,big) == len(small))

#def string_subsequence(L1, L2):
#    "Returns True where L1 is a subsequence of L2 (both contain strings)"
#    regexp = re.compile('(^| )'+' .*'.join(L1)+'($| )')
#    return regexp.search(' '.join(L2)) != None


   

##################################
# CLASSES
##################################

class NgramsExtractor:
    
    def __init__(self, files_directory = ARCS_DIR, filenames = None, gzip=True):
        self._dir = files_directory
        if filenames == None:
            self._file_list = os.listdir(self._dir)
        else:
            self._file_list = filenames
        self._file_handle = None
        self._cur_file_index = -1
        self._gzip = gzip

    def next(self):
        """
        Returns the next Ngram. Returns None when it's done.
        """
        if self._file_handle == None or not self._file_handle:
            self._cur_file_index += 1
            if self._cur_file_index < len(self._file_list):
                filename = self._file_list[self._cur_file_index]
                if self._gzip:
                    self._file_handle = gzip.open(self._dir + '/' + filename,'r')
                else:
                    self._file_handle = open(self._dir + '/' + filename,'r')
            else:
                return None
        try:
            line = self._file_handle.readline()
        except IOError:
            return EMPTY
        try:
            ngram = Ngram(line.strip())
            return ngram
        except ValueError:
            return EMPTY
    
    def get_list_by_filter(self, ngramFilter):
        """
        Returns a list of all the ngrams in the directory where filter returns True.
        lineFilter: an instance of Filter
        """
        output = []
        if not isinstance(ngramFilter, NgramFilter):
            return output

        for filename in self._file_list:
            f = gzip.open(self._dir + '/' + filename,'r')
            while True:
                try:
                    line = f.readline()
                except IOError:
                    continue
                if not line:
                    break
                line = line.decode('UTF-8')
                ngram = Ngram(line.strip())
                if ngramFilter.apply(ngram):
                    output.append(ngram)
        
        return output
    
    def print_to_file_by_filter(self, ngramFilter, output_filename, applyLine = False):
        """
        Writes to a file all the lines matching a specific filter.
        ApplyLine : if True, it applies the filter to the unparsed line with apply_line
        """
        f_out = open(output_filename, 'w')
        
        for filename in self._file_list:
            print('Now processing '+filename)
            if filename.endswith('.gz'):
                f = gzip.open(self._dir + '/' + filename,'r')
                gzipped = True
            else:
                f = open(self._dir + '/' + filename,'r')
                gzipped = False
            while True:
                try:
                    line = f.readline()
                except IOError:
                    continue
                if not line:
                    break
                if gzipped:
                    line = line.decode('UTF-8').strip()
                else:
                    line = line.strip()
                if applyLine:
                    if ngramFilter.apply_line(line):
                        f_out.write(line+'\n')
                else:
                    ngram = Ngram(line.strip())
                    if ngramFilter.apply(ngram):
                        f_out.write(ngram.orig_text()+'\n')
            f.close()
        
        f_out.close()
        

class Ngram:
    """
    A representation of a single ngram in google ngrams.
    """    
    
    def __init__(self, line):
        "Receives a line in the google ngrams file"
        self._orig_line = line
        fields = mult_spaces.split(line)
        if len(fields) <= 2:
            raise ValueError()

        words_temp = fields[1].split(' ')
        words = [Word('ROOT/ROOT/ROOT')]
        head_indices = [-1] 
        for w in words_temp:
            words.append(Word(w))
            head_indices.append(int(w.split('/')[-1]))
            
        self._words = words
        self._head_indices = head_indices
        self._freq = fields[2]
        self._head_index = head_indices.index(0)
    
    def head(self):
        "Returns the headword (of type Word) of the ngram"
        return self._words[self._head_index]
    
    def deps(self, word_index=None):
        """
        Returns a list of the direct dependants of the headword of the ngram.
        If word is not None, it returns the dependants of word.
        """
        dep_indices =  [i for i, x in enumerate(self._head_indices) if x == word_index]
        return [self._words[index] for index in dep_indices]
    
    def verb_verb_linker_tuples(self, linker_word):
        """
        finds all occurrances where a verb has a daughter which is a verb, 
        which has a daughter that is a linker (i.e., has the 
        """
        output = []
        for ind,word in enumerate(self._words):
            if not word.pos().startswith('VB') or word.word() in AUX_VERBS: 
                continue
            for ind_dep, dep in enumerate(self.deps(ind)):
                if not dep.pos().startswith('VB') or dep.word() in AUX_VERBS:
                    continue
                for ind_marker, marker in enumerate(self.deps(ind_dep)):
                    if marker.label() == 'mark' and marker.word() == linker_word:
                        output.append((word.word(),dep.word(),marker.word()))
        return output
    
    def words(self):
        return self._words

    def words_str(self):
        return [w.word() for w in self._words]

    def pos_str(self):
        return [w.pos() for w in self._words]
            
    def freq(self):
        "Returns the frequency of the ngram"
        return self._freq
    
    def orig_text(self):
        return self._orig_line
    

class Word:
    
    def __init__(self, field):
        """
        Receives a file in the format of <word>/<pos>/<edge label>/number.
        It parses the line. 
        """
        temp = field.split('/')
        self._word = ''.join(temp[:-3])
        self._pos = temp[-3]
        self._label = temp[-2]
    
    def word(self):
        return self._word
    
    def pos(self):
        return self._pos
    
    def label(self):
        return self._label

    def __str__(self):
        return '/'.join([self._word, self._pos, self._label])

class NgramFilter:
    """
    Returns True if the Ngram meets the criteria, False otherwise.
    words: a list of list of words, of which one has to appear in the same order for the ngram to be applicable.
    verbs: a list of words of which one has to appear as verb " " " " 
    subjs: " " " subjects " " "
    objs: " " " objects " " " "
    """
    
    def __init__(self, verbs = None, subjs = None, objs = None, words = None):
        if words != None and (verbs != None or subjs != None or objs != None):
            raise Exception("words is only applicable when the rest of the arguments are off.")
        self._verbs = verbs
        self._subjs = subjs
        self._objs = objs
        self._words = words  
        
    def verb_match(self, ngram):        
        if VERB_POS.match(ngram.head().pos()) != None: # it's a verb
            if self._verbs == None or ngram.head().word() in self._verbs:
                return True
        return False
    
    def subj_match(self, ngram):
        "If one of the dependants which has the suject object is the specified kind, it takes it"
        for dep in ngram.deps():
            if dep.label() ==  SUBJ_LABEL and (self._subjs == None or dep.word() in self._subjs):
                return True
        return False
    
    def apply(self, ngram):
        if self._words != None:
            W = [x.word() for x in ngram.words()]
            for L in self._words:
                if string_subsequence(L,W):
                    return True
            return False
        else:
            return self.verb_match(ngram) and self.subj_match(ngram)
    
    def apply_line(self, line):
        return self.apply(Ngram(line))
                

class HeadwordFilter:
    """
    Returns True if the n-gram has a head which is written in one of the forms given to the filter.
    """
    
    def __init__(self, filename = None, L = None):
        if filename != None and L != None:
            raise Exception('Headword filter cannot be inputted both a list and a file.')
        if filename == None and L == None:
            raise Exception('Headword filter needs at least one non-none argument.')
        if L != None:
            self._list = L
        else:
            self._list = []
            f = open(filename)
            while True:
                try:
                    line = f.readline()
                except IOError:
                    continue
                if not line:
                    break
                if line.strip() != '':
                    self._list.append(line.strip())
    
    def apply_line(self, line):
        fields = line.strip().split('\t')
        return fields[0] in self._list



class QuadarcFilter:
    """
    Returns True if the Ngram meets the criteria, False otherwise.
    words: a list of words that have to appear (with the same order).
    pos: a list of POS that have to appear (with the same order).

    """
    
    def __init__(self, linkers):
        self._linkers = linkers

    def apply_line(self, line):
        return self.apply(Ngram(line))
    
    def apply(self,ngram):
        state = 0
        for w in ngram.words():
            if state == 0 and w.pos().startswith('VB') and \
                    w.word() not in AUX_VERBS:
                state = 1
            elif state == 1 and w.word() in self._linkers and \
                    w.label() == 'mark':
                state = 2
            elif state == 2 and w.pos().startswith('VB') and \
                    w.word() not in AUX_VERBS:
                state = 3
        return (state == 3)
            



