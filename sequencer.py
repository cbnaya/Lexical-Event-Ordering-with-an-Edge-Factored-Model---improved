#!/usr/local/bin/python

import sys, re

# A sequencer for strings
class sequencer():
    
    def __init__(self):
        self._cur_max = 0
        self._words = dict()
        self._indices_to_words = []
    
    def get(self, word, update=True):
        output = self._words.get(word, None)
        if output != None or not update:
            return output
        else:
            self._words[word] = self._cur_max 
            self._indices_to_words.append(word)
            self._cur_max += 1
            return self._words[word]

    def get_all(self,prefix):
        output = []
        for ind,word in enumerate(self._indices_to_words):
            if word[0] == prefix:
                output.append((ind,word))
        return output              
    
    def get_max(self):
        return self._cur_max
    
    def print_sequencer(self,f_out):
        inv_dict = dict([(x[1], x[0]) for x in self._words.items()])
        for key in inv_dict.keys():
            f_out.write(str(key)+'\t'+ str(inv_dict[key])+'\n')

    def get_reversed_dictionary(self):
        return dict([(x[1], x[0]) for x in self._words.items()])
    
    def inv_get(self, index):
        if index < len(self._indices_to_words):
            return self._indices_to_words[index]
        else:
            return None
        #for key in self._words.keys():
        #    if self._words[key] == index:
        #        return key

# END OF CLASS
