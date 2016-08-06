'''
Created on 30 Sep 2013

@author: oabend
'''
import utils

FREQ_THRESH = 10
FUNCTION_POS = ['DT', 'EX', 'MD', 'PDT', 'POS', 'PRP', 'PRP$', 'SYM', 'WDT', 'CD', 'CC', '.', ',', '``', ':'] # IN and TO are deliberately excluded
FUNCTION_LABELS = ['auxpass', 'aux', 'punct']
OBJ_LABELS = ['']


"Receives a tuple and returns a tuple of strings using str"
stringify = lambda x: tuple([str(y) for y in x]) 
     
class SelPrefModel:
    '''
    A model of selectional preferences. Given a predicate, returns its preference for different arguments.
    Alternatively, given a predicate and an argument, returns the preference of the argument towards that predicate.
    '''

    def __init__(self, processor_type):
        '''
        Constructor. Creates an empty model.
        '''
        self._pred_to_arg1 = dict()
        self._pred_to_arg2 = dict()
        self._sent_processor = BaselineProcessor(processor_type) 


    def __str__(self):
        return str(self._pred_to_arg1) + '\n' + str(self._pred_to_arg2)

    
    def _update_model(self, pred, arg1, arg2):
        cur_entry1 = self._pred_to_arg1.get(pred, {})
        cur_entry1[arg1] = cur_entry1.get(arg1, 0) + 1
        self._pred_to_arg1[pred] = cur_entry1
        cur_entry2 = self._pred_to_arg1.get(pred, {})
        cur_entry1[arg2] = cur_entry2.get(arg2, 0) + 1
        self._pred_to_arg2[pred] = cur_entry2

    
    def update_from_dep_data(self, filename):
        """
        Initiates the model from dependency data (in the file named filename).
        """
        f = open(filename)
        sent = []
        for line in f:
            line = line.strip()
            if line  != "":
                sent.append(line)
            else:
                pred, arg1, arg2 = self._sent_processor.apply(utils.Sentence(sent))
                sent = []
                self._update_model(pred, arg1, arg2)
    
    def update_from_sp(self, other):
        """
        Updates the model from another sp model.
        """
        return
              
    def get_pred_prefs(self, pred):
        return self._pred_to_arg1.get(pred, {}), self._pred_to_arg2.get(pred, {})

                
    def get_pred_arg_prefs(self, pred, arg, isRight):
        "isRight is True means we take the right argument. we take the left otherwise."
        if isRight:
            d = self._pred_to_arg2.get(pred, {})
        else: 
            d = self._pred_to_arg1.get(pred, {})
        return d.get(arg,0)
            

class BaselineProcessor:
    """
    The C model: the arguments are considered as the direct content word dependents of the verb.
    """    
    def __init__(self, processor_type):
        "Type should be the name of the required processor"
        self._type = processor_type
    
    def apply(self, sent):
        if self._type == "COMP":
            pred = sent.head()
            arg1 = self._extract_arg(sent.word_range(pred.get_field('index')))
            arg2 = self._extract_arg(sent.word_range(None,pred.get_field('index')+1))
        elif self._type == "VOBJ":
            verb = sent.head()
            obj = self._get_obj(sent)
            pred = (str(verb), str(obj))
            arg1 = self._extract_arg('X')
            arg2 = self._extract_arg('Y')
        print(pred, arg1, arg2)
        return stringify(pred), stringify(arg1), stringify(arg2)
        
    def _extract_arg(self, words):
        output = []
        for word in words:
            if word.get_field('freq') >= FREQ_THRESH and word.get_field('POS') not in FUNCTION_POS and word.get_field('label') not in FUNCTION_LABELS:
                output.append(word.get_field('word'))
        return tuple(output)
    
    def _get_obj(self, sent):
        "Returns a representation of the object"
        head = sent.head()
        output = [] 
        for dep in sent.get_deps(head.get_field('index')):
            label = dep.get_field('label')
            if label in ['dobj', 'iobj']:
                output = [dep]
            elif label == 'prep':
                output = [dep, sent.get_deps(dep.get_field('index'))]
        return tuple(output)

