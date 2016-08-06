"""
Input: 
A list of extractions in the format
boost(increased security measures[nsubj] ### will[aux] ### operating costs[dobj])	lead(operating costs[nsubj] 
### will[aux] ### higher fares[prep_to])	(IN in) (NN turn)

I.e., 
pred1(###-delimited arguments followed by a syntactic type in []) \t pred2(the same format of arguments) 
\t one or more linkers in the format (POS word).

Output:
A pickle file of predicates. Each predicate is defined by a lemma, 
two syntactic slots, 2 corresponding topics, distribution over pairs of words for these slots and topics.
"""
import extract_events, sys, pdb, itertools
from optparse import OptionParser
import cPickle as pickle
from gensim import corpora, models, similarities

class LDAwrapper:

    def __init__(self,lda):
        self._model = lda
        self._word_given_topic = []
    
    def D_topic_given_slot_type(self,slot_bow,word):
        """
        returns the distribution of topics for the slot (document) syn_key
        and word. slot_bow should be a Bag-Of-Words dictionary of the
        slot fillers.
        """
        output = []
        return output
        
        
def cmd_line_parser():
    "Returns the command line parser"  
    usage = "usage: %prog [options]\n"
    opt_parser = OptionParser(usage=usage)
    opt_parser.add_option("-i", action="store", type="string", dest="input",
                          help="the event extractions")
    opt_parser.add_option("-t", action="store", type="int", dest="num_topics",
                          help="the number of topics")
    opt_parser.add_option("--lda_save", action="store", type="string", dest="lda_save",
                          help="the file with the saved lda model")
    opt_parser.add_option("-c", action="store", type="string", dest="corpus",
                          help="the slot-filler corpus")
    opt_parser.add_option("-d", action="store", type="string", dest="doc_to_pred",
                          help="the file with the pickled document to predicate mapping")
    opt_parser.add_option("--ro", action="store", type="string", dest="repository_output",
                          help="the file to pickle tje repository into")

    return opt_parser


class PredType:
    
    def __init__(self,lemma,syn1,syn2,topic1,topic2,topic_model,dictionary):
        self._lemma = lemma
        self._syn1 = syn1
        self._syn2 = syn2
        self._topic1 = topic1
        self._topic2 = topic2
        self._topic_model = topic_model
        self._dictionary = dictionary
        self._filler_distribution = {}
        
    def update(self,w1,w2,C1,C2):
        """
        updates the scores for this predicate. the instance is (w1,w2). C1 is some factor,
        which probably should be equal to Pr(topic1|lemma,slot1). The same for C2.
        """
        self._filler_distribution[(w1,w2)] = self._filler_distribution.get((w1,w2),0.0) + \
            C1 * self._topic_model.word_given_topic(self._topic1,self._dictionary.word2id(w1)) * \
            C2 * self._topic_model.word_given_topic(self._topic2,self._dictionary.word2id(w2))


class PredRepository:

    def __init__(self,topic_model,dictionary,num_topics,slot_to_topic_D):
        self._repository = {} # mapping a (predicate,slot1,slot2) to a matrix of PredTypes
        self._topic_model = topic_model
        self._dictionary = dictionary
        self._num_topics = num_topics
        self._slot_to_topic_D = slot_to_topic_D

    def update(self,lemma,syn1,syn2,w1,w2):
        if (lemma,syn1,syn2) not in self._repository:
            self._add_pred(lemma,syn1,syn2)
        for topic1 in range(self._num_topics):
            for topic2 in range(self._num_topics):
                self._repository[(lemma,syn1,syn2)][topic1][topic2].update(w1,w2, \
                                 self.topic_prob(lemma,syn1,topic1), \
                                 self.topic_prob(lemma,syn2,topic2))

    def topic_prob(self,lemma,syn_key,topic):
        return self._slot_to_topic_D[(lemma,syn_key)][topic]
    
    def _add_pred(self,lemma,syn1,syn2):
        self._repository[(lemma,syn1,syn2)] = \
            [[PredType(lemma,syn1,syn2,topic1,topic2,self._topic_model,self._dictionary) \
                  for topic1 in range(self._num_topics)] for topic2 in range(self._num_topics)]
    
    def get_pred(self,lemma,slot1,slot2,topic1,topic2):
        predType = self._repository.get((lemma,slot1,slot2),None)
        if predType != None:
            return predType[topic1][topic2]
        else:
            return None

class PredPairRepository:
    
    def __init__(self):
        # each row corresponds to a sparse vector which is the distributional
        # representation of the pair of predicates it stands for.
        self._M = sparse.csr_matrix() 
        # a list of pairs of predicates corresponding to each row
        self._row_legend = [] # labels for the matrix's rows
        self._col_legend = [] # labels for the matrix's columns
        


class SlotFillerCorpus:

    def __init__(self,corpus_filename):
        self._file = open(corpus_filename)
        self._slot_fillers = Counter()
        for line in self._file:
            argstruct = extract_events.ArgStruct.parse(line.strip())
            for arg in argstruct.args1():
                pred = argstruct.pred1()
                for leaf in arg.leaves:
                    self._slot_fillers.update(pred,arg.deptype,preprocess(leaf[1]),1)
            for arg in argstruct.args2():
                pred = argstruct.pred2()
                for leaf in arg.leaves:
                    self._slot_fillers.update(pred,arg.deptype,preprocess(leaf[1]),1)
    
    def write_to_file(self,f_out_fn,pred_to_doc_dict_fn):
        f_out = open(f_out_fn,'w')
        pred_to_doc_dict = []
        for pred_lemma,slot_filler in self._slot_fillers.iteritems():
            pred_to_doc_dict_entry = []
            for slot_type,fillers in slot_filler.iteritems():
                for w,count in fillers.items():
                    f_out.write(' '.join([w for ind in range(count)])+' ')
                f_out.write('\n')
                pred_to_doc_dict_entry.append((pred_lemma,slot_type))
            pred_to_doc_dict.append(pred_to_doc_dict_entry)
        f_out.close()
        pickle.dump(pred_to_doc_dict,open(pred_to_doc_dict_fn,'wb'),pickle.HIGHEST_PROTOCOL)
        return pred_to_doc_dict

class CorpusAsTuples:
    
    def __init__(self,fname):
        self._fname = fname

    def __iter__(self):
        "retrurns an iterator that iterates over tuples of (pred,slot1,filler1,slot2,filler2)"
        f = open(self._fname)
        for line in f:
            argstruct = extract_events.ArgStruct.parse(line.strip())
            # iterating over arguments of pred1
            for argA,argB in itertools.product(argstruct.args1(),argstruct.args1()):
                if argA == argB:
                    continue
                for w1,w2 in itertools.product(argA.leaf_words(),argB.leaf_words()):
                    if w1 == w2:
                        continue
                    yield (argstruct.pred1(),argA.deptype,preprocess(w1),argB.deptype,preprocess(w2))
            # iterating over arguments of pred2
            for argA,argB in itertools.product(argstruct.args2(),argstruct.args2()):
                if argA == argB:
                    continue
                for w1,w2 in itertools.product(argA.leaf_words(),argB.leaf_words()):
                    if w1 == w2:
                        continue
                    yield (argstruct.pred2(),argA.deptype,preprocess(w1),argB.deptype,preprocess(w2))
        f.close()

class Counter:
    
    def __init__(self):
        self._d = {}

    def iteritems(self):
        return self._d.iteritems()

    def update(self,f1,f2,f3,N):
        cur_entry = self._d.get(f1,{})
        cur_entry2 = cur_entry.get(f2,{})
        cur_entry2[f3] = cur_entry2.get(f3,0) + N
        cur_entry[f2] = cur_entry2
        self._d[f1] = cur_entry

class TypingCorpus(corpora.TextCorpus):
    """
    Based on the WikiCorpus of gensim.
    """

    def __init__(self, fname):
        """
        Initialize the corpus. Unless a dictionary is provided, this scans the
        corpus once, to determine its vocabulary.
        """
        self.fname = fname
        self.dictionary = corpora.dictionary.Dictionary(self.get_texts())

    def get_texts(self):
        """
        Iterate over the dump, returning text version of each article as a list
        of tokens.
        Note that this iterates over the **texts**; if you want vectors, just use
        the standard corpus interface instead of this function::
        >>> for vec in wiki_corpus:
        >>>     print vec
        """
        f = open(self.fname)
        for line in f:
            yield preprocess(line.strip()).split()
        
    def doc2bow(self,line):
        "returns a bow representation of the string in line"
        return self.dictionary.doc2bow(line.split())


def preprocess(s):
    "preprocesses a string to turn it into a legal token"
    return s.lower()

def convert_to_feature_vector(topics,num_topics):
    v = [0 for ind in range(num_topics)]
    for num,val in topics:
        v[num] = val
    return v


def main(args):
    parser = cmd_line_parser()
    (options,args) = parser.parse_args(args)
    num_topics = options.num_topics
    
    # load the corpus    
    corpus = SlotFillerCorpus(options.input)
    pred_to_doc_dict = corpus.write_to_file(options.corpus,options.doc_to_pred)
    text_corpus = TypingCorpus(options.corpus)
    
    # train LDA
    lda = models.LdaModel(text_corpus, \
                              num_topics=options.num_topics, \
                              distributed=False)
    if options.lda_save:
        lda.save(options.lda_save)

    # create the topic distribution dictionary
    slot_to_topic_D = {}
    it = iter(text_corpus)
    for pred_slot_list in pred_to_doc_dict:
        for pred_slot in pred_slot_list:
            slot_to_topic_D[pred_slot] = \
                convert_to_feature_vector(lda[next(it)],num_topics)
   
    # create the distributional representations
    pred_slot_filler_tuples = CorpusAsTuples(options.input)
    pred_repository = PredRepository(lda,text_corpus.dictionary,num_topics,slot_to_topic_D)
    for pred,ss1,filler1,ss2,filler2 in pred_slot_filler_tuples:  # (pred,slot1,filler1,slot2,filler2)
        pred_repository.update(pred,ss1,ss2,filler1,filler2)
    
    # dump the repository into a file
    pickle.dump(pred_repository,open(options.repository_output,'wb'))


if __name__ == "__main__":
    main(sys.argv)


