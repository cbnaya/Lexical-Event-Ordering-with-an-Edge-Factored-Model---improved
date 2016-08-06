import cPickle as pickle
from gensim.models.ldamodel import LdaModel
from gensim import corpora
from gensim.corpora.textcorpus import TextCorpus
import gensim.corpora
import sys
import analyze_events
import pdb
import itertools

class RecipeCorpus(TextCorpus):

    #def __init__(self,filename):
    #    self._file = filename
    
    def get_texts(self):
        f = open(self.input)
        events = []
        for line in f:
            line = line.strip()
            if line.startswith('EVENT'):
                events.append(analyze_events.TextEvent(line))
            elif line == "=========":
                #args = list(itertools.chain(*[e.args() for e in events]))
                #yield [a.lower() for a in args]
                yield get_all_words(events)
                events = []
        f.close()

class EventCorpus(TextCorpus):
    
    #def __init__(self,filename):
    #    self._file = filename
    
    def get_texts(self):
        f = open(self.input)
        events = []
        for line in f:
            line = line.strip()
            if line.startswith('EVENT'):
                e = analyze_events.TextEvent(line)
                yield [w.lower() for w in e.all_words()]
        f.close()


def get_args(recipe):
    "receives a list of events, returns a list of arguments"
    args = list(itertools.chain(*[e.args() for e in recipe]))
    return [a.lower() for a in args]

def get_all_words(recipe):
    all_words = list(itertools.chain(*[e.all_words() for e in recipe]))
    return [a.lower() for a in all_words]

def convert_to_feature_vector(topics,num_topics):
    v = [0 for ind in range(num_topics)]
    for num,val in topics:
        v[num] = val
    return v

def train_lda(recipe_file,num_topics,output_file):
    corpus = RecipeCorpus(recipe_file)
    
    corpora.MmCorpus.serialize(output_file+'.corpus.mm', corpus)
    lda = LdaModel(corpus, id2word=corpus.dictionary, num_topics=int(num_topics), distributed=False)
    lda.save(output_file)
    return lda

def train_event_lda(recipe_file,num_topics,output_file):
    corpus = EventCorpus(recipe_file)
    
    corpora.MmCorpus.serialize(output_file+'.corpus.mm', corpus)
    lda = LdaModel(corpus, id2word=corpus.dictionary, num_topics=int(num_topics), distributed=False)
    lda.save(output_file)
    return lda

    
    """
    pred_list = utils.get_list_from_file(sys.argv[4])
    preds_to_features = {}
    
    if len(pred_list) != len(corpus):
        raise Exception('Incompatible files')

    for d,pred in zip(corpus,pred_list):
        preds_to_features[pred] = convert_to_feature_vector(lda[d],int(sys.argv[2]))

    pickle.dump(preds_to_features,open('preds_to_features.pickle','wb'))
    """



if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: train_lda.py <recipe file> <num topics> <save model file>')
        sys.exit(-1)
    lda = train_event_lda(sys.argv[1],sys.argv[2],sys.argv[3])
    pdb.set_trace()


