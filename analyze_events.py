import re, sys, collections, itertools, math, operator
import numpy, sequencer, pdb
import preprocess_data
from scipy import sparse
STOP_LIST = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves","minutes","seconds","hours","moments","minute","hour","moment"]
OBJ_TYPES = ['obj', 'dobj', 'iobj', 'pobj','xcomp','acomp','ccomp']

SUPER_END_RECIPE = 'super_end_recipe'
START_RECIPE = 'start_recipe'
END_RECIPE = 'end_recipe'
REL_DIR_FILE = '/afs/inf.ed.ac.uk/user/o/oabend/event_ordering_resources/rel_directions.txt'

sbrackets = re.compile('[\\[\\]]+')
rbrackets = re.compile('[\\(\\)]+')
linker_regexp = re.compile('[\\(\\) ]+')
bracket_re = re.compile('\\(.*?\\)')

INF = 100 # 1000000
LINEAR = '0<1'
terminate_alpha_pscount = 0.1
begin_alpha_pscount = 0.0
smoothing_alpha = 0


class TextEvent:

    def __init__(self,text):
        self._parse_from_string(text)

    def _parse_from_string(self,s):
        "parses the event from a string"
        if not s.startswith('EVENT'):
            if s == START_RECIPE or s == END_RECIPE:
                self._pred = s
                self._pred_obj = s
                self._features = ''
                self._args = []
                self._pred_obj2 = s
            else:
                print(s)
                return None
        else:
            s = s[6:]
            fields = rbrackets.split(s)
            self._pred = fields[0]
            fields2 = bracket_re.findall(s)
            self._args = [TextEvent._parse_arg(a) for a in fields2[0][1:-1].split('#') if a != '']
            self._features = fields2[1][1:-1]
            self._pred_obj = self._make_pred_obj()
            self._pred_obj2 = self._pred + ' ' + self.obj2()
    
    def pred(self):
        return self._pred
    
    def str_form(self):
        return self._pred_obj

    def str_form2(self):
        return self._pred_obj2

    def all_words(self):
        words = [self._pred] + list(itertools.chain(*[a[0] for a in self._args]))
        words = [w for w in words if w != '' and w[0].isalpha() and w not in STOP_LIST]
        return words
       
    def args(self):
        args = []
        for a in self._args:
            if a[2] in OBJ_TYPES or a[2].startswith('prep'):
                args.append(a[0][a[1]])
        return args
    
    def pred_obj(self):
        return self._pred_obj
    
    def _make_pred_obj(self):
        for a in self._args:
            if a[2] in OBJ_TYPES:
                return self._pred + ' ' + a[0][a[1]]
            elif a[2].startswith('prep'):
                return self._pred + ' ' + a[2] + ' ' +a[0][a[1]]
        return self._pred
    
    def obj(self):
        if self._pred == START_RECIPE or self._pred == END_RECIPE:
            return self._pred
        for a in self._args:
            if a[2] in OBJ_TYPES:
                return a[0][a[1]]
            elif a[2].startswith('prep'):
                return a[2] + ' ' +a[0][a[1]]
        return ''

    def obj2(self):
        seen_args = 0
        if self._pred == START_RECIPE or self._pred == END_RECIPE:
            return self._pred
        for a in self._args:
            if a[2] in OBJ_TYPES:
                if seen_args == 0:
                    seen_args = 1
                else:
                    return a[0][a[1]]
            elif a[2].startswith('prep'):
                if seen_args == 0:
                    seen_args = 1
                else:
                    return a[2] + ' ' +a[0][a[1]]
        return ''
    
    def secondary(self):
        return self._features

    def secondary_pred(self):
        if self._features == '':
            return self._features
        else:
            return self._features + ' ' + self._pred
    
    @staticmethod
    def _parse_arg(s):
        fields = sbrackets.split(s)
        args = fields[0].split()
        for ind,x in enumerate(args):
            if x.startswith('{'): # and x.endswith('}'):
                head_ind = ind
                break
        args[head_ind] = args[head_ind][1:]
        if args[head_ind].endswith('}'):
            args[head_ind] = args[head_ind][:-1]
        arg_type = fields[1]
        return (args,head_ind,arg_type)
    
    def __str__(self):
        return "EVENT "+self._pred+'('+'#'.join([' '.join(args)+' '+str(head_ind)+' '+\
                             str(arg_type) for args,head_ind,arg_type in self._args])+')('+\
                             self._features+')'


class TextLinkage:

    def __init__(self,text):
        fields = text.split('\t')
        self._e1 = TextEvent('EVENT '+fields[1].strip())
        self._e2 = TextEvent('EVENT '+fields[2].strip())
        linker_seq = [y for y in linker_regexp.split(fields[3].lower()) if y != '']
        self._linker_pos = linker_seq[::2]
        self._linker_words = linker_seq[1::2]
    
    def get_event(self,ind):
        if ind == 0:
            return self._e1
        elif ind == 1:
            return self._e2
        else:
            return None

    def get_linker_str_form(self):
        return ' '.join(self._linker_words)

class EventStats:

    def __init__(self):
        self._linkage_pair_counts = {} # maps a linkage type to a counter of pairs
        self._linkage_pred_counts = {} # maps a linkage type to a counter of unigrams
        self._total_linkage = collections.Counter()
        self._ordinals_counter = collections.Counter()
        self._ordinals_total = collections.Counter()
    
    def _read_reltype_directions(self,filename):
        f = open(filename)
        D = {}
        D[LINEAR] = 'true'
        for line in f:
            fields = line.strip().split('\t')
            if fields[1] in ['true', 'false', 'sync', 'ignore']:
                D[fields[0]] = fields[1]
            else:
                raise Exception('Illegal type of linker found')
        f.close()
        return D

    def reltype_direction(self,reltype):
        return self._reltype_direction[reltype]

    def all_reltypes(self):
        return [LINEAR] + [k for k in self._total_linkage.keys() if k != LINEAR]

    def add_pair(self,s1,s2,rel_type=LINEAR):
        cur_pair_counter = self._linkage_pair_counts.get(rel_type,collections.Counter())
        cur_pred_counter = self._linkage_pred_counts.get(rel_type,collections.Counter())
        
        cur_pair_counter[(s1,s2)] += 1
        cur_pred_counter[s1] += 1
        cur_pred_counter[s2] += 1
        self._total_linkage[rel_type] += 1
        self._linkage_pair_counts[rel_type] = cur_pair_counter
        self._linkage_pred_counts[rel_type] = cur_pred_counter

    def add_ordinal(self,s1,ind):
        self._ordinals_counter[s1] += ind
        self._ordinals_total[s1] += 1

    def get_mean_ordinal(self,s1):
        tot = self._ordinals_total.get(s1,0)
        if tot > 0:
            return 1.0 * self._ordinals_counter.get(s1,0) / tot
        else:
            return 0
    
    def get_pair_counts(self,p1,p2,rel_type=LINEAR):
        return self._linkage_pair_counts[rel_type].get((p1,p2),0)

    def get_pmi(self,p1,p2,rel_type=LINEAR,freq_thresh=1):
        freq1 = self._linkage_pred_counts[rel_type].get(p1,0)
        freq2 = self._linkage_pred_counts[rel_type].get(p2,0)
        if freq1 > freq_thresh and freq2 > freq_thresh:
            try:
                pmi = math.log(self._total_linkage[rel_type]) + \
                    math.log(self._linkage_pair_counts[rel_type].get((p1,p2),0)) - \
                    math.log(freq1) - math.log(freq2)
            except ValueError:
                pmi = -INF
        else:
            pmi = 0.0
        return pmi

    def get_pmi_for_pairs(self,freq_thresh,rel_type=LINEAR):
        pmi = {}
        for pair in self._linkage_pair_counts[rel_type]:
            if pair[0] < pair[1]:
                continue
            freq1 = self._linkage_pred_counts[rel_type][pair[0]]
            freq2 = self._linkage_pred_counts[rel_type][pair[1]]
            if freq1 > freq_thresh and freq2 > freq_thresh:
                try:
                    pmi1 = math.log(self._total_linkage[rel_type]) + \
                        math.log(self._linkage_pair_counts[rel_type][pair]) - \
                        math.log(freq1) - math.log(freq2)
                except ValueError:
                    pmi1 = -INF
                try:
                    pmi2 = math.log(self._total_linkage[rel_type]) + \
                        math.log(self._linkage_pair_counts[rel_type][(pair[1],pair[0])]) - \
                        math.log(freq1) - math.log(freq2)
                except Exception:
                    pmi2 = -INF
                pmi[pair] = (pmi1,pmi2,freq1,freq2)
        return pmi
    
    def unigram_prob(self,pred,rel_type=LINEAR,no_normalize=False):
        if no_normalize:
            return self._linkage_pred_counts[rel_type].get(pred,0.0)
        else:
            return self._linkage_pred_counts[rel_type].get(pred,0.0) / self._total_linkage[rel_type]
    
    def output_most_common(self,rel_type=LINEAR):
        for pair,count in self._linkage_pair_counts[rel_type].most_common():
            print(str(pair)+' '+str(count))
        for pair,count in self._linkage_pred_counts[rel_type].most_common():
            print(str(pair)+' '+str(count))

def to_stoch_matrix(pairs,stats):
    row_col_sequencer = sequencer.sequencer()
    start_ind = row_col_sequencer.get(START_RECIPE)
    end_ind = row_col_sequencer.get(END_RECIPE)    
    super_end_ind = row_col_sequencer.get(SUPER_END_RECIPE)
    [(row_col_sequencer.get(p1),row_col_sequencer.get(p2)) for p1,p2,count in pairs]

    num_rows = row_col_sequencer.get_max()
    M = sparse.dok_matrix((num_rows,num_rows))
    for p1,p2,count in pairs:
        ind1 = row_col_sequencer.get(p1,update=False)
        ind2 = row_col_sequencer.get(p2,update=False)
        M[ind1,ind2] = count
    if terminate_alpha_pscount > 0:
        M[:,super_end_ind] = terminate_alpha_pscount # * numpy.ones(M[:,end_ind].shape)
    row_sums = sparse.lil_matrix(M.shape)
    row_sums.setdiag(1.0 / M.sum(axis=1))
    new_matrix = row_sums * M
    return new_matrix, row_col_sequencer

def event_subsets(events):
    temp_events = [TextEvent(START_RECIPE)] + events + [TextEvent(END_RECIPE)]
    return zip(temp_events,temp_events[1:]) 

def event_subsets_skip(events):
    temp_events = [TextEvent(START_RECIPE)] + events + [TextEvent(END_RECIPE)]
    return zip(temp_events,temp_events[2:])

def get_transition_stats(filename,verb_list):
    """
    Returns the transition statistics given the filename
    """
    stats = EventStats()
    f = open(filename)
    events = []
    skipped_lines = 0
    for line in f:
        line = line.strip()
        if line.startswith('EVENT'):
            events.append(TextEvent(line))
        elif line == "=========":
            [stats.add_pair(e[0].str_form().lower(),e[1].str_form().lower()) \
                 for e in event_subsets(events)]
            #[stats.add_pair(e[0].pred().lower(),e[1].pred().lower(),LINEAR+'_PRED') \
            #     for e in event_subsets(events)]
            #[stats.add_pair(e[0].secondary_pred().lower(),e[1].secondary_pred().lower(),LINEAR+'_SEC') \
            #     for e in event_subsets(events)]
            #[stats.add_pair(e[0].str_form().lower(),e[1].str_form().lower(),LINEAR+'2') \
            #     for e in event_subsets_skip(events)]
            #[stats.add_pair(e[0].pred().lower(),e[1].pred().lower(),LINEAR+'_2PRED') \
            #     for e in event_subsets_skip(events)]
            [stats.add_ordinal(e.str_form().lower(),ev_index - 0.5 * len(events)) \
                 for ev_index,e in enumerate(events)]
            events = []
        elif line.startswith('LINKAGE'):
            try:
                cur_linkage = TextLinkage(line)
                stats.add_pair(cur_linkage.get_event(0).str_form(),cur_linkage.get_event(1).str_form(),\
                                   cur_linkage.get_linker_str_form())
                #[stats.add_pair(cur_linkage.get_event(0).pred(),cur_linkage.get_event(1).pred(),\
                #                    cur_linkage.get_linker_str_form()+'_PRED') \
                #     for e in event_subsets(events)]
            except Exception:
                skipped_lines += 1
    sys.stderr.write('Skipped lines:'+str(skipped_lines)+'\n')
    f.close()
    return stats


def get_transition_matrix(filename,verb_list,max_path_len=1,\
                              freq_thresh_linear=10,high_pmi_thresh_linear=3,\
                              low_pmi_thresh_linear=-2,freq_thresh_linker=3,\
                              high_pmi_thresh_linker=3,low_pmi_thresh_linker=-2):
    stats = EventStats()
    f = open(filename)
    events = []
    skipped_lines = 0
    for line in f:
        line = line.strip()
        if line.startswith('EVENT'):
            events.append(TextEvent(line))
        elif line == "=========":
            [stats.add_pair(e[0].str_form().lower(),e[1].str_form().lower()) \
                 for e in event_subsets(events)]
            events = []
        elif line.startswith('LINKAGE'):
            try:
                cur_linkage = TextLinkage(line)
                stats.add_pair(cur_linkage.get_event(0).str_form(),cur_linkage.get_event(1).str_form(),\
                                   cur_linkage.get_linker_str_form())
            except Exception:
                skipped_lines += 1
    sys.stderr.write('Skipped lines:'+str(skipped_lines)+'\n')
    f.close()

    pairs = {}
    for reltype in stats.all_reltypes():
        if reltype == LINEAR:
            signif_pairs = get_significant_pairs(stats,reltype,freq_thresh_linear,high_pmi_thresh_linear,\
                                                     low_pmi_thresh_linear,verb_list)
        else:
            signif_pairs = get_significant_pairs(stats,reltype,freq_thresh_linker,high_pmi_thresh_linker,\
                                                     low_pmi_thresh_linker,verb_list)
        #seq = to_stoch_matrix(signif_pairs,stats,seq,sequencer_dry_run=True) #only get the sequencer
        pairs[reltype] = signif_pairs

    Ms = {}
    seqs = {}
    for reltype in stats.all_reltypes():
        cur_M, seq = to_stoch_matrix(pairs[reltype],stats)
        last_M = sparse.identity(cur_M.shape[0])
        seqs[reltype] = seq
        # exponentiating
        for path_len in range(1,max_path_len+1):
            last_M = last_M * cur_M
            Ms[(reltype,path_len)] = last_M
        #print_to_file(cur_M,seq,'mat_form_'+reltype+'_'+str(thresh),False,thresh,include_start_end=False)
    return Ms, seqs

def get_significant_pairs(stats,reltype,freq_thresh,high_pmi_thresh,low_pmi_thresh,verb_list):
    signif_pairs = []
    for pair,pmis in stats.get_pmi_for_pairs(freq_thresh,reltype).items():
        if verb_list == None or (pair[0].split()[0] in verb_list and pair[1].split()[0] in verb_list):
            if (pmis[0] <= low_pmi_thresh) and (pmis[1] > high_pmi_thresh):
                signif_pairs.append((pair[1],pair[0],stats.get_pair_counts(pair[0],pair[1],reltype)))
            if (pmis[0] > high_pmi_thresh and pmis[1] <= low_pmi_thresh):
                signif_pairs.append((pair[0],pair[1],stats.get_pair_counts(pair[0],pair[1],reltype)))
            #elif stats.reltype_direction(reltype) == 'false':
    return signif_pairs

"""
def read_event_chains(filename):
    event_chains = []
    events = []
    f = open(filename)
    for line in f:
        line = line.strip()
        if line.startswith('EVENT'):
            events.append(TextEvent(line))
        elif line == "=========":
            event_chains.append(events)
            events = []
    f.close()
    return event_chains
"""

def print_to_file(M,seq,filename_out,matrix_form=False,thresh=0.0,include_start_end=False):
    """
    when matrix_form is False, it prints all pairs above the transition
    probability thresh.
    """
    f_out = open(filename_out,'w')
    if matrix_form == False:
        for ind1,ind2 in zip((M > thresh).nonzero()[0],(M > thresh).nonzero()[1]):
            entry1 = seq.inv_get(ind1)
            entry2 = seq.inv_get(ind2)
            if (include_start_end) or (entry1 != START_RECIPE and entry2 != END_RECIPE):
                f_out.write(entry1 + ' | ' + entry2 +' '+str(M[ind1,ind2])+'\n')
    else:
        seq.print_sequencer(f_out)
        for ind1 in range(M.shape[0]):
            for ind2 in range(M.shape[1]):
                f_out.write('%.3g'%M[ind1,ind2]+'\t')
            f_out.write('\n')
    f_out.close()

if __name__ == '__main__':
    verb_list = preprocess_data.get_target_verb_list(preprocess_data.TARGET_VERB_FILE)
    
    if len(sys.argv) < 4:
        print('Usage: analyze_events <training data> <test data> <gamma>')
        sys.exit(-1)
    else:
        gamma = float(sys.argv[3])
        training_file = sys.argv[1]
        test_file = sys.argv[2]
        Ms, seq = get_transition_matrix(training_file,verb_list)


