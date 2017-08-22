from nltk.tree import *
import sys, pdb, re
import preprocess_data
import pattern.en

delim = re.compile('[,()]')
non_alpha = re.compile('[^A-Za-z]')
ARG_TYPES = ['agent', 'comp', 'acomp', 'ccomp', 'xcomp', 'obj', 'dobj', 'iobj', 'pobj', \
                 'subj', 'nsubj', 'nsubjpass', 'csubj', 'csubjpass']
AUX_VERBS = ['\'m', '\'s', '\'re', '\'', 'am','is', 'are', 'was', 'were', 'be', 'been', \
                 'being', 'have', 'has', 'had', 'do', 'does', 'did', \
                 'become', 'becomes', 'became', 'becoming', 'turn', 'turned', 'turning', \
                 'turns', 'get', 'got', 'gotten', 'getting', 'gets']
IGNORE_SD_TYPES = ['conj']
SECONDARY_VERBS = []

###################
# Classes
###################

class Event:
    "Defines the argument structure of an event consisting of an argument and a linker"

    def __init__(self,pred,pred_ind,stanford_deps,tree,words_to_avoid):
        self._pred = pred
        words_to_avoid.append(self.pred())
        try:
            self._args, self._features = extract_args(pred, pred_ind, stanford_deps, tree, words_to_avoid)
        except Exception:
            self._args = []
            self._features = []

    def __str__(self):
        event_str = str(self._pred[0]) + '(' + '#'.join([str(a) for a in self._args]) + ')' + '(' + self.features_str() + ')'
        return event_str

    def features_str(self):
        "returns a string of the features"
        temp = [f[0]+'_'+f[1] for f in self._features]
        return ','.join(temp)
    
    def pred(self):
        return str(self._pred[0])

    def args(self):
        return self._args
    
    def secondary_verb_of(self):
        if self.pred() in SECONDARY_VERBS:
            xcomp_args = [arg for arg in self.args() if arg.deptype == 'xcomp' and arg.head_leaf().label().startswith('VB')]
            if xcomp_args != []:
                print(xcomp_args[0].head_node)
                return xcomp_args[0].head_node
        print(None)
        return None
    
            

class Linkage:
    "Defines a linkage between two predicates, their arguments and a linker"
    
    def __init__(self, lpred, lpred_ind, rpred, rpred_ind, connective, stanford_deps, tree):
        self._connective = [x[1] for x in connective]
        connective_words = [str(x[0]) for x in self._connective]
        self._events = \
            [Event(lpred,lpred_ind,stanford_deps,tree,connective_words), \
             Event(rpred,rpred_ind,stanford_deps,tree,connective_words)]
        self._swap_secondaries(stanford_deps,tree,connective_words)

    def _swap_secondaries(self,stanford_deps,tree,connective_words):
        new_events = []
        for ev in self._events:
            new_pred = ev.secondary_verb_of()
            if new_pred != None:
                new_pred_ind = leaf_index(new_pred)
                new_events.append(Event(new_pred,new_pred_ind,stanford_deps,tree,connective_words))
            else:
                new_events.append(ev)
        self._events = new_events

    def args1(self):
        return self._events[0].args()

    def args2(self):
        return self._events[1].args()
    
    def pred1(self):
        return self._events[0].pred()

    def pred2(self):
        return self._events[1].pred()

    def __str__(self):
        con = ' '.join([str(x) for x in self._connective])
        return 'LINKAGE\t' + '\t'.join([str(ev) for ev in self._events]) + '\t' + con 
        


class StanfordDep:
    
    def __init__(self,s):
        "returns a stanford dependency from an str"
        fields = delim.split(s)
        try:
            self.type = fields[0].split('#')[0].strip()

            self.head = '-'.join(fields[1].split('-')[:-1]).strip().split('#')[0]
            self.dep = '-'.join(fields[2].split('-')[:-1]).strip().split('#')[0]
            
            self.head_ind = int(fields[1].split('-')[-1])
            self.dep_ind = int(fields[2].split('-')[-1])
        except Exception:
            raise ValueError()

class StanfordGraph:
    
    def __init__(self,L):
        "L is a list of stanford dependencies"
        self._graph = L

    def deps(self,head_ind,exclude_types=[]):
        "returns all the dependencies in which head_ind is the index of the head"
        output = []
        for sd in self._graph:
            if sd.head_ind == head_ind and sd.type not in exclude_types:
                output.append(sd)
        return output

    def dep_inds(self,head_ind):
        return [d.dep_ind for d in self.deps(head_ind)]

    def get_word(self,ind):
        "returns the word with the given ind"
        for sd in self._graph:
            if sd.head_ind == ind:
                return sd.head
            elif sd.dep_ind == ind:
                return sd.dep
        return None

    def sub_tree(self,head_ind):
        """
        returns all the word and indices of descendents of the word indexed head_ind.
        """
        output = [(head_ind,self.get_word(head_ind))]
        Q = [head_ind]
        encountered = set([x[0] for x in output])
        counter = 0
        while Q != []:
            cur_ind = Q.pop(0)
            deps = self.deps(cur_ind)
            output.extend([(d.dep_ind,d.dep) for d in deps])
            encountered.update([d.dep_ind for d in deps])
            Q.extend([d.dep_ind for d in deps if d.dep_ind not in encountered])
            counter += 1
            if counter > 100:
                print('Warning: Counter is '+str(counter))
        return sorted(output,key=lambda x: x[0])

    def __iter__(self):
        return iter(self._graph)
            
class Arg:

    def __init__(self,deptype,head_node_ind,head_node):
        self.deptype = deptype
        self.head_node_ind = head_node_ind
        self.head_node = head_node

    def leaves(self):
        "returns a list of the leaves of the argument, i.e., all the descendants of head_node"
        return self.head_node.leaves()

    def leaves_pos(self):
        return [x for x in get_subtrees(self.head_node) if len(x) > 0 and isinstance(x[0],str)]

    def head_leaf(self):
        return self.leaves_pos()[self.head_node_ind]

    def __str__(self):
        leaves = [x.split('#')[0].strip() for x in self.leaves()]
        return ' '.join([ (x if leaf_ind != self.head_node_ind else '{'+x+'}') for leaf_ind,x in enumerate(leaves)]) + \
            ' ['+self.deptype+']'

###################
# Static Methods
###################

def read_sent_sds(f_handle):
    "reads a sentence from a file in SD format"
    output = []
    for line in f_handle:
        if line.strip() == '' and output != []:
            break
        try: 
            sd = StanfordDep(line.strip())
            if sd.type in IGNORE_SD_TYPES:
                continue
            output.append(sd)
        except ValueError:
            pass
    return StanfordGraph(output)

def read_sds_from_string(s):
    "reads a sentence from a string. each dependency is a separate line."
    output = []
    sents = s.split('\n')
    for line in sents:
        if line.strip() == '' and output != []:
            break
        try: 
            sd = StanfordDep(line.strip())
            if sd.type in IGNORE_SD_TYPES:
                continue
            output.append(sd)
        except ValueError:
            pass
    return StanfordGraph(output)


def leaf_index(leaf):
    """
    receives a leaf of the type Parented_Tree, returns its index in the tree.
    the indexing starts from 1.
    """
    T = leaf.root()
    ind = 0
    for sub_tree in T.subtrees():
        if not isinstance(sub_tree[0],Tree):
            ind += 1
        if sub_tree == leaf:
            return ind
    return None


def get_leaf(T,ind):
    """
    receives a Parented_Tree and an index (int). 
    returns the leaf whose index is ind (starting from 1)
    """
    try:
        tp = T.leaf_treeposition(ind-1)[:-1]
    except Exception:
        return None
    return T[tp]    

def all_sibs(node):
    sibs = []
    par = node.parent()
    if par == None:
        return []
    else:
        return [ch for ch in par if ch != node]

def extract_args(leaf, leaf_ind, stanford_deps, tree = None, avoid_words = []):
    """
    Extracts all the arguments of the node using stanford_dep. 
    stanford_deps is a list of StanfordDep.
    The arguments are defined by: 
    1. dependency type (according to SD)
    2. list of leaves in the sub-tree of the node: each is a lemma and POS
    3. head word
    """
    args = []
    features = []
    for standep in stanford_deps:
        if standep.head_ind == leaf_ind and \
                (standep.type in ARG_TYPES or standep.type.startswith('prep')) and \
                standep.dep not in avoid_words:
            head_node, head_node_ind = get_proximate_ancestor(tree,leaf,standep.dep_ind)
            new_arg = Arg(standep.type,head_node_ind,head_node)
            args.append(new_arg)
        elif standep.dep_ind == leaf_ind and standep.type == 'xcomp' and \
                standep.head in SECONDARY_VERBS: # if there is a secondary verb which modifies it
            features.append(('SECONDARY',standep.head))
    return args,features

def get_proximate_ancestor(tree,leaf1,leaf2_ind):
    """
    returns the minimal ancestor leaf2 which is a sibling of an ancestor of leaf1_ind.
    """
    ancestors1 = get_all_ancestors(leaf1)
    leaf2 = get_leaf(tree,leaf2_ind)
    if leaf2 != None:
        ancestors2 = get_all_ancestors(leaf2)
        min_ancestors = next( (x, y) for (x, y) in zip(ancestors1, ancestors2) if x!=y )
        head_node_ind = [x == leaf2 for x in get_leaves(min_ancestors[1])].index(True)
        return min_ancestors[1], head_node_ind

def get_all_ancestors(leaf):
    ancestors = []
    cur_node = leaf
    while cur_node != None:
        ancestors.append(cur_node)
        cur_node = cur_node.parent()
    ancestors.reverse()
    return ancestors
    

def read_tree(tree_string):
    """
    reads Stanford style phrase structure trees in string format and returns tree object
    """
    tree = Tree.fromstring(tree_string)
    parented_tree = ParentedTree.convert(tree)
    return parented_tree

def get_subtrees(tree):
    """
    returns all the subtrees including the main tree of a tree. 
    you are probably looking for get_children function.
    """
    subtrees = [subtree for subtree in tree.subtrees()]
    return subtrees

def get_leaves(tree):
    "returns the leaves of the tree which are Trees (i.e., the word and POS)"
    return [subtree for subtree in tree.subtrees() if subtree.height() == 2]

def get_children(tree):
    "returns the children sub-trees of tree"
    subtrees = get_subtrees(tree)
    children = []
    i = 1
    while i < len(subtrees):
        children.append(subtrees[i])
        subsubtrees = get_subtrees(subtrees[i])
        i = i + len(subsubtrees)
    return children

def head_leaf(candidate, rightToLeft):
    """
    Receives a Parented_Tree object. Extracts the top-most verb. 
    If there are several, returns the leftmost or the rightmost according to right.
    """
    # traverse the tree in bfs, left to right (or right to left)
    # if the node is noun or verb, return it
    if candidate.label().startswith('VB'):
        return candidate
    Q = []
    Q.append(candidate)
    while Q != []:
        T = Q.pop(0)
        child_iter = (reversed(T) if rightToLeft else iter(T))
        for ch in child_iter:
            if not isinstance(ch,Tree):
                continue
            elif ch.label().startswith('VB'):
                return ch
            else:
                Q.append(ch)
    return None

def find_pred(tree,right):
    """
    finds the first predicate to the right (if right==True) or to the 
    left (if right == False) in the Parented_Tree tree.
    """
    candidate = tree[0][1]
    sib = candidate
    while True:
        # if candidate has a right brother, take it
        while sib != None:
            if right:
                x = sib.right_sibling()
            else:
                x = sib.left_sibling()
            if x is None:
                sib = sib.parent()
            else:
                sib = x
                break
        if sib is None:
            break
        else:
            head = head_leaf(sib,not right)
            if head != None:
                return head
    return None

def get_all_events(stanford_deps,ptree):
    """
    returns all the events (i.e., verbs and their arguments) found in the tree.
    """
    leaves = get_leaves(ptree)
    output = []
    for leaf_ind,leaf in enumerate(leaves):
        if leaf.label().startswith('VB'):
            event = Event(leaf,leaf_ind+1,stanford_deps,ptree,[])
            if len(event.args()) >= 1 and event.pred()[0].isalpha() and \
                    event.pred() not in AUX_VERBS and (event.secondary_verb_of() == None):
                output.append(event)

    return output
            
   

def all_non_alphas(S):
    "checks if all elements in S all contain non-alphabet characters"
    return all([ (non_alpha.search(x) != None) for x in S])

def write_events_linkages(stanford_deps,ptree,f_out):
    # extract events
    events = get_all_events(stanford_deps,ptree)
    for event in events:
        f_out.write('EVENT '+str(event)+'\n')
    
    # extract temporal linkers
    connectives = {} # maps the id of the connective to its strings and leaf objects
    for ind,leaf in enumerate(get_leaves(ptree)):
        fields = str(leaf[0]).split('#')
        if len(fields) >= 3:
            leaf[0] = '#'.join(fields[:-2])
            if fields[-1].lower() == 'temporal':
                curL = connectives.get(fields[-2],[])
                curL.append((ind,leaf))
                connectives[fields[-2]] = curL

    # now we have the connectives, we move on to extracting the events
    # we find the POSs IN, RB, WRB (as 'when')
    for connective in connectives.values():
        rpred = find_pred(connective,True)
        lpred = find_pred(connective,False)
        if rpred != None and lpred != None:
            rpred_ind = leaf_index(rpred)
            lpred_ind = leaf_index(lpred)
            linkage = Linkage(lpred,lpred_ind,rpred,rpred_ind,connective,stanford_deps,ptree)
            if linkage.pred1() not in AUX_VERBS and linkage.pred2() not in AUX_VERBS:
                f_out.write(str(linkage)+'\n')



def main(args):
    if len(args) != 4:
        print('Usage: extract_events.py <trees with linkers> <corresponding SD file> <output>')
        sys.exit(-1)
    f = open(sys.argv[1])
    f_sd = open(sys.argv[2])
    f_out = open(sys.argv[3],'w')
    for line in f:
        stanford_deps = read_sent_sds(f_sd)
        ptree = read_tree(line.strip())
        write_events_linkages(stanford_deps,ptree,f_out)
    f_out.close()
    f.close()


sec_verb_list = preprocess_data.get_target_verb_list(\
    '/afs/inf.ed.ac.uk/user/o/oabend/event_ordering_resources/secondary_verbs.txt')
for v in sec_verb_list:
    SECONDARY_VERBS.extend(pattern.en.lexeme(v))

if __name__ == "__main__":
    main(sys.argv)


# check if the heuristic works for adverbials as well: sort of
# figure out the strange thing with the cycles: sort of OK
# ideally we should handle secondary verbs, eventive nouns
