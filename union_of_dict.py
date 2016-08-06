'''
Created on 30 Sep 2013

@author: oabend
'''
import pickle, sys, os

def add_vectors(d1,d2):
    output = d1
    for k in d2.keys():
        output[k] = output.get(k,0) + d2[k]
    return output
    

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: union_of_dict <directory> <pickle dump>')
        sys.exit(-1)
    
    filenames = os.listdir(sys.argv[1])
    L = []
    for filename in filenames:
        if filename.endswith('.pickle'):
            f = open(filename)
            L.append(pickle.load(f))
            f.close()
    
    d_total = L.pop()
    for d in L:
        add_vectors(d_total, d)
    
    f_out = open(sys.argv[2], 'wb')
    pickle.dump(d_total, f_out)
    f_out.close()
    
    
        
        