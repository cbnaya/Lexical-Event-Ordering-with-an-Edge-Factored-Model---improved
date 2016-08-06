'''
Created on 30 Sep 2013

@author: oabend
'''
import copy, math

class SparseVectorsUtil:

    @staticmethod
    def add_vectors(d1,d2):
        output = copy.copy(d1)
        for k in d2.keys():
            output[k] = output.get(k,0) + d2[k]
        return output
    
    @staticmethod
    def dot_product(d1,d2):
        output = 0
        for k in set(d1.keys()) & set(d2.keys()):
            output += d1[k]*d2[k]
        return output
    
    @staticmethod
    def norm(d1):
        return math.sqrt(sum([d1[x]*d1[x] for x in d1.keys()]))
    
    @staticmethod
    def cosine_vectors(d1,d2):
        n1 = SparseVectorsUtil.norm(d1)
        n2 = SparseVectorsUtil.norm(d2)
        if n1 == 0 or n2 == 0:
            return 0
        else:
            return SparseVectorsUtil.dot_product(d1,d2)/(n1*n2)

        