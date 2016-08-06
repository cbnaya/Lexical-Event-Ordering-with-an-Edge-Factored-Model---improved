import extract_features, decoder, driver, time
import cPickle as pickle
import pdb


"""
print('Loading')
skipped_tr,skipped_tr_perm,skipped_te,skipped_te_perm,x = pickle.load(open('x'))
non_skipped_tr,non_skipped_tr_perm,non_skipped_te,non_skipped_te_perm,y = \
    pickle.load(open('debug_fe_pickle_instances'))
print('Done loading')
print(len(non_skipped_tr))
t = time.time()
non_skipped_tr_new, non_skipped_tr_perm_new = driver.scramble_vertices(non_skipped_tr,non_skipped_tr_perm)
non_skipped_te_new, non_skipped_te_perm_new = driver.scramble_vertices(non_skipped_te,non_skipped_te_perm)
print(time.time() - t)

pickle.dump((non_skipped_tr_new, non_skipped_tr_perm_new,non_skipped_te_new, non_skipped_te_perm_new,x),open('debug_fe_pickle_instances.scrambled','w'))
"""

(non_skipped_tr_new, non_skipped_tr_perm_new,non_skipped_te_new, non_skipped_te_perm_new,x) = \
    pickle.load(open('debug_fe_pickle_instances.scrambled'))

non_skipped_tr_new, non_skipped_tr_perm_new = driver.scramble_vertices(non_skipped_tr_new,non_skipped_tr_perm_new)

perceptron = decoder.StructuredPerceptron(len(x),1,1.0,True,x,False,True)
#perceptron.fit(non_skipped_tr_new+non_skipped_te_new, non_skipped_tr_perm_new+non_skipped_te_perm_new)

#test_res = perceptron.test_on_data(non_skipped_te_new,non_skipped_te_perm_new)
#test_res = perceptron.test_on_data(non_skipped_tr_new,non_skipped_tr_perm_new)
test_res = perceptron.test_on_data(non_skipped_te_new,non_skipped_te_perm_new)
print(test_res)

