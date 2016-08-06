import sys, pdb
import cPickle as pickle

if len(sys.argv) != 4:
    print('Usage: finalize_feature_list.py <feat. extractor> <train vectors> <output feature extract>')
    sys.exit(-1)

print('Loading feature extractor')
f_feat_extract = open(sys.argv[1])
feat_extractor = pickle.load(f_feat_extract)
f_feat_extract.close()
print('Feature extractor loaded')

f_pickle = open(sys.argv[2])
training_samples,training_labels,test_samples,test_labels,vec_feature_names = pickle.load(f_pickle)
f_pickle.close()
print('Feature list loaded')

feat_extractor.finish_feature_list([x for x in vec_feature_names if x in feat_extractor._feature_counts])

f_feat_extract_out = open(sys.argv[3],'w')
pickle.dump(feat_extractor,f_feat_extract_out)
f_feat_extract_out.close()
print('Feature extractor repickled')

