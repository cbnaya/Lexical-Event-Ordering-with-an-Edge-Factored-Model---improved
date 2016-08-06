import sys
import cPickle as pickle

if len(sys.argv) != 5:
    print('Usage: take_prefix <input filename> <output filename> <training prefix size> <test prefix size>')
    sys.exit(-1)

filename = sys.argv[1]
out_filename = sys.argv[2]
prefix_size_train = int(sys.argv[3])
prefix_size_test = int(sys.argv[4])

print('loading '+filename)
f = open(filename)
train_set,train_perms,test_set,test_perms,feature_names = pickle.load(f)
f.close()

train_set = train_set[:prefix_size_train]
train_perms = train_perms[:prefix_size_train]
test_set = test_set[:prefix_size_test]
test_perms = test_perms[:prefix_size_test]

print("train size "+str(len(train_set)))
print("test size "+str(len(test_set)))

f_out = open(out_filename,'w')
pickle.dump((train_set,train_perms,test_set,test_perms,feature_names),f_out,pickle.HIGHEST_PROTOCOL)
f_out.close()



