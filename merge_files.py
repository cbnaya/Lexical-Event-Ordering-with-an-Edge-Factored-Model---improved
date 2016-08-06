import extract_features, decoder, driver, sys
import cPickle as pickle

#files = ['all_full_5lda.skip_0_4000', 'all_full_5lda.skip_4000_8000','all_full_5lda.skip_8000_12000',\
#             'all_full_5lda.skip_12000_16000','all_full_5lda.skip_16000_20000',\
#             'all_full_5lda.skip_20000_40000','all_full_5lda.skip40000']

if len(sys.argv) != 3:
    print('Usage: merge_files <intput filenames :-delimited> <output filename>')
    sys.exit(-1)

files = sys.argv[1].split(':')
out_filename = sys.argv[2]

train_set = []
train_perms = []


for ind,filename in enumerate(files):
    print('loading '+filename)
    f = open(filename)
    training_ins,training_perms,test_set,test_perms,feature_names = pickle.load(f)
    f.close()
    print(filename+' loaded')
    
    if ind == 0:
        train_set.extend(training_ins)
        train_perms.extend(training_perms)
    elif ind < len(files) - 1:
        if len(training_ins[0]._edges) != len(train_set[-1]._edges):
            raise Exception('incompatible files')
        train_set.extend(training_ins[1:])
        train_perms.extend(training_perms[1:])
    else:
        if len(training_ins[0]._edges) != len(train_set[-1]._edges):
            raise Exception('incompatible files')
        train_set.extend(training_ins[1:])
        train_perms.extend(training_perms[1:])

print('Num train instances:'+str(len(train_set)))
print('Num test instances:'+str(len(test_set)))

# scrambling
train_set, train_perms = driver.scramble_vertices(train_set,train_perms)
test_set,test_perms = driver.scramble_vertices(test_set,test_perms)

f_out = open(out_filename,'w')
pickle.dump((train_set,train_perms,test_set,test_perms,feature_names),f_out,pickle.HIGHEST_PROTOCOL)
f_out.close()


