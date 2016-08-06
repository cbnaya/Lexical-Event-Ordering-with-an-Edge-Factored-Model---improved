"""
Receives an index of the recipes and a list of (preprocessed) text files.
Writes the recipes that match each of the data sets to a separate file.
"""
import sys
import random
random.seed(7)


f = open(sys.argv[1])
L = []
for line in f:
    fields = line.strip().split()
    L.append((fields[1], int(fields[0])))
f.close()

L = random.sample(L,len(L))
tot_recipes = 0
train_files = []
test_files = []
dev_files = []

for filename,recipes in L:
    tot_recipes += recipes
    if tot_recipes < 60000:
        train_files.append((filename,recipes))
    elif tot_recipes < 67500:
        test_files.append((filename,recipes))
    else:
        dev_files.append((filename,recipes))
   

print(' '.join([x[0] for x in train_files])+'\n')
print(sum([x[1] for x in train_files]))
print(' '.join([x[0] for x in test_files])+'\n')
print(sum([x[1] for x in test_files]))
print(' '.join([x[0] for x in dev_files])+'\n')
print(sum([x[1] for x in dev_files]))


"""
def non_empty(L):
    "returns True if L contains a non-empty string"
    for x in L:
        if x != '':
            return True
    return False

if len(sys.argv) != 3:
    print('Usage: grep_orig_text.py <index file> ' + \
              '<preprocessed recipe files :-delimited>')
    sys.exit(-1)

f = open(sys.argv[1])
index = {}
for line in f:
    fields = line.strip().split()
    index[(fields[0].split('.')[0],int(fields[1]))] = fields[2]
f.close()

filenames = sys.argv[2].split(':')
for filename in filenames:
    recipe_index = 0
    cur_recipe = []
    f = open(filename)
    f_outs = {}
    f_outs['train'] = open(filename+'.train','w')
    f_outs['test'] = open(filename+'.test','w')
    f_outs['dev'] = open(filename+'.dev','w')
    for line in f:
        line = line.strip()
        if 'END_RECIPE' in line:
            if non_empty(cur_recipe):
                f_outs[index[(filename.split('.')[0],recipe_index)]].write('\n'.join(cur_recipe)+'\n')
                cur_recipe = []
                recipe_index += 1
        else:
           cur_recipe.append(line)
    f.close()
    for f_out in f_outs.values():
        f_out.close()


            

    
"""
