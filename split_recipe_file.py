import random, sys, pdb

if len(sys.argv) != 6:
    print('Usage: split_recipe_file <event files to split :-delimited> <training output> <dev output> <test output> <recipe numbers>')
    sys.exit(-1)

random.seed(10)
# the file names
all_files = sys.argv[1].split(':')
training_output = sys.argv[2]
dev_output = sys.argv[3]
test_output = sys.argv[4]
recipe_numbers_output = sys.argv[5]

recipes = []

for fn in all_files:
    f = open(fn)
    recipe_index = 0
    for line in f:
        line = line.strip()
        if line == "=========":
            recipes.append((fn,recipe_index))
            recipe_index += 1
    f.close()

shuffled_recipes = random.sample(recipes,len(recipes))

training_end_index = int(0.8 * len(recipes))
dev_end_index = int(0.9 * len(recipes))

train_recipes = [(x,'train') for x in shuffled_recipes[:training_end_index]]
dev_recipes = [(x,'dev') for x in shuffled_recipes[training_end_index:dev_end_index]]
test_recipes = [(x,'test') for x in shuffled_recipes[dev_end_index:]]
recipe_mapping = dict(train_recipes + dev_recipes + test_recipes)

# outputting the event files
f_training = open(training_output,"w")
f_dev = open(dev_output,"w")
f_test = open(test_output,"w")

for fn in all_files:
    f = open(fn)
    recipe_index = 0
    cur_recipe = ""
    for line in f:
        line = line.strip()
        cur_recipe = cur_recipe + line+'\n'
        if line == "=========":
            if recipe_mapping[(fn,recipe_index)] == 'train':
                f_training.write(cur_recipe)
            elif recipe_mapping[(fn,recipe_index)] == 'dev':
                f_dev.write(cur_recipe)
            elif recipe_mapping[(fn,recipe_index)] == 'test':
                f_test.write(cur_recipe)
            else:
                raise Exception
            cur_recipe = ""
            recipe_index += 1
    f.close()

f_training.close()
f_dev.close()
f_test.close()

# outputting the recipe numbers
f_recipe_numbers = open(recipe_numbers_output,"w")
for rec,label in recipe_mapping.items():
    f_recipe_numbers.write(' '.join([str(rec[0]),str(rec[1]),str(label)])+'\n')
f_recipe_numbers.close()

