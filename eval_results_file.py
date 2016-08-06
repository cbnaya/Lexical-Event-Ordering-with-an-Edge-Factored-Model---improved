"""
Runs an evaluation on the results file.
"""
import sys, re, decoder, math
import pdb

def eval_perms(predicted,correct):
    if predicted == correct:
        exact_match = 1
    else:
        exact_match = 0
    pair_swaps = decoder.num_agreeing_tuples(predicted,correct,2)
    triple_swaps = decoder.num_agreeing_tuples(predicted,correct,3)
    quad_swaps = decoder.num_agreeing_tuples(predicted,correct,4)
    if pair_swaps[1] > 0:
        micro_pair = 1.0 * pair_swaps[0] / pair_swaps[1]
        random_micro = 0.5
    else:
        micro_pair = 1.0
        random_micro = 1.0
    random_exact = 1.0 / math.factorial(len(predicted))
    return exact_match,pair_swaps,triple_swaps,quad_swaps,micro_pair,random_micro, random_exact

if len(sys.argv) != 2:
    print('Usage: python eval_results_file.py <results file>')
    sys.exit(-1)

nums = re.compile('[0-9]+')
f = open(sys.argv[1])
predicted_or_correct = True
all_pair_swaps = []
all_triple_swaps = []
all_quad_swaps = []
tot_micro_pair = 0
tot_samples = 0 
tot_exact = 0
tot_random_exact = 0
tot_random_micro = 0

for line in f:
    line = line.strip()
    if line.startswith('['):
        L = [int(x) for x in nums.findall(line)]
        L2 = []
        for x in range(len(L)/2):
            L2.append((L[2*x],L[2*x+1]))
        if predicted_or_correct:
            predicted = decoder.convert_edges_perm(L2)
            predicted_or_correct = False
        else:
            correct = decoder.convert_edges_perm(L2)
            predicted_or_correct = True
            exact_match,pair_swaps,triple_swaps,quad_swaps,micro_pair,random_micro,random_exact = \
                eval_perms(predicted,correct)
            all_pair_swaps.append(pair_swaps)
            all_triple_swaps.append(triple_swaps)
            all_quad_swaps.append(quad_swaps)
            tot_micro_pair += micro_pair
            tot_random_micro += random_micro
            tot_random_exact += random_exact
            tot_exact += exact_match
            tot_samples += 1.0
            if tot_samples % 10 == 0:
                print('Num samples:'+str(tot_samples))

f.close()
macro_avg_pair_swaps = 1.0 * sum([p[0] for p in all_pair_swaps]) / sum([p[1] for p in all_pair_swaps])
macro_avg_triple_swaps = 1.0 * sum([p[0] for p in all_triple_swaps]) / sum([p[1] for p in all_triple_swaps])
macro_avg_quad_swaps = 1.0 * sum([p[0] for p in all_quad_swaps]) / sum([p[1] for p in all_quad_swaps])
final_exact = 1.0 * tot_exact / tot_samples
final_micro_pair = 1.0 * tot_micro_pair / tot_samples
final_random_exact = 1.0 * tot_random_exact / tot_samples
final_random_micro = 1.0 * tot_random_micro / tot_samples

L = [macro_avg_pair_swaps,macro_avg_triple_swaps,macro_avg_quad_swaps,final_exact,final_random_exact,final_micro_pair,final_random_micro]
print(' '.join([str(x) for x in L]))

