import cPickle as pickle
import sys, time


f_out = open(sys.argv[1])
t = time.time()
x = pickle.load(f_out)
print(len(x[0]),len(x[1]),len(x[2]),len(x[3]))
print('Time elapsed:'+str(time.time() - t))
f_out.close()


