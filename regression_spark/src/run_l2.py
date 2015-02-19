'''
This is an example illustrating how to invoke the l2 solver.
'''

from pyspark import SparkContext
from least_squares import *
from matrix import *
from utils import *
import sys
import getopt
import numpy as np

def usage():
    print sys.exit(__doc__)

class ArgumentError(Exception):
    pass

class OptionError(Exception):
    pass

if __name__ == "__main__":

    try:                                
        opts, args = getopt.getopt(sys.argv[1:], "hd:     p:k:s:c:m:n:", ["help","data=","low","high","projection","sampling","npartitions="])
    except getopt.GetoptError:          
        usage()              
        sys.exit(2)

    proj_methods = ('gaussian','rademacher','srdht','cw')

    #setting default variables
    (low_precision, high_precision) = (False, False)
    (projection, sampling) = (False, False)
    (c, s, k, projection_type, m, n) = (None, None, 3, None, None, None)
    load_N = True #load N matrices whenever possible to save time
    save_N = True #save N matrices generated during the run for future use
    npartitions = 300
    local_file = True
    dire = '../data/'

    for (opt, arg) in opts:
        #print (opt, arg)
        if opt in ('-h','--help'):
            usage()
        elif opt in ('-d','--data'):
            filename = arg
        elif opt == '--low':
            low_precision = True
            if low_precision and high_precision:
                raise OptionError('Options --low and --high cannot be chosen at the same time!')
        elif opt == '--high':
            high_precision = True
            if low_precision and high_precision:
                raise OptionError('Options --low and --high cannot be chosen at the same time!')
        elif opt == '--projection':
            projection = True
            if projection and sampling:
                raise OptionError('Options --projection and --sampling cannot be chosen at the same time!')
        elif opt == '--sampling':
            sampling = True
            if projection and sampling:
                raise OptionError('Options --projection and --sampling cannot be chosen at the same time!')
        elif opt == '-p':
            if arg not in proj_methods:
                a = 'Argument of -p must be gaussian, rademacher, srdht or cw, now is {0}!'.format(arg)
                raise ArgumentError(a)
            projection_type = arg
        elif opt == '-c':
            try:
                c = int(arg)
            except:
                raise ArgumentError('Bad value for -c')
        elif opt == '-s':
            try:
                s = int(arg)
            except:
                raise ArgumentError('Bad value for -s')
        elif opt == '-k':
            try:
                k = int(arg)
            except:
                raise ArgumentError('Bad value for -k')
        elif opt == '-m':
            try:
                m = int(arg)
            except:
                raise ArgumentError('Bad value for -m')
        elif opt == '-n':
            try:
                n = int(arg)
            except:
                raise ArgumentError('Bad value for -n')

    if c is None:
        raise ValueError('No value for c!')

    if projection_type is None:
        projection_type = 'gaussian'

    if low_precision and projection:
        kwargs = {'solver_type':'low_precision','sketch_type':'projection','projection_type':projection_type,'c':c,'k':k}

    sc = SparkContext(appName="l2_exp")

    #loading dataset
    if local_file:
        Ab = np.loadtxt(dire+filename+'_Ab.txt') #loading dataset from local disc
        Ab_rdd = sc.parallelize(Ab.tolist(),npartitions)
    else:
        Ab_rdd = sc.textFile('data/'+filename+'_Ab.txt',npartitions) #loading dataset from HDFS

    matrix_Ab = Matrix(Ab_rdd,filename,m,n,True)
    matrix_Ab.zip_with_index()

    ls = RandLeastSquares(matrix_Ab,**kwargs)

    ls.fit(load_N, save_N)

    print 'time elapsed:', ls.time, '\n'

    sys.exit(2)

    b = np.loadtxt('../data/'+filename+'_b.txt')
    x_opt = np.loadtxt('../data/'+filename+'_x_opt.txt')
    f_opt = np.loadtxt('../data/'+filename+'_f_opt.txt')

    #instantiate a RLS object
    #ls = RandLeastSquares(matrix_Ab,solver_type='low_precision',sketch_type='projection',projection_type='cw',c=1e4,k=3)
    #ls = RandLeastSquares(matrix_Ab,solver_type='low_precision',sketch_type='projection',projection_type='gaussian',c=5e3,k=3)
    ls = RandLeastSquares(matrix_Ab,solver_type='low_precision',sketch_type='projection',projection_type='srdht',c=1e3,k=1)
    #ls = RandLeastSquares(matrix_Ab,solver_type='low_precision',sketch_type='sampling',sc=sc,projection_type='cw',c=1e3,s=1e3,k=3)
    #ls = RandLeastSquares(matrix_Ab,solver_type='high_precision',sc=sc,sketch_type='projection',projection_type='cw',c=5e4,num_iter=5,k=3)

    #solving the least squares problems

    #evaluating the solutions
    rx, rf = ls.comp_relerr(b,x_opt,f_opt)

    print 'relative error on x is:', rx, ', relative error on f is:', rf, '\n'


