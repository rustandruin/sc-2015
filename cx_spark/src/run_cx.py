'''
CX decomposition with approximate leverage scores
Usage:
Input options:
  -h [ --help ]                     check for usage
  -d [ --data ] arg                 arg_A.txt stores the input matrix to run CX on
                                    arg_U.txt stores the lef-singular vectors of the input matrix (for test purpose)
                                    arg_D.txt stores the singular values of the input matrix (for test purpose)
  -m, -n arg                        the size of the input matrix is m by n                              
  -b [ --dire ] arg (='../data/')   directory that stores the matrix file and relavant files
  -k [ --rank ] arg (=5)            rank parameter in the definition of leverage scores
                                    this value shoud not be greater than m or n
  -r [ --nrows ] arg (=20)          number of rows to select in CX
  -q [ -- niters ] arg (=2)         number of iterations to run in approximation of leverage scores 
  --determined, --randomized        when selecting rows, use determined or randomized scheme
                                    cannot flag both, default is --determined
  --no_load_n                       do not reuse the precomputed N files
  --no_save_n                       do not save the computed N files
  --npartitions arg (=200)          number of partitions in Spark

Output options:
  -l [ --leverage ]                 only return the approximate leverage scores and compute the approximation accuracy
  -i [ --index ]                    only return the selected row indices
  -f [ --full ]                     comput the full CX decomposition and evaluate its reconstruction accuracy
                                    if no stage if specified, default is -f 
                                    note flag for earlier stage will be suppresed by flag for later stage
                                    for example, -l will be suppressed by -i
'''

from pyspark import SparkContext
from cx import *
from matrix import *
from utils import *
import sys
import getopt
import scipy.stats
import numpy as np

def usage():
    print sys.exit(__doc__)

def print_params(**params):
    print 'Start experiment!'
    print 'filename: {0}'.format( params['filename'] )
    print 'size: {0} by {1}'.format( params['m'],params['n'] )
    print 'number of partitions: {0}'.format( params['npartitions'] )
    print 'rank: {0}'.format( params['k'] )
    print 'number of rows to select: {0}'.format( params['r'] )
    print 'number of iterations to run: {0}'.format( params['q'] )
    print 'scheme to use: {0}'.format( params['scheme'] )
    print 'stages to run: {0}'.format( params['stage'] )
    print '----------------------------------------------'

def check_input(s,arg):
    try:
        pa[s] = int(arg)
    except:
        raise ArgumentError('Bad value({1}) for option {0}'.format(s,arg))

class ArgumentError(Exception):
    pass

class OptionError(Exception):
    pass

if __name__ == "__main__":

    #setting default values
    pa = {}
    (dire, pa['scheme'], pa['stage']) = ('../data/', 'deter', 3)
    (pa['filename'], pa['k'], pa['r'], pa['q'], pa['m'], pa['n'], pa['npartitions']) = (None, 5, 20, 2, None, None, 200)
    (load_N,save_N,leverage,index,full,deterministic,randomized) = (True, True, False, False, False, False, False)

    try:
        opts, args = getopt.getopt(sys.argv[1:],'d:b:lifk:r:q:     m:n:',['data=','directory=','leverage','index','full','rank='
            ,'nrows=','niters=','no_load_n','no_save_n','deterministic','randomized','npartitions='])
    except getopt.GetoptError:          
        usage()              
        sys.exit(2)

    #parsing parameters
    for (opt, arg) in opts:
        #print (opt, arg)
        if opt in ('-h','--help'):
            usage()
        elif opt in ('-d','--data'):
            pa['filename'] = arg
        elif opt in ('-b','--dire'):
            dire = arg
        elif opt in ('-l','--leverage'):
            leverage = True
        elif opt in ('-i','--index'):
            index = True
        elif opt in ('-f','--full'):
            full = True
        elif opt == '--npartitions':
            check_input('npartitions',arg)
        elif opt == '--deterministic':
            deterministic = True
        elif opt == '--randomized':
            randomized = True
        elif opt in ('-r','--nrows'):
            check_input('r',arg)
        elif opt in ('-k','--rank'):
            check_input('k',arg)
        elif opt in ('-q','--niters'):
            check_input('q',arg)
        elif opt == '-m':
            check_input('m',arg)
        elif opt == '-n':
            check_input('n',arg)
        elif opt == '--no_load_n':
            load_N = False
        elif opt == '--no_save_n':
            save_N = False

    #validating
    if pa['r'] is None:
        raise ValueError('No value for number of rows to select!')

    if pa['k'] > pa['m'] or pa['k'] > pa['n']:
        raise ValueError('Rank parameter({0}) should not be greater than m({1}) or n({2})'.format(pa['k'],pa['m'],pa['n']))

    if pa['m'] < pa['n']:
        raise ValueError('Number of rows({0}) should be greater than number of columns({1})').format(pa['m'],pa['n'])

    if deterministic and randomized:
                raise OptionError('Options --deterministic and --randomized cannot be chosen at the same time!')

    pa['scheme'] = 'deterministic' if deterministic else 'randomized' #this means default is rand

    if full:
        pa['stage'] = 3
    elif index:
        pa['stage'] = 2
    elif leverage:
        pa['stage'] = 1

    #print parameters
    print_params(**pa)

    #instantializing a Spark instance    
    sc = SparkContext(appName="cx_exp")

    #Ab_rdd = sc.textFile('data/'+filename+'_Ab.txt',340) #loading dataset from HDFS
    A = np.loadtxt(dire+pa['filename']+'_A.txt') #loading dataset from local disc
    D = np.loadtxt(dire+pa['filename']+'_D.txt')
    U = np.loadtxt(dire+pa['filename']+'_U.txt')
    A_rdd = sc.parallelize(A.tolist(),pa['npartitions'])

    matrix_A = Matrix(A_rdd,pa['filename'],pa['m'],pa['n'],True)
    matrix_A.zip_with_index()

    cx = CX(matrix_A,sc=sc)

    if pa['stage'] > 0:
        lev, p = cx.get_lev(pa['k'], load_N, save_N, q=pa['q']) #getting the approximate row leverage scores. it has the same size as the number of rows 
        #lev, p = cx.get_lev(n, load_N, save_N, projection_type='gaussian',c=1e3) #target rank is n

        lev_exact = np.sum(U[:,:pa['k']]**2,axis=1)
        p_exact = lev_exact/pa['k']
        print 'KL divergence between the estimation of leverage scores and the exact one is {0}'.format( scipy.stats.entropy(p_exact,p) )
        print 'finished stage 1'
        print '----------------------------------------------'

    if pa['stage'] > 1:
        idx = cx.comp_idx(pa['scheme'],pa['r']) #choosing rows based on the leverage scores
        #maybe to store the indices to file
        print 'finished stage 2'
        print '----------------------------------------------'

    if pa['stage'] > 2:
        rows = cx.get_rows() #getting back the selected rows based on the idx computed above (this might give you different results if you rerun the above)
        diff = cx.comp_err() #computing the relative error

        print 'relative error ||A-CX||/||A|| is {0}'.format( diff/np.linalg.norm(A,'fro') )
        print 'raltive error of the best rank-{0} approximation is {1}'.format( pa['k'], np.sqrt(np.sum(D[pa['k']:]**2))/np.sqrt(np.sum(D**2)) )
        print 'finished stage 3'
    
