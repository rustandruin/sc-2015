import numpy.linalg as npl
import numpy as np

from utils import *

def parse(row,lim='all'):
    if isinstance(row, unicode):
        row = [float(x) for x in row.split(' ')]

    if lim == 'all':
        return row
    elif lim == "b":
        return row[-1]
    elif lim == "A":
        return row[:-1]
    else:
        print "Please enter a correct value!"

def parse_get_key(row,lim='all'):
    return (row[1], parse(row[0],lim))

def comp_l2_obj(Ab_rdd, x):
    # x is a np array
    return np.sqrt( Ab_rdd.map( lambda row: (np.dot(parse(row,'A'),x) - parse(row,'b'))**2 ).reduce(add) )

def add_index(rdd): 
    starts = [0] 
    nums = rdd.mapPartitions(lambda it: [sum(1 for i in it)]).collect() 
    for i in range(len(nums) - 1): 
        starts.append(starts[-1] + nums[i]) 

    def func(k, it): 
        for i, v in enumerate(it, starts[k]): 
            yield v, i

    return rdd.mapPartitionsWithIndex(func)

class SRDHT_Map(Block_Mapper):
    def __init__(self,c,k,m,seed_s,lim):
        Block_Mapper.__init__(self)
        self.m = m
        self.seed_s = seed_s
        self.c = c
        self.k = k
        self.lim = lim
        self.PA = [None for i in range(self.k)]

    def process(self):
        data = np.array(self.data)
        r = data.shape[0]
        row_idx = np.array(self.key)

        for i in xrange(self.k):
            S = np.arange(self.m)
            np.random.seed(self.seed_s[i])
            np.random.shuffle(S)
            S = S[:self.c]
            np.random.seed()
            rs = (np.random.rand(r)<0.5)*2-1
            rand_data = np.dot(np.diag(rs),data)
            #for j in xrange(self.c):
            #    yield ((i,j), np.dot( np.sqrt(2)*np.cos(2*np.pi*S[j]*row_idx/self.m-np.pi/4), rand_data)/np.sqrt(self.m))

            if self.PA[i] is None:
                self.PA[i] = np.dot( np.sqrt(2)*np.cos(2*np.pi*np.outer(S,row_idx)/self.m-np.pi/4), rand_data)/np.sqrt(self.m)
            else:
                self.PA[i] += np.dot( np.sqrt(2)*np.cos(2*np.pi*np.outer(S,row_idx)/self.m-np.pi/4), rand_data)/np.sqrt(self.m) 

        return iter([])

    def parse(self,row):
        return parse_get_key(row,self.lim)

    def close(self):
        for i in xrange(self.k):
            for j in xrange(self.c):
                yield ( (i,j), self.PA[i][j,:] )

class CW_Map(Block_Mapper):
    def __init__(self,c,k,lim):
        Block_Mapper.__init__(self,1)
        self.c = c
        self.k = k
        self.lim = lim

    def process(self):
        row = np.array(self.data[0])
        np.random.seed()
        rt = np.random.randint(self.c,size=self.k).tolist()
        coin = (np.random.rand(self.k)<0.5)*2-1
        for i in xrange(self.k):
            yield ((i,rt[i]),coin[i]*row)

    def parse(self,row):
        return parse(row,self.lim)

class Gaussian_Map(Block_Mapper):
    def __init__(self,c,k,lim):
        Block_Mapper.__init__(self)
        self.c = c  #projection size
        self.k = k  #number of independent trials
        self.lim = lim
        self.PA = [None for i in range(self.k)]

    def process(self):
        data = np.array(self.data)
        r = data.shape[0]

        np.random.seed()
        for i in xrange(self.k):
            if self.PA[i] is None:
                self.PA[i] = np.dot(np.random.randn(self.c,r),data)/np.sqrt(self.c)
            else:
                self.PA[i] += np.dot(np.random.randn(self.c,r),data)/np.sqrt(self.c)

        return iter([])

    def parse(self,row):
        return parse(row,self.lim)

    def close(self):
        print 'a'
        for i in xrange(self.k):
        #    yield ( i, self.PA[i] )
            for j in xrange(self.c):
                yield ( (i,j), self.PA[i][j,:] )

class Rademacher_Map(Block_Mapper):
    def __init__(self,c,k,lim):
        Block_Mapper.__init__(self)
        self.c = c
        self.k = k
        self.lim = lim
        self.PA = [None for i in range(self.k)]

    def process(self):
        data = np.array(self.data)
        r = data.shape[0]

        np.random.seed()
        for i in xrange(self.k):
            if self.PA[i] is None:
                self.PA[i] = np.dot((np.random.rand(self.c,r)<0.5)*2-1,data)/np.sqrt(self.c)
            else:
                self.PA[i] += np.dot((np.random.rand(self.c,r)<0.5)*2-1,data)/np.sqrt(self.c)
   
        return iter([])

    def parse(self,row):
        return parse(row,self.lim)

    def close(self):
        print 'a'
        for i in xrange(self.k):
        #    yield ( i, self.PA[i] )
            for j in xrange(self.c):
                yield ( (i,j), self.PA[i][j,:] )

def get_x(pa,return_N=False):
    pa = [row for row in pa]
    pa = np.array(pa)
    A = pa[:, :-1]
    b = pa[:, -1]
    m = A.shape[0]

    [U, s, V] = npl.svd(A, 0)
    N = V.transpose()/s

    if return_N:
        return (m, (N, np.dot(N, np.dot(U.T,b))))
    else:
        return (m, np.dot(N, np.dot(U.T,b)))

def get_N(pa,alg='cx'):
    pa = [row for row in pa]
    if alg == 'cx':
        pa = np.array(pa)
    elif alg == 'ls':
        pa = np.array(pa)[:,:-1]
    [U, s, V] = npl.svd(pa, 0)
    N = V.transpose()/s
    return N

def comp_lev_sum(rows,sc,N,alg):
    N = sc.broadcast(N)
    if alg == 'cx':
        return rows.map(lambda row:xN(parse(row,'all'),N.value)).reduce(add).tolist()
    elif alg == 'ls':
        return rows.map(lambda row:xN(parse(row,'A'),N.value)).reduce(add).tolist()

def comp_lev(rows,sc,N,alg):
    N = sc.broadcast(N)
    if alg == 'cx':
        return rows.map(lambda row:(row[1], xN(parse(row[0],'all'),N.value))).sortByKey().collect()
    elif alg == 'ls':
        return rows.map(lambda row:(row[1], xN(parse(row[0],'A'),N.value))).sortByKey().collect()

def xN(x,N):
    return np.array([npl.norm(np.dot(np.array(x),N1))**2 for N1 in N])

def sample_solve(rows,sc,N,sumLev,s,return_N=False):
    N = sc.broadcast(N)
    return rows.flatMap(lambda row:sample(parse(row),N.value,sumLev,s)).groupByKey().map(lambda sa: get_x(sa[1],return_N)).collect()

def sample_svd(rows,sc,N,sumLev,s,alg):
    N = sc.broadcast(N)
    return rows.flatMap(lambda row:sample(parse(row),N.value,sumLev,s)).groupByKey().map(lambda sa: get_N(sa[1],alg)).collect()

def sample(x,N,sumLev,s):
    x = np.array(x)
    k = len(N)
    np.random.seed()
    for i in range(k):
        q = npl.norm(np.dot(x[:-1],N[i]))**2
        p = min(q*s/sumLev[i],1.0)
        if np.random.rand() < p:
            yield (i,(x/p).tolist())

def unif_sample_solve(rows,m,k,s):
    return rows.flatMap(lambda row:unif_sample(parse(row),k,m,s)).groupByKey().map(lambda sa: get_x(sa[1])).collect()

def unif_sample(x,k,m,s):
    x = np.array(x)
    np.random.seed()
    for i in range(k):
        p = s/m
        if np.random.rand() < p:
            yield (i,x.tolist())

def matvec(A,vec,lr,sc,lim='all'):
    vec = sc.broadcast(vec)
    if lr=='r':
        b = A.map(lambda row:(row[1],np.dot(np.array(parse(row[0],lim)),vec.value))).sortByKey().values().collect()
        b = np.array(b)
    elif lr=='l':
        b = A.map(lambda row:np.array(parse(row[0],lim))*vec.value[row[1]]).reduce(add)

    return b

def matmat(A,mat,lr,sc,lim='all'):
    mat = sc.broadcast(mat)
    if lr=='r':
        b = A.map(lambda row:(row[1],np.dot(np.array(parse(row[0],lim)),mat.value).tolist())).sortByKey().values().collect()
        b = np.array(b)
    elif lr=='l':
        b = A.map(lambda row: np.outer( mat.value[:,row[1]] , parse(row[0],lim) )).reduce(add)

    return b

def idx_in(row, idx):
    if row[1] in idx:
        return True
    else:
        return False

def compLevExact(A, k, axis):
    """ This function computes the column or row leverage scores of the input matrix.
         
        :param A: n-by-d matrix
        :param k: rank parameter, k <= min(n,d)
        :param axis: 0: compute row leverage scores; 1: compute column leverage scores.
        
        :returns: 1D array of leverage scores. If axis = 0, the length of lev is n.  otherwise, the length of lev is d.
    """

    U, D, V = np.linalg.svd(A, full_matrices=False)

    if axis == 0:
        lev = np.sum(U[:,:k]**2,axis=1)
    else:
        lev = np.sum(V[:k,:]**2,axis=0)

    p = lev/k

    return lev, p

def _test_matmat():
    #A = np.random.rand(1000,50)
    #u = np.random.rand(50,10)
    #v = np.random.rand(20,1000)

    sc = SparkContext(appName="test") # initiate an Spark object
    #A_rdd = sc.parallelize(A.tolist(),4) # Create a RDD for the rows of A.
    A_rdd = sc.textFile('data/nonunif_bad_10000_50_Ab.txt', 100)
    A = np.loadtxt('../data/nonunif_bad_10000_50_Ab.txt')
    u = np.random.rand(51,10)
    v = np.random.rand(20,10000)
    A_rdd = add_index(A_rdd)

    print np.array(parse(A_rdd.first()[0],'all')).shape

    #accum = sc.accumulator(np.zeros(500), VectorAccumulatorParam())
    #A_rdd.foreach(lambda x: accum.add(np.array(x[1])*v_vec.value[x[0]]))
    #c = accum.value

    b = matmat(A_rdd,u,'r',sc)
    c = matmat(A_rdd,v,'l',sc)

    print np.linalg.norm(b - np.dot(A,u))
    print np.linalg.norm(c - np.dot(v,A))


if __name__ == '__main__':
    from pyspark import SparkContext
    from pyspark.accumulators import AccumulatorParam
    from utils import *

    _test_matmat()


