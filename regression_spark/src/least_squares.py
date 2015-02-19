'''
This class performs randomized algorithms for linear regressions on Spark.
parameters:
    matrix_Ab: a 'matrix' object which stores the matrix A and column b and relevant information
    solver_type: it can be 'projection', 'sampling_lev', 'sampling_unif' and 'lsqr'
    k: number of solution to get through a single run

Author: Jiyan Yang (jiyan@stanford.edu)
'''

from projections import *
from rma_utils import *
from matrix import *
from lsqr_spark import *
from comp_sketch import *
import time

class RandLeastSquares:
    def __init__(self, matrix_Ab, solver_type, **kwargs):
        self.matrix_Ab = matrix_Ab
        self.solver_type = solver_type
        self.k = kwargs.get('k')
        self.params = kwargs

    def fit(self, load_N=True, save_N=False, debug=False):

        alg = 'ls'

        if self.solver_type == 'low_precision':
            x, time = comp_sketch(self.matrix_Ab, 'x', load_N, save_N, alg, **self.params)
            time = time/self.k
            x = [a[1] for a in x]
        elif self.solver_type == "sampling_unif":
            s = self.params.get('s')
            t = time.time()
            x = unif_sample_solve(self.matrix_Ab.matrix,self.matrix_Ab.m,self.k,s)
            time = time.time() - t
            time = time/self.k
            x = [a[1] for a in x]
        elif self.solver_type == 'high_precision':
            num_iter = self.params.get('num_iter')
            sc = self.params.get('sc')
            sketch_type = self.params.get('sketch_type')

            # start computing a sketch
            if sketch_type is not None:
                N, time_proj = comp_sketch(self.matrix_Ab, 'N', load_N, save_N, alg, **self.params)
            else:
                N = [np.eye(self.matrix_Ab.n)]
                self.k = 1
                time_proj = 0

            b = np.array(self.matrix_Ab.matrix_with_index.map(lambda x: parse_get_key(x,'b')).sortByKey().values().collect())

            # start lsqr
            time = [time_proj for i in range(num_iter)]
            x = []
 
            for i in range(self.k):
                x_iter, y_iter, time_iter = lsqr_spark(self.matrix_Ab.matrix_with_index,b,self.matrix_Ab.m,self.matrix_Ab.n,N[i],sc,1e-10,num_iter)
                x.append(x_iter)
                time = [time[i] + time_iter[i] for i in range(num_iter)]
            
            time = [t/self.k for t in time]
        else:
            raise ValueError("invalid solver_type")

        self.x = x
        self.time = time

        if debug:
            x_opt, f_opt = self.__ideal_cost(A, b)

    def __ideal_cost(self, A, b):
        x_opt = npl.lstsq(A, b)[0]
        f_opt = npl.norm(np.dot(A, x_opt)-b)
        return x_opt, f_opt

    def __comp_cost(self,x,b,stack_type,repnum):
        if stack_type == 1:
            costs = [ comp_l2_obj(self.matrix_Ab.matrix_original,np.array(p)) for p in x ]
        elif stack_type == 2: 
            n = self.matrix_Ab.n
            a = [ repnum*comp_l2_obj(self.matrix_Ab.matrix_original,p)**2 - (repnum-1)*npl.norm( p[n/2:] - b[-n/2:])**2 for p in x ]
            costs = [ np.sqrt(aa)/np.sqrt(repnum) for aa in a ]

        return costs

    def comp_relerr(self,b,x_opt,f_opt):
        x_relerr = []
        f_relerr = []

        if self.solver_type == 'high_precision':
            costs = [self.__comp_cost(x,b,self.matrix_Ab.stack_type,self.matrix_Ab.repnum) for x in self.x]
            f_relerr = [ (np.abs(f_opt- np.array(c))/f_opt).tolist() for c in costs ]
            x_relerr = [ [ npl.norm( p - x_opt ) / npl.norm(x_opt) for p in self.x[i] ] for i in range(self.k) ] # a list of list. each element is a list with length self.iter

            f_final = [ p[-1] for p in f_relerr ]
            idx = np.where(f_final == np.median(f_final))[0]
            if len(idx) > 1:
                idx = idx[0]
            self.x_relerr_median = x_relerr[idx]
            self.f_relerr_median = f_relerr[idx]
        else:
            costs = self.__comp_cost(self.x,b,self.matrix_Ab.stack_type,self.matrix_Ab.repnum)
            f_relerr = [ np.abs(f_opt - c)/f_opt for c in costs ]
            x_relerr = [ npl.norm(p - x_opt)/npl.norm(x_opt) for p in self.x ]

            self.x_relerr_median = np.median(x_relerr)
            self.f_relerr_median = np.median(f_relerr)

        return self.x_relerr_median, self.f_relerr_median

