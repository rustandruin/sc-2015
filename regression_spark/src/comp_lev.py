from projections import *
from rma_utils import *
from matrix import *
from lsqr_spark import *
from comp_sketch import *
import time
import logging

logger = logging.getLogger(__name__)

class CompLev:
    def __init__(self, matrix, **kwargs):
        self.matrix = matrix
        self.k = kwargs.get('k')
        self.s = kwargs.get('s')
        self.N_exact = kwargs.get('N_exact')
        self.sumLev_exact = kwargs.get('sumLev_exact')
        self.params = kwargs
        #self.matrix_Ab = Matrix(self.matrix_A.matrix.zip(self.matrix_b.matrix).map(lambda x: x[0]+[x[1]]),self.matrix_A.m,self.matrix_A.n+1)

    def execute(self, sc, p_norm):
        self.N, self.time = comp_sketch(self.matrix, 'N', fast=True, **self.params)

        #t = time.time()
    	#self.projection = Projections(**self.params)
        #self.N = self.projection.execute(self.matrix, 'svd')
        #self.time = time.time() - t

        self.sumLev = comp_lev(self.matrix.matrix,sc,self.N)
        p_err = comp_p_err(self.matrix.matrix, sc, self.N, self.N_exact, self.sumLev, self.sumLev_exact)
        self.time = self.time/self.k
        self.lev_relerr_median = np.sqrt(np.median(p_err)/p_norm**2)

        return self.lev_relerr_median, self.time

    def samp_solve(self, svec, sc, A, b, x_opt, f_opt):

        self.x_relerr_median = []
        self.f_relerr_median = []
        #sumLev = comp_lev(self.matrix.matrix,sc,self.N)
        for s in svec:
            x = sample_solve(self.matrix.matrix,sc,self.N,self.sumLev,s)
            self.x = [a[1] for a in x]

            costs = [ npl.norm(np.dot(A, p) - b) for p in self.x ]
            f_relerr = [ np.abs(f_opt - c)/f_opt for c in costs ]
            x_relerr = [ npl.norm(p - x_opt)/npl.norm(x_opt) for p in self.x ]

            self.x_relerr_median.append( np.median(x_relerr) )
            self.f_relerr_median.append( np.median(f_relerr) )

        return self.x_relerr_median, self.f_relerr_median
