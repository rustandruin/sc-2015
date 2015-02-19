'''
This class performs random projections on a given matrix.
Let the input matrix be A. 
parameters:
    projection_type: can be cw, gaussian, rademacher or srdht
    k: number of projections to get (default value: 1)
    c: projection dimension
    output_type: either 'x' or 'N'
'''

from rma_utils import *
from utils import *
import numpy.linalg as npl
import numpy as np

class Projections(object):
    def __init__(self, **kwargs):
        self.projection_type = kwargs.pop('projection_type', 'cw')
        self.k = kwargs.pop('k',1)
        self.c = kwargs.pop('c')
        self.s = kwargs.pop('s', None)
        self.sc = kwargs.pop('sc', None)
        self.__validate()

    def __validate(self):
        if self.projection_type not in Projections.projection_type:
            raise NotImplementedError('%s projection_type not yet implemented' % self.projection_type)
        if not self.c:
            raise ValueError('"c" param is missing')

    def execute(self, matrix, output_type, alg='cx', return_N=False):
        PA = self.__project(matrix)
        if output_type == 'x':
            return PA.map(lambda pa: get_x(pa[1],return_N)).collect()
        elif output_type == 'N':
            return PA.map(lambda pa: get_N(pa[1],alg)).collect()
        else:
            return PA

    def __project(self, matrix, lim='all'):
        c = self.c
        k = self.k
        if self.projection_type == 'cw':
            cwm = CW_Map(self.c, self.k, lim)
            PA = matrix.matrix.mapPartitions(cwm).reduceByKey(add).map(lambda x: (x[0][0],x[1].tolist())).groupByKey()
        elif self.projection_type == 'gaussian':
            gm = Gaussian_Map(self.c, self.k, lim)
            PA = matrix.matrix.mapPartitions(gm).reduceByKey(add).map(lambda x: (x[0][0],x[1].tolist())).groupByKey()
        elif self.projection_type == 'rademacher':
            rm = Rademacher_Map(self.c, self.k, lim)
            PA = matrix.matrix.mapPartitions(rm).reduceByKey(add).map(lambda x: (x[0][0],x[1].tolist())).groupByKey()
        elif self.projection_type == 'srdht':
            seed_s = np.random.randint(10000,size=k)
            srdm = SRDHT_Map(self.c, self.k, matrix.m, seed_s, lim)
            #Note one can use zipWithIndex to substitute .zip(idx)
            matrix.zip_with_index()
            PA = matrix.matrix_with_index.mapPartitions(srdm).reduceByKey(add).map(lambda x: (x[0][0],x[1].tolist())).groupByKey()

        return PA

    projection_type = ['cw','gaussian','rademacher','srdht']
