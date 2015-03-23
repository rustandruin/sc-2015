import logging

import numpy as np

from rma_utils import add_index
from rma_utils import convert_rdd 
from rma_utils import form_csr_matrix
from utils import BlockMapper, add
from utils import prepare_matrix


logger = logging.getLogger(__name__)

class SparseRowMatrix(object):
    """
    A sparse row matrix class
    Each record is of the format: (row_idx, column_id, val)
    """

    def __init__(self, raw_rdd, name, m, n, cache=False):
        self.rdd = raw_rdd#prepare_matrix(raw_rdd)
        self.name = name
        self.m = m
        self.n = n

        if cache:
            self.rdd.cache()

    def gaussian_projection(self, c):
        """
        compute G*A with G is Gaussian matrix with size r by m
        """
        gaussian_projection_mapper = GaussianProjectionMapper()
        n = self.n
        gp = self.rdd.mapPartitions(lambda records: gaussian_projection_mapper(records,n=n,c=c)).filter(lambda x: x is not None).sum()

        return gp

    def ltimes(self, mat):
        """
        compute B*A
        """
        if mat.ndim == 1:
            mat = mat.reshape((1,len(mat)))

        matrix_ltimes_mapper = MatrixLtimesMapper()
        n = self.n
        mat = self.rdd.context.broadcast(mat)
        pd = self.rdd.mapPartitions(lambda records: matrix_ltimes_mapper(records,mat=mat.value,n=n)).filter(lambda x: x is not None).sum()
        mat.unpersist()

        return pd

    def atamat(self,mat):
        """
        compute A.T*A*B
        """
        # TO-DO: check dimension compatibility
        if mat.ndim == 1:
            mat = mat.reshape((len(mat),1))

        [n,c] = mat.shape

        if n*c > 5e3: # the size of mat is too large to broadcast
            b = []
            mini_batch_sz = 5e3/n # make sure that each mini batch has less than 1e8 elements
            start_idx = np.arange(0, c, mini_batch_sz)
            end_idx = np.append(np.arange(mini_batch_sz, c, mini_batch_sz), c)

            for j in range(len(start_idx)):
                print "processing mini batch {0}".format(j)
                b.append(self.__atamat_sub(mat[:,start_idx[j]:end_idx[j]]))
            
            return np.hstack(b)

        else:
            return self.__atamat_sub(mat)

    def __atamat_sub(self,mat):
        mat = self.rdd.context.broadcast(mat)

        n = self.n

        atamat_mapper = MatrixAtABMapper()
        #b = self.rdd.mapPartitions(lambda records: atamat_mapper(records,mat=mat.value,feats=feats) ).sum()
        b_dict = self.rdd.mapPartitions(lambda records: atamat_mapper(records,mat=mat.value,n=n) ).filter(lambda x: x is not None).reduceByKey(add).collectAsMap()

        order = sorted(b_dict.keys())
        b = []
        for i in order:
            b.append( b_dict[i] )

        b = np.vstack(b)

        mat.unpersist()

        return b

    def transpose(self):
        pass

class GaussianProjectionMapper(BlockMapper):

    def __init__(self):
        BlockMapper.__init__(self, 5)
        self.gp = None
        self.data = {'row':[],'col':[],'val':[]}

    def parse(self, r):
        self.keys.append(r[0])
        self.data['row'] += [self.sz]*len(r[1][0])
        self.data['col'] += r[1][0].tolist()
        self.data['val'] += r[1][1].tolist()

    def process(self, n, c):
        sz = len(self.keys)

        if self.gp is not None:
            self.gp += (form_csr_matrix(self.data,sz,n).T.dot(np.random.randn(sz,c))).T
        else:
            self.gp = (form_csr_matrix(self.data,sz,n).T.dot(np.random.randn(sz,c))).T

        return iter([])

    def close(self):
        yield self.gp

class MatrixLtimesMapper(BlockMapper):

    def __init__(self):
        BlockMapper.__init__(self, 5)
        self.ba = None
        self.data = {'row':[],'col':[],'val':[]}

    def parse(self, r):
        self.keys.append(r[0])
        self.data['row'] += [self.sz]*len(r[1][0])
        self.data['col'] += r[1][0].tolist()
        self.data['val'] += r[1][1].tolist()

    def process(self, mat, n):
        if self.ba is not None:
            #self.ba += ( form_csr_matrix(self.data,len(self.keys),n).T.dot( mat[:,self.keys[0]:(self.keys[-1]+1)].T ) ).T
            self.ba += ( form_csr_matrix(self.data,len(self.keys),n).T.dot( mat[:,self.keys].T ) ).T
        else:
            self.ba = ( form_csr_matrix(self.data,len(self.keys),n).T.dot( mat[:,self.keys].T ) ).T

        return iter([])

    def close(self):
        yield self.ba

class MatrixAtABMapper(BlockMapper):

    def __init__(self):
        BlockMapper.__init__(self, 5)
        self.atamat = None
        self.data = {'row':[],'col':[],'val':[]}

    def parse(self, r):
        self.keys.append(r[0])
        self.data['row'] += [self.sz]*len(r[1][0])
        self.data['col'] += r[1][0].tolist()
        self.data['val'] += r[1][1].tolist()

    def process(self, mat, n):
        data = form_csr_matrix(self.data,len(self.keys),n)
        if self.atamat is not None:
            self.atamat += data.T.dot( data.dot(mat) )
        else:
            self.atamat = data.T.dot( data.dot(mat) )
        return iter([])

        #yield np.dot( data.T, np.dot( data, mat ) )

    def close(self):
        #yield self.atamat

        if self.atamat is None:
            yield None
        else:
            block_sz = 50
            m = self.atamat.shape[0]
            start_idx = np.arange(0, m, block_sz)
            end_idx = np.append(np.arange(block_sz, m, block_sz), m)

            for j in range(len(start_idx)):
                yield j, self.atamat[start_idx[j]:end_idx[j],:]
