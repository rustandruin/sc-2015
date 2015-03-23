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
        self.rdd = prepare_matrix(raw_rdd)
        self.name = name
        self.m = m
        self.n = n

        if cache:
            self.rdd.cache()

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

        return pd

    def atamat(self,mat):
        """
        compute A.T*A*B
        """
        # TO-DO: check dimension compatibility
        if mat.ndim == 1:
            mat = mat.reshape((len(mat),1))

        n = self.n
        mat = self.rdd.context.broadcast(mat)

        atamat_mapper = MatrixAtABMapper()
        #b = self.rdd.mapPartitions(lambda records: atamat_mapper(records,mat=mat.value,feats=feats) ).sum()
        b_dict = self.rdd.mapPartitions(lambda records: atamat_mapper(records,mat=mat.value,n=n) ).filter(lambda x: x is not None).reduceByKey(add).collectAsMap()

        order = sorted(b_dict.keys())
        b = []
        for i in order:
            b.append( b_dict[i] )

        b = np.vstack(b)

        return b

    def transpose(self):
        pass

class MatrixLtimesMapper(BlockMapper):

    def __init__(self):
        BlockMapper.__init__(self)
        self.ba = None
        self.data = {'row':[],'col':[],'val':[]}

    def parse(self, r):
        self.keys.append(r[0])
        self.data['row'] += [self.sz]*len(r[1][0])
        self.data['col'] += r[1][0].tolist()
        self.data['val'] += r[1][1].tolist()

    def process(self, mat, n):
        if self.ba:
            #self.ba += ( form_csr_matrix(self.data,len(self.keys),n).T.dot( mat[:,self.keys[0]:(self.keys[-1]+1)].T ) ).T
            self.ba += ( form_csr_matrix(self.data,len(self.keys),n).T.dot( mat[:,self.keys].T ) ).T
        else:
            self.ba = ( form_csr_matrix(self.data,len(self.keys),n).T.dot( mat[:,self.keys].T ) ).T

        return iter([])

    def close(self):
        yield self.ba

class MatrixAtABMapper(BlockMapper):

    def __init__(self):
        BlockMapper.__init__(self)
        self.atamat = None
        self.data = {'row':[],'col':[],'val':[]}

    def parse(self, r):
        self.keys.append(r[0])
        self.data['row'] += [self.sz]*len(r[1][0])
        self.data['col'] += r[1][0].tolist()
        self.data['val'] += r[1][1].tolist()

    def process(self, mat, n):
        data = form_csr_matrix(self.data,len(self.keys),n)
        if self.atamat:
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
