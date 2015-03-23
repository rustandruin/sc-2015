import sys
sys.path.append('../src/')
import numpy as np
from scipy.sparse import coo_matrix
import unittest
from rowmatrix import RowMatrix
from sparse_row_matrix import SparseRowMatrix
from rma_utils import to_sparse
from cx import CX
import scipy.stats

class SparseRowMatrixTestCase(unittest.TestCase):
    def setUp(self):
        self.matrix_A = SparseRowMatrix(sparse_matrix_rdd,'test_data',1000,100)
        self.matrix_A2 = SparseRowMatrix(sparse_matrix_rdd2,'test_data',100,1000)

    def test_size(self):
        c = self.matrix_A.rdd.count()
        self.assertEqual(c, 1000)

    def test_size2(self):
        c = self.matrix_A2.rdd.count()
        self.assertEqual(c, 100)

    def test_mat_ltimes(self):
        mat = np.random.rand(10,1000)
        p = self.matrix_A.ltimes(mat)
        p_true = np.dot( mat,A )
        self.assertTrue( np.linalg.norm(p-p_true)/np.linalg.norm(p_true) < 1e-5 )

    def test_atamat(self):
        mat = np.random.rand(100,20)
        p = self.matrix_A.atamat(mat)
        p_true = np.dot( A.T, np.dot(A, mat) )
        self.assertTrue( np.linalg.norm(p-p_true)/np.linalg.norm(p_true) < 1e-5 )

    def test_mat_ltimes2(self):
        mat = np.random.rand(10,100)
        p = self.matrix_A2.ltimes(mat)
        p_true = np.dot( mat,A2 )
        self.assertTrue( np.linalg.norm(p-p_true)/np.linalg.norm(p_true) < 1e-5 )

    def test_atamat2(self):
        mat = np.random.rand(1000,20)
        p = self.matrix_A2.atamat(mat)
        p_true = np.dot( A2.T, np.dot(A2, mat) )
        self.assertTrue( np.linalg.norm(p-p_true)/np.linalg.norm(p_true) < 1e-5 )

class ComputeLeverageScoresSparseTestCase(unittest.TestCase):
    def setUp(self):
        self.matrix_A = SparseRowMatrix(sparse_matrix_rdd,'test_data',1000,100)
        self.matrix_A2 = SparseRowMatrix(sparse_matrix_rdd2,'test_data',100,1000)

    def test_col_lev(self):
        cx = CX(self.matrix_A)
        lev, p = cx.get_lev(5, q=10)
        lev_exact, p_exact = compLevExact(A, 5, axis=1)
        print scipy.stats.entropy(p_exact,p)

        self.assertEqual(len(lev), 100)

    def test_col_lev2(self):
        cx = CX(self.matrix_A2)
        lev, p = cx.get_lev(10, q=10)
        lev_exact, p_exact = compLevExact(A2, 10, axis=1)
        print scipy.stats.entropy(p_exact,p)

        self.assertEqual(len(lev), 1000)

        
class MatrixMultiplicationTestCase(unittest.TestCase):
    def setUp(self):
        self.matrix_A = RowMatrix(matrix_rdd,'test_data',1000,100)
        self.matrix_A2 = RowMatrix(matrix_rdd2,'test_data',100,1000)

    def test_mat_rtimes(self):
        mat = np.random.rand(100,50)
        p = self.matrix_A.rtimes(mat)
        p_true = np.dot( A, mat )
        self.assertTrue( np.linalg.norm(p-p_true)/np.linalg.norm(p_true) < 1e-5 )

    def test_mat_ltimes(self):
        mat = np.random.rand(100,1000)
        p = self.matrix_A.ltimes(mat)
        p_true = np.dot( mat,A )
        self.assertTrue( np.linalg.norm(p-p_true)/np.linalg.norm(p_true) < 1e-5 )

    def test_atamat(self):
        mat = np.random.rand(100,20)
        p = self.matrix_A.atamat(mat)
        p_true = np.dot( A.T, np.dot(A, mat) )
        self.assertTrue( np.linalg.norm(p-p_true)/np.linalg.norm(p_true) < 1e-5 )

    def test_mat_rtimes2(self):
        mat = np.random.rand(1000,50)
        p = self.matrix_A2.rtimes(mat)
        p_true = np.dot( A2, mat )
        self.assertTrue( np.linalg.norm(p-p_true)/np.linalg.norm(p_true) < 1e-5 )

    def test_mat_ltimes2(self):
        mat = np.random.rand(50,100)
        p = self.matrix_A2.ltimes(mat)
        p_true = np.dot( mat,A2 )
        self.assertTrue( np.linalg.norm(p-p_true)/np.linalg.norm(p_true) < 1e-5 )

    def test_atamat2(self):
        mat = np.random.rand(1000,20)
        p = self.matrix_A2.atamat(mat)
        p_true = np.dot( A2.T, np.dot(A2, mat) )
        self.assertTrue( np.linalg.norm(p-p_true)/np.linalg.norm(p_true) < 1e-5 )

    def test_mat_rtimes_sub(self):
        mat = np.random.rand(99,50)
        p = self.matrix_A.rtimes(mat, (0,98))
        p_true = np.dot( A[:,:-1], mat )
        self.assertTrue( np.linalg.norm(p-p_true)/np.linalg.norm(p_true) < 1e-5 )

    def test_mat_ltimes_sub(self):
        mat = np.random.rand(100,1000)
        p = self.matrix_A.ltimes(mat, (0,98))
        p_true = np.dot( mat,A[:,:-1] )
        self.assertTrue( np.linalg.norm(p-p_true)/np.linalg.norm(p_true) < 1e-5 )

    def test_atamat_sub(self):
        mat = np.random.rand(99,50)
        p = self.matrix_A.atamat(mat, (0,98))
        p_true = np.dot( A[:,:-1].T, np.dot(A[:,:-1], mat) )
        self.assertTrue( np.linalg.norm(p-p_true)/np.linalg.norm(p_true) < 1e-5 )

class ComputeLeverageScoresTestCase(unittest.TestCase):
    def setUp(self):
        self.matrix_A = RowMatrix(matrix_rdd,'test_data',1000,100)
        self.matrix_A2 = RowMatrix(matrix_rdd2,'test_data',100,1000)

    def test_col_lev(self):
        cx = CX(self.matrix_A)
        lev, p = cx.get_lev(5, q=5)
        self.assertEqual(len(lev), 100)

    def test_col_lev2(self):
        cx = CX(self.matrix_A2)
        lev, p = cx.get_lev(5, q=5)
        self.assertEqual(len(lev), 1000)

loader = unittest.TestLoader()
suite_list = []
suite_list.append( loader.loadTestsFromTestCase(SparseRowMatrixTestCase) )
suite_list.append( loader.loadTestsFromTestCase(ComputeLeverageScoresSparseTestCase) )
#suite_list.append( loader.loadTestsFromTestCase(MatrixMultiplicationTestCase) )
#suite_list.append( loader.loadTestsFromTestCase(ComputeLeverageScoresTestCase) )
suite = unittest.TestSuite(suite_list)

def to_sparse(A):
    sA = coo_matrix(A)
    return [ (r,c,v) for (r,c,v) in zip(sA.row, sA.col, sA.data) ]

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

if __name__ == '__main__':
    from pyspark import SparkContext

    A = np.loadtxt('../data/unif_bad_1000_100.txt')
    A2 = np.loadtxt('../data/unif_bad_100_1000.txt')
    sA = to_sparse(A)
    sA2 = to_sparse(A2)

    sc = SparkContext(appName="cx_test_exp")

    matrix_rdd = sc.parallelize(A.tolist(),140)
    matrix_rdd2 = sc.parallelize(A2.tolist(),20)
    sparse_matrix_rdd = sc.parallelize(sA,140)  # sparse_matrix_rdd has records in (row,col,val) format
    sparse_matrix_rdd2 = sc.parallelize(sA2,50) 

    runner = unittest.TextTestRunner(stream=sys.stderr, descriptions=True, verbosity=1)
    runner.run(suite)

