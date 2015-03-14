import sys
sys.path.append('../src/')
import numpy as np
from scipy.sparse import csr_matrix
import unittest
from rowmatrix import RowMatrix
from sparse_row_matrix import SparseRowMatrix
from rma_utils import to_sparse
from cx import CX

class SparseRowMatrixTestCase(unittest.TestCase):
    def setUp(self):
        self.matrix_A = SparseRowMatrix(sparse_matrix_rdd,'test_data',1000,100)

    def test_size(self):
        c = self.matrix_A.rdd.count()
        self.assertEqual(c, 1000)

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

class ComputeLeverageScoressSparseTestCase(unittest.TestCase):
    def setUp(self):
        self.matrix_A = SparseRowMatrix(sparse_matrix_rdd,'test_data',1000,100)

    def test_col_lev(self):
        cx = CX(self.matrix_A)
        lev, p = cx.get_lev(5, q=5)
        self.assertEqual(len(lev), 100)

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
suite_list.append( loader.loadTestsFromTestCase(ComputeLeverageScoressSparseTestCase) )
suite_list.append( loader.loadTestsFromTestCase(MatrixMultiplicationTestCase) )
suite_list.append( loader.loadTestsFromTestCase(ComputeLeverageScoresTestCase) )
suite = unittest.TestSuite(suite_list)

if __name__ == '__main__':
    from pyspark import SparkContext

    sc = SparkContext(appName="cx_test_exp")
    A = np.loadtxt('../data/unif_bad_1000_100.txt')
    A2 = np.loadtxt('../data/unif_bad_100_1000.txt')
    matrix_rdd = sc.parallelize(A.tolist(),140)
    matrix_rdd2 = sc.parallelize(A2.tolist(),50)

    sA = to_sparse(A)
    sparse_matrix_rdd = sc.parallelize(sA,140)

    runner = unittest.TextTestRunner(stream=sys.stderr, descriptions=True, verbosity=1)
    runner.run(suite)

