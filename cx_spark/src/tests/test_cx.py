import sys
sys.path.append('../')
import numpy as np
import unittest
from rowmatrix import RowMatrix
from cx import CX

class MatrixMultiplicationTestCase(unittest.TestCase):
    def setUp(self):
        self.matrix_A = RowMatrix(matrix_rdd,'test_data',1000,100)
        self.matrix_A2 = RowMatrix(matrix_rdd2,'test_data',100,1000)

    def test_mat_rtimes(self):
        mat = np.random.rand(100,50)
        p = self.matrix_A.rtimes(mat, sc)
        p_true = np.dot( A, mat )
        self.assertTrue( np.linalg.norm(p-p_true)/np.linalg.norm(p_true) < 1e-5 )

    def test_mat_ltimes(self):
        mat = np.random.rand(100,1000)
        p = self.matrix_A.ltimes(mat, sc)
        p_true = np.dot( mat,A )
        self.assertTrue( np.linalg.norm(p-p_true)/np.linalg.norm(p_true) < 1e-5 )

    def test_atamat(self):
        mat = np.random.rand(100,20)
        p = self.matrix_A.atamat(mat, sc)
        p_true = np.dot( A.T, np.dot(A, mat) )
        self.assertTrue( np.linalg.norm(p-p_true)/np.linalg.norm(p_true) < 1e-5 )

    def test_mat_rtimes2(self):
        mat = np.random.rand(1000,50)
        p = self.matrix_A2.rtimes(mat, sc)
        p_true = np.dot( A2, mat )
        self.assertTrue( np.linalg.norm(p-p_true)/np.linalg.norm(p_true) < 1e-5 )

    def test_mat_ltimes2(self):
        mat = np.random.rand(50,100)
        p = self.matrix_A2.ltimes(mat, sc)
        p_true = np.dot( mat,A2 )
        self.assertTrue( np.linalg.norm(p-p_true)/np.linalg.norm(p_true) < 1e-5 )

    def test_atamat2(self):
        mat = np.random.rand(1000,20)
        p = self.matrix_A2.atamat(mat, sc)
        p_true = np.dot( A2.T, np.dot(A2, mat) )
        self.assertTrue( np.linalg.norm(p-p_true)/np.linalg.norm(p_true) < 1e-5 )

    def test_mat_rtimes_sub(self):
        mat = np.random.rand(99,50)
        p = self.matrix_A.rtimes(mat, sc, (0,98))
        p_true = np.dot( A[:,:-1], mat )
        self.assertTrue( np.linalg.norm(p-p_true)/np.linalg.norm(p_true) < 1e-5 )

    def test_mat_ltimes_sub(self):
        mat = np.random.rand(100,1000)
        p = self.matrix_A.ltimes(mat, sc, (0,98))
        p_true = np.dot( mat,A[:,:-1] )
        self.assertTrue( np.linalg.norm(p-p_true)/np.linalg.norm(p_true) < 1e-5 )

    def test_atamat_sub(self):
        mat = np.random.rand(99,50)
        p = self.matrix_A.atamat(mat, sc, (0,98))
        p_true = np.dot( A[:,:-1].T, np.dot(A[:,:-1], mat) )
        self.assertTrue( np.linalg.norm(p-p_true)/np.linalg.norm(p_true) < 1e-5 )

class ComputeLeverageScoresTestCase(unittest.TestCase):
    def setUp(self):
        self.matrix_A = RowMatrix(matrix_rdd,'test_data',1000,100)
        self.matrix_A2 = RowMatrix(matrix_rdd2,'test_data',100,1000)
    
    def test_row_lev(self):
        cx = CX(self.matrix_A, sc)
        lev, p = cx.get_lev(5, axis=0, q=5)
        self.assertEqual(len(lev), 1000)

    def test_col_lev(self):
        cx = CX(self.matrix_A, sc)
        lev, p = cx.get_lev(5, axis=1, q=5)
        self.assertEqual(len(lev), 100)

    def test_row_lev2(self):
        cx = CX(self.matrix_A2, sc)
        lev, p = cx.get_lev(5, axis=0, q=5)
        self.assertEqual(len(lev), 100)

    def test_col_lev2(self):
        cx = CX(self.matrix_A2, sc)
        lev, p = cx.get_lev(5, axis=1, q=5)
        self.assertEqual(len(lev), 1000)

loader = unittest.TestLoader()
suite_list = []
suite_list.append( loader.loadTestsFromTestCase(MatrixMultiplicationTestCase) )
suite_list.append( loader.loadTestsFromTestCase(ComputeLeverageScoresTestCase) )
suite = unittest.TestSuite(suite_list)

if __name__ == '__main__':
    from pyspark import SparkContext

    sc = SparkContext(appName="cx_test_exp")
    A = np.loadtxt('small_test_data/unif_bad_1000_100.txt')
    A2 = np.loadtxt('small_test_data/unif_bad_100_1000.txt')
    matrix_rdd = sc.parallelize(A.tolist(),140)
    matrix_rdd2 = sc.parallelize(A2.tolist(),50)

    runner = unittest.TextTestRunner(stream=sys.stderr, descriptions=True, verbosity=1)
    runner.run(suite)

