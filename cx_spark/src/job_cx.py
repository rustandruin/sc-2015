import sys
print >> sys.stderr, "job_cx starting"
from pyspark import SparkContext
from spark_msi import MSIDataset, MSIMatrix
from sparse_row_matrix import SparseRowMatrix
from cx import CX
from utils import prepare_matrix
import os
import logging.config
logging.config.fileConfig('logging.conf',disable_existing_loggers=False)

print "job_cx making sc"
sc = SparkContext()
inpath = '/scratch1/scratchdirs/jeyk/data/'
#outpath = os.path.join(os.getenv('SCRATCH'), 'dev-data')
name='Lewis_Dalisay_Peltatum_20131115_hexandrum_1_1-masked'
rddpath = os.path.join(inpath, name, name + ".rdd")
print "job_cx loading RDD from %s" % rddpath
dataset = MSIDataset.load(sc, rddpath, minPartitions=32768).cache()
msimat = MSIMatrix(dataset)
mat = prepare_matrix(msimat.nonzeros).cache()
mat = SparseRowMatrix(mat, "msimat", msimat.shape[0], msimat.shape[1])
print "job_cx entering cx"
cx = CX(mat)
print "job_cx done cx"
k = 8
q = 3
lev, p = cx.get_lev(k,axis=0, q=q)
print "lev:"
print lev
print "p:"
print p
