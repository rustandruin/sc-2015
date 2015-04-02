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
logger = logging.getLogger(__name__)

logger.info("job_cx starting")
print "job_cx making sc"
sc = SparkContext()
prefix = 'hdfs:///'
metapath = 'Lewis_Dalisay_Peltatum_20131115_hexandrum_1_1-masked.rdd'
rddpath = prefix + 'Lewis_Dalisay_Peltatum_20131115_hexandrum_1_1-masked-100x100.rdd'
#matpath = prefix + 'Lewis_Dalisay_Peltatum_20131115_hexandrum_1_1-masked.mat'
print "job_cx loading RDD from %s" % rddpath
dataset = MSIDataset.load(sc, metapath, rddpath)#.cache()
dataset.spectra = dataset.spectra.coalesce(256)
msimat = MSIMatrix.from_dataset(sc, dataset)
print "done loading"
print "shape:", msimat.shape
mat = prepare_matrix(msimat.nonzeros).cache()
#mat = prepare_matrix(msimat.nonzeros.map(lambda (i,j,v): (j,i,v))).cache()
mat = SparseRowMatrix(mat, "msimat", msimat.shape[0], msimat.shape[1], cache=False)
print "job_cx entering cx"
cx = CX(mat)
print "job_cx done cx"
k = 2
q = 2
r = 20
lev, p = cx.get_lev(k,axis=0, q=q)
idx = cx.comp_idx('randomized', r)
rows = cx.get_rows()
with open('dump.pkl', 'w') as outf:
  import cPickle as pickle
  data = {
      'lev': lev,
      'idx': idx,
      'rows': rows
  }
  pickle.dump(data, outf)
