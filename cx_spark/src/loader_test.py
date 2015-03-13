import numpy as np
from spark_msi import MSIDataset, MSIMatrix
from pyspark import SparkContext
from pyspark import SparkConf
from spark_msi import converter
from scipy.sparse import csr_matrix,lil_matrix

from cx import *
from rowmatrix import *
from utils import *
import sys
ROOT="/project/projectdirs/openmsi/projects/mantissa/ddalisay/2014Nov15_PDX_IMS_imzML/"
imzXMLPath = ROOT+'Lewis_Dalisay_Peltatum_20131115_PDX_Std_1.imzml'                                                       
imzBinPath = ROOT+"Lewis_Dalisay_Peltatum_20131115_PDX_Std_1.ibd" 
logs_dire = '/global/u2/m/msingh/sc_paper/new_version/sc-2015/logs'
conf = SparkConf().set('spark.eventLog.enabled','true').set('spark.eventLog.dir',logs_dire).set("spark.driver.maxResultSize", "2g") 
sc = SparkContext(appName="cx_exp",conf=conf)
outpath='/global/u2/m/msingh/sc_paper/new_version/sc-2015/output/data.rdd'
dataset = converter(sc, imzXMLPath, imzBinPath, outpath)
print "shape ", dataset.shape
print dataset
rdd = dataset.spectra
dataset.save(outpath)
data = MSIDataset.load(sc, outpath)
mat = MSIMatrix(data)
non_zer = mat.nonzeros

print "take two ",non_zer.take(2)
print "mat shape ", mat.shape

#[(11, 41455451, 12.0), (11, 46759055, 10.0)]
row_count = non_zer.map(lambda x:x[0]).max()
column_count = non_zer.map(lambda x:x[1]).max()
min_rc = non_zer.map(lambda x:x[0]).min()
mint_cc = non_zer.map(lambda x:x[1]).min()
entries = non_zer.count()


rdd = non_zer.map(lambda x:  str(x[0]) +','+str(x[1])+','+str(x[2]) )
rdd.saveAsTextFile("/global/u2/m/msingh/sc_paper/new_version/sc-2015/output/column_out")



