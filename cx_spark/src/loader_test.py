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
print "filling matrix"
print "row count ", row_count
print "column_count ", column_count
print "min row ", min_rc
print "min c ", mint_cc
num_rows = 10000
#half_rdd = non_zer.filter(lambda x:x[1]<num_rows)
#maxi =  half_rdd.map(lambda x:x[1]).max()
#print maxi
rdd = non_zer.map(lambda x:  str(x[0]) +','+str(x[1])+','+str(x[2]) )
rdd.saveAsTextFile("/global/u2/m/msingh/sc_paper/new_version/sc-2015/output/out")
grouped = non_zer.map(lambda x:(x[0],(x[1],x[2]))).groupByKey()
print grouped.take(2)
#mint_cc = half_rdd.map(lambda x:x[1]).min()
#sliced_matrix = non_zer.map(lambda x:(x[0], [x[1],x[2]]))



sys.exit()
def func(v,l):
	l += v
	return l

broadcasted = sc.broadcast(sliced_matrix.collect())
check_rdd = sc.parallelize([0]).map(lambda x:broadcasted.value)
print "take 2 ", check_rdd.take(2)
sys.exit()
mapped_rdd = sliced_matrix.reduceByKey(lambda x,l=[]: func(list(x),l))
print "two rows ", mapped_rdd.take(2)

grpd = sliced_matrix.groupByKey()
print "two rows " , grpd.take(2)
#grouped = non_zer.groupByKey()

matrix = lil_matrix((row_count+1,column_count+1), dtype=np.float) #((row_count, column_count), dtype=np.float)
local_data = non_zer.collect()
print "filling matrix"
def fill_matrix(intuple):
	row_index, column_index, value = intuple
	#print intuple
	matrix[row_index, column_index] = value
	return matix[row_index].toarray()

matrix_rdd = non_zer.map(lambda x: fill_matrix(x))
twos =  matrix_rdd.take(2)
print twos
print matrix[0]
sys.exit()
#for tup in local_data:
#	fill_matrix(tup)
#matrix_rdd = non_zer.map(lambda x: fill_matrix(x))
#for tup in matrix:
#	row, col, value = tup
#	matrix[row, col] = value


#print matrix_rdd
#print "full matrix ",matrix[1:20,:].toarray()

"""
def fill_matrix(intuple):
	row_index, column_index, value = intuple
	matrix[row_index, column_index] = value
	#print "row index ", row_index, matrix.shape
	return matrix[row_index].toarray()[0]
#matrix = np.zeros((int(row_count), int(column_count)))

matrix = lil_matrix((row_count,column_count), dtype=np.float)#((row_count, column_count), dtype=np.float)
transposed_matrix = matrix.transpose()
print "matrix filled"
def fill_matrix(intuple):
	row_index, column_index, value = intuple
	matrix[row_index, column_index] = value
	print "row index ", row_index
	return matrix[row_index].toarray()[0]

def fill_matrix_transpose(intuple):
	row_index, column_index, value = intuple
	transposed_matrix[column_index, row_index] = value
	print "row index ", row_index
	return transposed_matrix[row_index].toarray()[0]

matrix_rdd = non_zer.map(lambda x: fill_matrix_transpose(x))
matrix_rdd1 = matrix_rdd.map(lambda x:list(x)).map(lambda x:x[:100000])
m,n = transposed_matrix.shape
print "clean up the proxy matrix"
print "shapes ", m, n
cou = matrix_rdd1.count()
print cou
#del matrix
"""
matrix_rdd = non_zer.map(lambda x: fill_matrix(x))
matrix_A = RowMatrix(matrix_rdd,'output',row_count,column_count ,True)
print "matrix read"
cx = CX(matrix_A,sc=sc)
k = 5
q = 3
lev, p = cx.get_lev(k,axis=0, q=q) 

print "lev score ",lev
print "p is ",p



