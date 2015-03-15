import time

import numpy as np
from pyspark import SparkContext
from pyspark import SparkConf

from cx import *
from rowmatrix import *
from sparse_row_matrix import SparseRowMatrix
from utils import *

logs_dire = '/global/u2/m/msingh/sc_paper/new_version/sc-2015/logs'
conf = SparkConf().set('spark.eventLog.enabled','true').set('spark.eventLog.dir',logs_dire).set("spark.driver.maxResultSize", "8g") 
sc = SparkContext(appName="cx_exp",conf=conf)
rows_assigned = sc.textFile("/global/u2/m/msingh/sc_paper/new_version/sc-2015/output/new_matrix").map(lambda x:x.split(',')).map(lambda x:(int(x[0]),int(x[1]),float(x[2])))
row_shape = rows_assigned.map(lambda x:x[0]).max() +1 
column_shape = rows_assigned.map(lambda x:x[1]).max() +1

#gprdd = rdd.map(lambda x:list(x))
"""
gprdd = rows_assigned.map(lambda x:(x[0],(x[1],x[2]))).groupByKey().map(lambda x :(x[0],list(x[1])))
row_shape = rows_assigned.map(lambda x:x[0]).max() +1 
column_shape = rows_assigned.map(lambda x:x[1]).max() +1

def indexed(l):
    indexed, values = [],[]
    for tup in l:
        indexed.append(tup[0])
        values.append(tup[1])
    return np.array(indexed), np.array(values)

flattening = gprdd.map(lambda x: (x[0],indexed(x[1])))
def densify(indices, values):	
    vector = np.zeros(column_shape)
    vector[indices] = values
    return vector

#densed_rdd = flattening.map(lambda x:(x[0],densify(x[1][0],x[1][1]))).map(lambda x:x[1])
#print densed_rdd.take(1)
print "flattening ", flattening.take(1)
sorted_rdd = flattening.sortByKey()
"""
matrix_A = SparseRowMatrix(rows_assigned,'output',row_shape,column_shape ,True)
start = time.time()
cx = CX(matrix_A)
k = 5
q = 3
lev, p = cx.get_lev(k,axis=0, q=q) 
end = time.time()
np.savetxt("/global/u2/m/msingh/sc_paper/new_version/sc-2015/output/leverage_scores_test", np.array(lev))
np.savetxt("/global/u2/m/msingh/sc_paper/new_version/sc-2015/output/p_scores_test", np.array(p))
print "lev score ",lev, len(lev)
print "p is ",p, len(p)
print "time ", end -start