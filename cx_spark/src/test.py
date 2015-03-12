import numpy as np
from pyspark import SparkContext
from pyspark import SparkConf
from cx import *
from rowmatrix import *
from utils import *

logs_dire = '/global/u2/m/msingh/sc_paper/new_version/sc-2015/logs'
conf = SparkConf().set('spark.eventLog.enabled','true').set('spark.eventLog.dir',logs_dire).set("spark.driver.maxResultSize", "2g") 
sc = SparkContext(appName="cx_exp",conf=conf)
data = sc.textFile("/global/u2/m/msingh/sc_paper/new_version/sc-2015/output/out")
#rdd = data.map(lambda x:x.split(',')).map(lambda x:(int(x[0]),(int(x[1]),float(x[2]))))
rdd = data.map(lambda x:x.split(',')).map(lambda x:(int(x[1]),(int(x[0]),float(x[2]))))
#gprdd = rdd.map(lambda x:list(x))
gprdd = rdd.groupByKey().map(lambda x :(x[0],list(x[1])))
row_shape = rdd.map(lambda x:x[0]).max()+1
column_shape = rdd.map(lambda x:x[1][0]).max() +1
#print gprdd.take(1)
#[(256, [(7247424, 4.0)....(index, value)]

def indexed(l):
	indexed, values = [],[]
	for tup in l:
		indexed.append(tup[0])
		values.append(tup[1])
	return indexed, np.array(values)

flattening = gprdd.map(lambda x: (x[0],indexed(x[1])))
def densify(indices, values):
	vector = np.zeros(column_shape)
	vector[indices] = values
	return vector

densed_rdd = flattening.map(lambda x:(x[0],densify(x[1][0],x[1][1]))).map(lambda x:x[1])
print densed_rdd.take(1)
matrix_A = RowMatrix(densed_rdd,'output',row_shape,column_shape ,True)
cx = CX(matrix_A,sc=sc)
k = 5
q = 3
lev, p = cx.get_lev(k,axis=0, q=q) 

print "lev score ",lev
print "p is ",p