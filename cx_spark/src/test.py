import numpy as np
from pyspark import SparkContext
from pyspark import SparkConf
from cx import *
from rowmatrix import *
from utils import *
import time
logs_dire = '/global/u2/m/msingh/sc_paper/new_version/sc-2015/logs'
conf = SparkConf().set('spark.eventLog.enabled','true').set('spark.eventLog.dir',logs_dire).set("spark.driver.maxResultSize", "2g") 
sc = SparkContext(appName="cx_exp",conf=conf)
data = sc.textFile("/global/u2/m/msingh/sc_paper/new_version/sc-2015/output/out")





#rdd = data.map(lambda x:x.split(',')).map(lambda x:(int(x[0]),(int(x[1]),float(x[2]))))
rdd = data.map(lambda x:x.split(',')).map(lambda x:(int(x[1]),int(x[0]),float(x[2])))


row_ids = rdd.map(lambda x:x[0]).distinct()
rows_zipped = row_ids.zipWithIndex()
rows_zipped.map(lambda x:str(x[0])+','+str(x[1])).saveAsTextFile("/global/u2/m/msingh/sc_paper/new_version/sc-2015/output/mappings")

annotated_zipped = rows_zipped.map(lambda x:(x[0],x[1],'zip'))
unioned = rdd.union(annotated_zipped)
rows_grouped = unioned.map(lambda x:(x[0],(x))).groupByKey().map(lambda x:(x[0],list(x[1])))
def replace(long_list,index):
    replace_id = None
    for tuple in long_list:
        _, other, val = tuple
        if val =='zip':
            replace_id = other
            break
    new_list  = []
    for tuple in long_list:
        if tuple[2]!='zip':
            if index == 0:
                construct_tup = (replace_id, tuple[1], tuple[2])
            else:
                construct_tup = (tuple[0], replace_id, tuple[2])
            new_list.append(construct_tup)
    return new_list
rows_assinged = rows_grouped.map(lambda x:replace(x[1],0)).flatMap(lambda x:x)
rows_assinged.map(lambda x:  str(x[0]) +','+str(x[1])+','+str(x[2]) ).saveAsTextFile("/global/u2/m/msingh/sc_paper/new_version/sc-2015/output/new_matrix")

#gprdd = rdd.map(lambda x:list(x))
gprdd = rows_assinged.map(lambda x:(x[0],(x[1],x[2]))).groupByKey().map(lambda x :(x[0],list(x[1])))
row_shape = rows_assinged.map(lambda x:x[0]).max()+1
column_shape = rows_assinged.map(lambda x:x[1]).max() +1
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
start = time.time()
cx = CX(matrix_A,sc=sc)
k = 5
q = 3
lev, p = cx.get_lev(k,axis=0, q=q) 
end = time.time()
print "lev score ",lev, len(lev)
print "p is ",p, len(p)
print "time ", end -start