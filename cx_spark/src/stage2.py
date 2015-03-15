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
local_filename = '/Users/msingh/Desktop/research/out'
edison_filename = "/global/u2/m/msingh/sc_paper/new_version/sc-2015/output/column_out"
data = sc.textFile(edison_filename)





rdd = data.map(lambda x:x.split(',')).map(lambda x:(int(x[0]),int(x[1]),float(x[2])))
#rdd = data.map(lambda x:x.split(',')).map(lambda x:(int(x[1]),int(x[0]),float(x[2])))

def zipandsave(rdd,index, output=None):
    ids = rdd.map(lambda x:x[index]).distinct()
    ids_zipped = ids.zipWithIndex()  
    if output:
        ids_zipped.map(lambda x:str(x[0])+','+str(x[1])).saveAsTextFile(output)
    return ids_zipped
edison_rows_mapping_output='/global/u2/m/msingh/sc_paper/new_version/sc-2015/output/row_mappings'
#local_rows_mapping_output='/Users/msingh/Desktop/research/row_mappings'
rows_zipped = zipandsave(rdd,0, edison_rows_mapping_output)
def annotate_and_group(zipped_rdd,rdd,index):
    if index == 1:
        annotated_zipped = zipped_rdd.map(lambda x:(x[1],x[0],'zip'))
    else:
        annotated_zipped = zipped_rdd.map(lambda x:(x[0],x[1],'zip'))
    unioned = rdd.union(annotated_zipped)
    grouped = unioned.map(lambda x:(x[index],(x))).groupByKey().map(lambda x:(x[0],list(x[1])))
    return grouped
rows_grouped = annotate_and_group(rows_zipped, rdd,index=0)

def replace_rows(long_list):
    replace_id = None
    for tuple in long_list:
        _, other, val = tuple
        if val =='zip':
            replace_id = other
            break
    new_list  = []
    for tuple in long_list:
        if tuple[2]!='zip':        
            construct_tup = (replace_id, tuple[1], tuple[2])          
            new_list.append(construct_tup)
    return new_list

def replace_column(long_list):
    replace_id = None
    for tuple in long_list:
        other, _, val = tuple
        if val =='zip':
            replace_id = other
            break
    new_list  = []
    for tuple in long_list:
        if tuple[2]!='zip':
            construct_tup = (tuple[0], replace_id, tuple[2])          
            new_list.append(construct_tup)
    return new_list

rows_assigned = rows_grouped.map(lambda x:replace_rows(x[1])).flatMap(lambda x:x)
edison_columns_mapping_output='/global/u2/m/msingh/sc_paper/new_version/sc-2015/output/column_mappings'
#local_columns_mapping_output='/Users/msingh/Desktop/research/column_mappings'
columns_zipped = zipandsave(rows_assigned, 1,edison_columns_mapping_output)
columns_grouped = annotate_and_group(columns_zipped, rows_assigned, index =1)
columns_assigned = columns_grouped.map(lambda x:replace_column(x[1])).flatMap(lambda x:x)
columns_assigned.map(lambda x:  str(x[0]) +','+str(x[1])+','+str(x[2]) ).saveAsTextFile("/global/u2/m/msingh/sc_paper/new_version/sc-2015/output/new_matrix")
