import numpy as np
from spark_msi import MSIDataset
from pyspark import SparkContext
from pyspark import SparkConf
from spark_msi import converter

ROOT="/project/projectdirs/openmsi/projects/mantissa/ddalisay/2014Nov15_PDX_IMS_imzML/"
imzXMLPath = ROOT+'Lewis_Dalisay_Peltatum_20131115_PDX_Std_1.imzml'                                                       
imzBinPath = ROOT+"Lewis_Dalisay_Peltatum_20131115_PDX_Std_1.ibd" 
logs_dire = '/global/u2/m/msingh/sc_paper/new_version/sc-2015/logs'
conf = SparkConf().set('spark.eventLog.enabled','true').set('spark.eventLog.dir',logs_dire)
sc = SparkContext(appName="cx_exp",conf=conf)
outpath='/global/u2/m/msingh/sc_paper/new_version/sc-2015/output'
rdd = converter(sc, imzXMLPath, imzBinPath, outpath)
print "shape ", rdd.shape