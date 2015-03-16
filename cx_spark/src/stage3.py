import time

import numpy as np
from pyspark import SparkContext
from pyspark import SparkConf

from cx import CX
from parse_config import load_configuration
from sparse_row_matrix import SparseRowMatrix


def run_stage3(params_dict):
    logs_dir = params_dict.get('logsdir')
    input_matrix = params_dict.get('inputmatrix')
    leverage_scores_file = params_dict.get('leveragescores')
    p_score_file = params_dict.get('pscores')
 
    conf = SparkConf().set('spark.eventLog.enabled','true').set('spark.eventLog.dir',logs_dir).set('spark.driver.maxResultSize', '8g') 
    sc = SparkContext(appName='cx_exp',conf=conf)
    rows_assigned = sc.textFile(input_matrix).map(lambda x:x.split(',')).map(lambda x:(int(x[0]), int(x[1]), float(x[2])))
    row_shape = rows_assigned.map(lambda x:x[0]).max() + 1 
    column_shape = rows_assigned.map(lambda x:x[1]).max() + 1

    matrix_A = SparseRowMatrix(rows_assigned,'output', row_shape,column_shape, True)
    start = time.time()
    cx = CX(matrix_A)
    k = 5
    q = 3
    lev, p = cx.get_lev(k,axis=0, q=q) 
    end = time.time()
    np.savetxt(leverage_scores_file, np.array(lev))
    np.savetxt(p_score_file, np.array(p))
    print 'lev score ', lev, len(lev)
    print 'p is ', p, len(p)
    print 'time ', end-start

if __name__ == '__main__':
    config_params = load_configuration()
    stage_3_params = config_params['STAGE3']
    global_param = config_params['GLOBAL']
    stage_3_params.update(global_param)
    stage_3_params['inputmatrix'] = config_params['STAGE2']['mappedmatrix']
    print 'run stage 3 with params ', stage_3_params
   
    run_stage3(stage_3_params)
    print 'run finished'