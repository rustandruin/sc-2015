import time


from pyspark import SparkContext
from pyspark import SparkConf

from parse_config import load_configuration
from spark_msi import MSIDataset
from spark_msi import MSIMatrix
from spark_msi import converter


def replace(values):
        new_id = None
        for val in values:
            if len(val) == 3:
                new_id = val[1]
        for val in values:
            if len(val) == 2:
                return (new_id, val[1])
def transform_tomz(x):
    mz_index = x/tlen
    mz_val = mz_broadcast.value[mz_index]
    return mz_val           
def stage4(params_dict):
    column_leverage_score = params_dict.get('leveragescores')
    column_mappings = params_dict.get('columnmappingsfile')
    raw_rdd = params_dict.get('raw_rdd')
    mz_output = params_dict.get('mzvals')
    #local_column_leverage_scores_file ='/Users/msingh/Desktop/research/column_leverage_scores'
    #edison_column_leverage_scores_file = "/global/u2/m/msingh/sc_paper/new_version/sc-2015/output/column_leverage_scores"
    column_leverage_scores = sc.textFile(column_leverage_score).map(lambda x: float(str(x)))
    zipped = column_leverage_scores.zipWithIndex().map(lambda x:(x[1],x[0]))
    
    #sorted = zipped.sortBy(lambda x:x[0], ascending=False)
    #local_column_mappings = '/Users/msingh/Desktop/research/column_mappings'
    #edison_column_mappings = '/global/u2/m/msingh/sc_paper/new_version/sc-2015/output/column_mappings'
    mappings = sc.textFile(edison_column_mappings).map(lambda x: map(int ,x.split(','))).map(lambda x:(x[1],x[0],'zip'))
    unioned = zipped.union(mappings)


    #(0, [(0, 2.9774697577965027e-07), (0, 45613056, 'zip')])
    

    rows_grouped = unioned.map(lambda x:(x[0],(x))).groupByKey().map(lambda x:(x[0],list(x[1])))
    new_ids = rows_grouped.map(lambda x: replace(x[1]))


    #import sys
    #sys.path.append("/global/u2/m/msingh/sc_paper/new_version/sc-2015/cx_spark/src")
    #from spark_msi import *
    #outpath='/global/u2/m/msingh/sc_paper/new_version/sc-2015/output/data.rdd'
    data = MSIDataset.load(sc, raw_rdd)
    mz_axis = data.mz_axis
    mz_broadcast = sc.broadcast(mz_axis)

    xlen,ylen,tlen,mzlen = data.shape
  

    get_t = lambda x:x%tlen
    get_mz = lambda x:x/tlen
    get_t_mz = new_ids.map(lambda x: (get_t(x[0]),get_mz(x[0]),transform_tomz(x[0]),  x[0],x[1]))
    sorted_val = get_t_mz.sortBy(lambda x:x[4], ascending=False)
    formatted_vals = sorted_vals.map(lambda x: ", ".join(str(i) for i in x))
    formatted_vals.saveAsTextFile(mz_output)

if __name__ == '__main__':
    config_params = load_configuration()
    stage4_params = config_params['STAGE4']
    global_param = config_params['GLOBAL']
    stage4_params.update(global_param)
    stage4_params['leveragescores'] = config_params['STAGE3']['leveragescores']
    stage4_params['columnmappingsfile'] = config_params['STAGE2']['columnmappingsfile']
    stage4_params['raw_rdd'] = config_params['STAGE1']['outpathrdd']
    print "run stage 4 with params ", stage_4_params
   
    run_stage4(stage_4_params)
    print " run finished"




