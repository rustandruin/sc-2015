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

def transform_tomz(x, mz_broadcast, tlen):
    mz_index = x/tlen
    mz_val = mz_broadcast.value[mz_index]
    return mz_val    

def get_t(x, tlen): 
    return x % tlen

def get_mz(x, tlen):
    return x / tlen    

def run_stage4(params_dict):
    logs_dir = params_dict.get('logsdir')
    column_leverage_score = params_dict.get('leveragescores')
    column_mappings = params_dict.get('columnmappingsfile')
    raw_rdd = params_dict.get('raw_rdd')
    mz_output = params_dict.get('mzvals')
    conf = SparkConf().set('spark.eventLog.enabled', 'true').set('spark.eventLog.dir', logs_dir).set('spark.driver.maxResultSize', 'g') 
    sc = SparkContext(appName='post process', conf=conf)
    column_leverage_scores = sc.textFile(column_leverage_score).map(lambda x: float(str(x)))
    zipped = column_leverage_scores.zipWithIndex().map(lambda x:(x[1],x[0]))
    
    mappings = sc.textFile(column_mappings).map(lambda x: map(int ,x.split(','))).map(lambda x:(x[1],x[0],'zip'))
    unioned = zipped.union(mappings)

    rows_grouped = unioned.map(lambda x:(x[0], (x))).groupByKey().map(lambda x:(x[0], list(x[1])))
    new_ids = rows_grouped.map(lambda x: replace(x[1]))


    data = MSIDataset.load(sc, raw_rdd)
    mz_axis = data.mz_axis
    mz_broadcast = sc.broadcast(mz_axis)

    xlen,ylen,tlen,mzlen = data.shape
  
    get_t_mz = new_ids.map(lambda x: (get_t(x[0], tlen),get_mz(x[0], tlen), transform_tomz(x[0],mz_broadcast, tlen),  x[0],x[1]))
    sorted_vals = get_t_mz.sortBy(lambda x:x[4], ascending=False)
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
    print 'run stage 4 with params ', stage4_params
   
    run_stage4(stage4_params)
    print 'run finished'




