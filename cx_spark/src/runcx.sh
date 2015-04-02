pyfiles=`find $PWD -name \*.py | paste -sd , -`
export PYSPARK_PYTHON=python27
#    --conf spark.shuffle.blockTransferService=nio \
spark-submit \
    --verbose \
    --driver-memory 107353m \
    --conf spark.ui.showConsoleProgress=false \
    --conf spark.eventLog.enabled=true  \
    --conf spark.eventLog.dir=logs/events  \
    --py-files $pyfiles \
    job_cx.py \
    ;
