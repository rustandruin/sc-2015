pyfiles=`find $PWD -name \*.py | paste -sd , -`
spark-submit \
    --verbose \
    --conf spark.ui.showConsoleProgress=false \
    --conf spark.eventLog.enabled=true  \
    --conf spark.eventLog.dir=$SPARK_EVENTLOG_DIR  \
    --py-files $pyfiles \
    job_cx.py \
    ;
