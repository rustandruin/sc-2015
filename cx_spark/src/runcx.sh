pyfiles=`find $PWD -name \*.py | paste -sd , -`
export PYSPARK_PYTHON=python27
#    --conf spark.shuffle.blockTransferService=nio \
#    --conf spark.eventLog.dir=logs/events  \
spark-submit \
    --verbose \
    --driver-memory 107353m \
    --conf spark.cleaner.referenceTracking=true \
    --conf spark.cleaner.referenceTracking.blocking=true \
    --conf spark.cleaner.referenceTracking.blocking.shuffle=true \
    --conf spark.eventLog.enabled=true  \
    --conf spark.ui.showConsoleProgress=false \
    --conf log4j.configuration=file://$PWD/log4j.properties \
    --py-files $pyfiles \
    job_cx.py \
    ;
