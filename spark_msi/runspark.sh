#!/bin/bash -l
start-all.sh
spark-submit --conf spark.eventLog.enabled=true --conf spark.eventLog.dir=$SPARK_EVENTLOG_DIR --master $SPARKURL --executor-memory 32G --driver-memory 32G ./spark_msi.py
stop-all.sh 
