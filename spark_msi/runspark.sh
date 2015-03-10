#!/bin/bash -l

start-all.sh
spark-submit --master $SPARKURL --executor-memory 48G --driver-memory 48G ./spark_msi.py
stop-all.sh 
