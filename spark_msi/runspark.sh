#!/bin/bash -l

start-all.sh
spark-submit --master $SPARKURL --executor-memory 32G --driver-memory 32G ./spark_msi.py
stop-all.sh 
