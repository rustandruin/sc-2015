spark-submit --verbose \
  --driver-memory 32G \
  --conf spark.driver.maxResultSize=4g \
  --conf spark.task.maxFailures=1 \
  target/scala-2.10/heromsi-assembly-0.0.1.jar \
  csv hdfs:///Lewis_Dalisay_Peltatum_20131115_hexandrum_1_1-masked.mat.csv \
  8258911 131048 \
  cx-out.json \
  16 2 1

# notes
#  idxrow hdfs:///Lewis_Dalisay_Peltatum_20131115_hexandrum_1_1-masked.rowmat \
