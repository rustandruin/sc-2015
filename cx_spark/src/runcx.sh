spark-submit --verbose \
  --driver-memory 32G \
  --conf spark.driver.maxResultSize=4g \
  $1 \
  genmat s3n://amp-jey/sc-2015/Lewis_Dalisay_Peltatum_20131115_hexandrum_1_1-masked.mat/Lewis_Dalisay_Peltatum_20131115_hexandrum_1_1-masked.mat.csv \
  hdfs:///test.out

#  genmat hdfs:///Lewis_Dalisay_Peltatum_20131115_hexandrum_1_1-smoothed-mz=437.11407-sd=0.05.rawmat.csv.gz \
# notes
#  idxrow hdfs:///Lewis_Dalisay_Peltatum_20131115_hexandrum_1_1-masked.rowmat \
