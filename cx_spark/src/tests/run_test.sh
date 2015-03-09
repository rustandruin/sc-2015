spark-submit --driver-java-options '-Dlog4j.configuration=log4j.properties' --executor-memory 7G --driver-memory 8G --py-files ../rma_utils.py,../utils.py,../rowmatrix.py test_cx.py 
