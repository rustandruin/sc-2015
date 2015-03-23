#!/bin/bash -l
set -o errexit
set -o pipefail
echo "starting job"

remote_exec() {
    host="$1"
    shift
    cmd="$*"
    if [ -z "$host" -o "$host" = "localhost" ]; then
        /bin/bash -c "$cmd"
    else
        ssh "$host" "$cmd"
    fi
}

die() {
    msg="$1"
    echo "$msg" >&2
    exit 1
}

start_collectl() {
    host="$1"
    [ ! -z "$PERFLOG_DIR" ] || die "PERFLOG_DIR not set"
    outpath="$PERFLOG_DIR/collectl-$host"
    echo "starting collectl on $host"
    remote_exec "$host" collectl \
        --daemon \
        --align \
        --filename "$outpath" \
        --flush 0 \
        --interval 1:5 \
        --subsys sbcdfijmnstZ \
        --procfilt "U$USER" \
        --procopts ctw \
        ;
}

start_collectl localhost
for host in `cat ${SPARK_SLAVES}`; do
    start_collectl "$host"
done

start-all.sh
spark-submit \
    --conf pbs.jobId=$PBS_JOBID \
    --conf spark.eventLog.enabled=true  \
    --conf spark.eventLog.dir=$SPARK_EVENTLOG_DIR  \
    --master $SPARKURL  \
    --executor-memory 32G  \
    --driver-memory 32G ./spark_msi.py \
    ;
stop-all.sh 
