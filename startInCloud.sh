#!/bin/bash
if [ "$1" != "" ]; then
    export BUCKET_NAME=aerfio-bucket
    export JOB_NAME="$1_$(date '+%Y_%m_%d_%H_%M_%S')"
    export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
    export REGION=europe-west4

    gcloud ml-engine jobs submit training $JOB_NAME \
        --job-dir $JOB_DIR \
        --runtime-version 1.10 \
        --module-name trainer.inria \
        --package-path ./trainer \
        --region $REGION \
        -- \
        --train-file ./inzynierka
git add .
git commit -m "$1"
git push
else
    echo "Provide a name for a job!"
fi