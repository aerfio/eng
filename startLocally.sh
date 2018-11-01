#export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export JOB_DIR=./job_dir
gcloud ml-engine local train \
  --job-dir $JOB_DIR \
  --module-name trainer.inria \
  --package-path ./trainer \
  -- \
  --train-file ./inzynierka \
  --cache true