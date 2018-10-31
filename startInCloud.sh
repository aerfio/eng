export BUCKET_NAME=aerfio-bucket
export JOB_NAME="train_$(date '+%Y_%m_%d_%H_%M_%S')"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=europe-west4

gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $JOB_DIR \
    --runtime-version 1.10 \
    --module-name trainer.mnist_mlp \
    --package-path ./trainer \
    --region $REGION \
    -- \
    --train-file ./inzynierka

echo "Paste this into console in google cloud"
echo "tensorboard --logdir=$JOB_DIR/logs --port=8080"

gcloud ml-engine jobs submit training fuwork13 --job-dir gs://aerfio-bucket/fuwork13 --runtime-version 1.10 --module-name trainer.mnist_mlp --package-path ./trainer --region europe-west4 --  --train-file ./inzynierka