export BUCKET_NAME=aerfio-bucket
export JOB_NAME="train_$(date '+%Y_%m_%d_%H_%M_%S')_batch_save"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=europe-west4

gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $JOB_DIR \
    --runtime-version 1.10 \
    --module-name trainer.mnist_mlp \
    --package-path ./trainer \
    --region $REGION \
    -- \
    --train-file gs://$BUCKET_NAME/data/mnist.pkl

echo "Paste this into console in google cloud"
echo "tensorboard --logdir=$JOB_DIR/logs --port=8080"