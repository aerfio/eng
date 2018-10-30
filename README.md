# Simple python neural network deployed in Google Cloud

## Usage

Please first read [this](https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction) tutorial.

## Installation

Run following commands, that fetch MNIST dataset:

```bash
mkdir data
curl -O https://s3.amazonaws.com/img-datasets/mnist.pkl.gz
gzip -d mnist.pkl.gz
mv mnist.pkl data/
```

Then send that data to your Google Cloud Bucket:

```bash
gsutil mb gs://your-bucket-name
gsutil cp -r data/mnist.pkl gs://your-bucket-name/data/mnist.pkl
```

And change `startInCloud.sh` and `startLocally.sh`

## Useful links

1. https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction#setup
2. https://github.com/clintonreece/keras-cloud-ml-engine
3. https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction#setup
4. http://liufuyang.github.io/2017/04/02/just-another-tensorflow-beginner-guide-4.html
5. https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
6. https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/
7. https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d
8. https://becominghuman.ai/making-a-simple-neural-network-classification-2449da88c77e
9. https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/index.html#0
