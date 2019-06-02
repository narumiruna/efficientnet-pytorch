# EfficientNet

https://arxiv.org/abs/1905.11946

## Usage

### Train on MNIST dataset

```shell
$ python train.py -c configs/mnist.yaml
```

### Train on CIFAR10 dataset

```shell
$ python train.py -c configs/cifar10.yaml
```

## TO-DO

- [ ] Support ImageNet dataset
- [ ] Pre-trained weights

## References

- https://arxiv.org/abs/1905.11946
- https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
- https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
