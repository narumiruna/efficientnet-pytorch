# EfficientNet

https://arxiv.org/abs/1905.11946

## Prerequisites

- Ubuntu or macOS
- Python 3
  - torch 1.1.0
  - torchvision 0.3.0

## Usage

### Train

```shell
$ python train.py -c /path/to/config -r /path/to/dataset
```

## TO-DO

- [x] Support ImageNet dataset
- [ ] Pre-trained weights

## References

- https://arxiv.org/abs/1905.11946
- https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
- https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
