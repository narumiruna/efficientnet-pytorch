# EfficientNet

https://arxiv.org/abs/1905.11946

## Prerequisites

- Ubuntu
- Python 3
  - torch 1.0.1
  - torchvision 0.2.2.post3

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
