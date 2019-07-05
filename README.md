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

### Evaluate

```shell
$ python evaluate.py --arch efficientnet_b0 -r /path/to/dataset
```

## Pretrained models

| Model Name | Top-1 Accuracy |
| ------ | ------ |
| efficientnet-b0 | 75.74% |
| efficientnet-b1 |  |
| efficientnet-b2 |  |
| efficientnet-b3 |  |
| efficientnet-b4 |  |
| efficientnet-b5 |  |
| efficientnet-b6 |  |
| efficientnet-b7 |  |

## References

- https://arxiv.org/abs/1905.11946
- https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
- https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
