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
| efficientnet_b0 | 75.74% |
| efficientnet_b1 |  |
| efficientnet_b2 |  |
| efficientnet_b3 |  |
| efficientnet_b4 |  |
| efficientnet_b5 |  |
| efficientnet_b6 |  |
| efficientnet_b7 |  |

## References

- https://arxiv.org/abs/1905.11946
- https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
- https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
