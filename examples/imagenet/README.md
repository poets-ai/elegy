## Training ResNet on ImageNet

Adapted from the [Flax](https://github.com/google/flax) library.

This example currently runs only on one device.

See `requirements.txt` for required packages, additional to Elegy.

***
### Usage
```
resnet_imagenet.py --model=<ResNetXX> --output_dir=<./output/path> [flags]


flags:
  --model: <ResNet18|ResNet34|ResNet50|ResNet101|ResNet152|ResNet200>: Type of ResNet to train
  --output_dir: Directory to save model checkpoints and tensorboard log data

  --base_lr:      SGD optimizer base learning rate (default: '0.1')
  --batch_size:   Input batch size (default: '64')
  --[no]cache:    Whether to cache the data in RAM (default: 'false')
  --dataset:      TFDS dataset name and version (default: 'imagenet2012:*.*.*')
  --dtype: <float16|float32>: Mixed precision or normal mode (default: 'float32')
  --epochs:       Number of epochs to train (default: '90')
  --image_size:   Image size in pixels (default: '224')
  --L2_reg:       L2 weight regularization (default: '0.0001')
  --loss_scale:   Loss scale for numerical stability when dtype=float16 (default: '1.0')
  --momentum:     SGD optimizer momentum (default: '0.9')
  --[no]nesterov: SGD optimizer Nesterov mode (default: 'true')
```

***
### Pretrained Models

| Model    | Top-1 accuracy | Weight Files |
| ---      | ---            | ---      |
| ResNet18 | 68.7%          | [model.pkl](https://www.dropbox.com/s/ofwh7947y6t84zp/ResNet18_ImageNet.pkl?dl=1) |
| ResNet50 | 76.5%          | [model.pkl](https://www.dropbox.com/s/fmr7tm6rmah682s/ResNet50_ImageNet.pkl?dl=1) |

Pretrained weights can be loaded with: `elegy.nets.ResNet18(weights='path/to/ResNet18_ImageNet.pkl')`

or with automatic download: `elegy.nets.ResNet18(weights='imagenet')`


***
[1] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

[2] Russakovsky, Olga, et al. "Imagenet large scale visual recognition challenge." International journal of computer vision 115.3 (2015): 211-252.
