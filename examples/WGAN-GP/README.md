## Using Elegy low-level API to train WGAN-GP on the CelebA dataset


***
### Usage
```
main.py --dataset=path/to/celeb_a/*.png --output_dir=<./output/path> [flags]


flags:
  --dataset:      Search path to the dataset images e.g: path/to/*.png
  --output_dir:   Directory to save model checkpoints and tensorboard log data

  --batch_size:   Input batch size (default: '64')
  --epochs:       Number of epochs to train (default: '100')
```

***
### Examples of generated images:

After  10 epochs: ![Example of generated images after 10 epochs](images/epoch-0009.png)

After  50 epochs: ![Example of generated images after 10 epochs](images/epoch-0049.png)

After 100 epochs: ![Example of generated images after 10 epochs](images/epoch-0099.png)


***
[1] Arjovsky, Martin, Soumith Chintala, and Léon Bottou. "Wasserstein generative adversarial networks." International conference on machine learning. PMLR, 2017.

[2] Gulrajani, Ishaan, et al. "Improved training of wasserstein gans." arXiv preprint arXiv:1704.00028 (2017).

[3] Liu, Ziwei, et al. "Large-scale celebfaces attributes (celeba) dataset." Retrieved August 15.2018 (2018): 11.
