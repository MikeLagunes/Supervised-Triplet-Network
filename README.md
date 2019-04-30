# Supervised-Triplet-Network

Code for the ICRA 2019 paper "Learning Discriminative Embeddings for Object Recognition on-the-fly"

## Getting Started

The code was developed in PyTorch 1.0 and Python 3.6, we provide the original scripts used in our paper and a working demo with the CORe50 dataset.
### Prerequisites

We used Anaconda for managing the Python environment. The prerequisites are listed in the ```s-triplet-env.txt``` file. We recomend
creating a new conda environment by doing:


```
conda create --name s-triplet-v1 --file s-triplet-env.txt
```

### Installing

Simply clone this repository into your working directory and download the CORe50 dataset.

## Training the model

In the file ```train/train_triplet_resnet_softmax.py```, we load the CNN model and define the hyperparameters. 

```
python train/train_triplet_resnet_softmax.py
```

### Test the model

The model accuracy of the model is calculated every number of epochs, defined in the training script. It is also possible to manually
evaluate the accuary of the model by running the next line:

```
python test/test_embeddings.py
```

We also provide the file for visualizing the emddings into a two-dimensionality representation using t-SNE. Simply indicate the location of
the embeddings from training and testing images in the ```utils/tsne.py``` ans run:

### Visualize the embeddings
```
python utils/tsne.py
```


## Authors

* **Miguel Lagunes-Fortiz** - *Corresponding author* - [Bristol Robotics Lab](https://github.com/MikeLagunes)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
