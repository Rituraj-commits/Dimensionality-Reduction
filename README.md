# Dimensionality Reduction

This respository contains various Dimensionlity Reduction Technqiues tested on different datasets ( currently implemented only in Fashion-MNIST )

## Description

t-SNE and PCA are currently the most active techniques used for Dimensionlity Reduction.t-Distributed Stochastic Neighbor Embedding (t-SNE) is a technique for dimensionality reduction that is particularly well suited for the visualization of high-dimensional datasets. The technique can be implemented via Barnes-Hut approximations, allowing it to be applied on large real-world datasets. t-SNE is currently the most active technique for Dimensionlity Reduction. Here we have implemented t-SNE Cuda on the Fashion MNIST Dataset and compared the results with that of the classical t-SNE. For more information on TSNE CUDA go through this link https://github.com/CannyLab/tsne-cuda. 

## Getting Started

### Dependencies

* Linux
* GPU
* CUDA
* python >= 3.x
* matplotlib
* tsnecuda

## Installation

### Installing *tsnecuda*
Visit this link for tsnecuda overview and installation [tsne](https://github.com/CannyLab/tsne-cuda/wiki/Installation)

## Results

| Technique(s) | Time (in seconds) |
| --- | --- |
| PCA | ~0.8s |
| t-SNE | ~ 65s |
| t-SNE CUDA | ~ 4s |


## Acknowledgments

Inspiration, code snippets, etc.
* [LAURENS VAN DER MAATEN](https://lvdmaaten.github.io/tsne/)
* [TSNE CUDA](https://github.com/CannyLab/tsne-cuda)
* [Code](https://www.datacamp.com/community/tutorials/introduction-t-sne)
* [Dataset](https://github.com/zalandoresearch/fashion-mnist)
