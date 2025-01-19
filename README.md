# SignAttention: On the Interpretability of Transformer Models for Sign Language Translation

Official PyTorch implementation of [SignAttention: On the Interpretability of Transformer Models for Sign Language Translation](https://arxiv.org/abs/2410.14506)  
Accepted at IAI Workshop @ NeurIPS 2024  

Authors:  
[Pedro Alejandro Dal Bianco](https://pedroodb.github.io/) (&ast;), [Oscar Agustín Stanchi](https://indirivacua.github.io/) (&ast;), [Facundo Manuel Quiroga](https://facundoq.github.io/), [Franco Ronchetti](https://scholar.google.com/citations?user=yjCYizMAAAAJ&hl=es), [Enzo Ferrante](https://eferrante.github.io/).  
(&ast; denotes equal contribution)

## Overview
This repository provides the official implementation of our paper "SignAttention: On the Interpretability of Transformer Models for Sign Language Translation". The paper presents the first comprehensive interpretability analysis of a Transformer-based model for translating Greek Sign Language videos into glosses and text, leveraging attention mechanisms to provide insights into the model's decision-making processes.

## Dependencies
The code has been written in Python (3.10) and requires PyTorch (2.0). Install the required dependencies with:

```bash
pip install -r requirements.txt
```

## Data
We utilize the Greek Sign Language Dataset, which contains 10,290 samples of RGB videos, gloss annotations, and Greek language translations. This dataset is ideal for interpretability analysis due to its detailed gloss annotations and repetition of sentences. Download the dataset [here](https://vcl.iti.gr/dataset/gsl/).

## Experiments
You can view the poster we presented at NeurIPS 2024, which includes sample visualizations of the attention mechanisms analyzed in the paper, by clicking the link below:

[Poster for NeurIPS 2024](https://neurips.cc/media/PosterPDFs/NeurIPS%202024/99148.png?t=1733288876.9028351)

## Usage

```
- src/train.ipynb                          : Train the model

- src/get_interp_weights.ipynb             : Obtain attention weights

- src/interp.ipynb                         : Visualize attention for a single sample

- src/interp_all.ipynb                     : Generate global attention visualizations
```

## Citation
If you find this repository useful for your work, please cite:

```bibtex
@article{bianco2024signattention,
  title={SignAttention: On the Interpretability of Transformer Models for Sign Language Translation},
  author={Bianco, Pedro Alejandro Dal and Stanchi, Oscar Agust{\'\i}n and Quiroga, Facundo Manuel and Ronchetti, Franco and Ferrante, Enzo},
  journal={arXiv preprint arXiv:2410.14506},
  year={2024}
}
```

For queries regarding the paper or the code, please contact pdalbianco@lidi.info.unlp.edu.ar and ostanchi@lidi.info.unlp.edu.ar, or open an issue on this repository.
