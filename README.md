# GSDMM - MercadoLibre Data Challenge

## Description

This repository runs the GSDMM algorithm on a small subset of the MercadoLibre Data Challenge of 2019.

After training and finding the clusters, they were mapped into the original challenge categories ( [related SO question](https://stackoverflow.com/questions/55258457/find-mapping-that-translates-one-list-of-clusters-to-another-in-python/55258990) ).

Bear in mind that I have not performed any parameter fine-tuning whatsoever; I'm just sticking to the defaults. Nevertheless, the model has achieved an accuracy of 67%.

## Running

``` 
pip install -r requirements.txt
python submission.py 
```

## Recommended reading

- [Short Text Topic Modeling](https://towardsdatascience.com/short-text-topic-modeling-70e50a57c883)

## Future developments:

- [ ] Clean up the code

- [ ] Use the faster Rust implementation

- [ ] Try Jonker-Volgenant algorithm for mapping the clusters

- [ ] Hyperparameter tuning ( K, alpha, beta and iterations ).



The original readme for the Python GSDMM repository follows below.


# GSDMM: Short text clustering

This project implements the Gibbs sampling algorithm for a Dirichlet Mixture Model of [Yin and Wang 2014](https://pdfs.semanticscholar.org/058a/d0815ce350f0e7538e00868c762be78fe5ef.pdf) for the 
clustering of short text documents. 
Some advantages of this algorithm:
 - It requires only an upper bound `K` on the number of clusters
 - With good parameter selection, the model converges quickly
 - Space efficient and scalable

This project is an easy to read reference implementation of GSDMM -- I don't plan to maintain it unless there is demand. I am however actively maintaining the much faster Rust version of GSDMM [here](https://github.com/rwalk/gsdmm-rust).

## The Movie Group Process
In their paper, the authors introduce a simple conceptual model for explaining the GSDMM called the Movie Group Process.

Imagine a professor is leading a film class. At the start of the class, the students
are randomly assigned to `K` tables. Before class begins, the students make lists of
their favorite films. The professor repeatedly reads the class role. Each time the student's name is called,
the student must select a new table satisfying one or both of the following conditions:

- The new table has more students than the current table.
- The new table has students with similar lists of favorite movies.

By following these steps consistently, we might expect that the students eventually arrive at an "optimal" table configuration.

## Usage
To use a Movie Group Process to cluster short texts, first initialize a [MovieGroupProcess](gsdmm/mgp.py):
```python
from gsdmm import MovieGroupProcess
mgp = MovieGroupProcess(K=8, alpha=0.1, beta=0.1, n_iters=30)
```
It's important to always choose `K` to be larger than the number of clusters you expect exist in your data, as the algorithm
can never return more than `K` clusters.

To fit the model:
```python
y = mgp.fit(docs)
```
Each doc in `docs` must be a unique list of tokens found in your short text document. This implementation does not support
counting tokens with multiplicity (which generally has little value in short text documents).
