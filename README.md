# Information Maximizing Self Augmented Training (IMSAT)
This is a reproducing code for IMSAT [1]. IMSAT is a method for discrete representation learning using deep neural networks. It can be applied to clustering and hash learning to achieve the state-of-the-art results. This is the work performed while Weihua Hu was interning at Preferred Networks.

## Requirements 
You must have the following already installed on your system.
- Python 2.7
- Chainer 1.21.0, sklearn, munkres

## Quick start
For reproducing the experiments on MNIST datasets in [1], run the following codes.
- Clustering with MNIST: ``` python imsat_cluster.py ```
- Hash learning with MNIST: ``` python imsat_hash.py ```

`calculate_distance.py` can be used to calculate the perturbation range for Virtual Adversarial Training [2]. For MNIST dataset, we have already calculated the range.

## Run on MNIST and Fashion-MNIST (LYL)
OK to run `calculate_distance.py` and `imsat_cluster.py`, but not `imsat_hash.py`.

### Requirements

Python 3.5.6

requirements.txt (among which munkres need to be installed by pip)

All other stuff is in the repository.

### HOWTO

#### Step 1

(You can skip it for MNIST and Fashion-MNIST)

```shell
python calculate_distance.py --dataset {mnist,fashioin-mnist}
```

You should find `10th_neighbor.txt` in a folder with the same name as the dataset.

#### Step 2

```shell
python imsat_cluster.py --dataset {mnist,fashion-mnist}
```

### NOTE

`imsat_cluster.py` may behave somewhat differently (`chainer.links.Linear` no longer accepts `wscale` argument, see line 87 - 89 for details), but the results on MNIST seem quite similar.

## Reference ##
[1] Weihua Hu, Takeru Miyato, Seiya Tokui, Eiichi Matsumoto and Masashi Sugiyama. Learning Discrete Representations via Information Maximizing Self-Augmented Training. In ICML, 2017. Available at http://arxiv.org/abs/1702.08720

[2] Takeru Miyato, Shin-ichi Maeda, Masanori Koyama, Ken Nakae, and Shin Ishii. Distributional smoothing with virtual adversarial training. In ICLR, 2016.
