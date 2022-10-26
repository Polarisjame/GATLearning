# GAT Learning

A basic GAT model using residual connection as [**DeepGCNs**](https://arxiv.org/abs/1904.03751) described, trained to classify a paper node into 7 classes.

---
# Uploads

Upload GAT_selfattn.py using basic GAT model with residual connection and MultiHeadSelfAttention

## Requirements

My code works with the following environment.
* `python=3.7`
* `pytorch=1.12.1+cu116`
* `tensorboard=2.10.1`
* `dglcu116=0.9.1`

## Dataset

using `CoraGraphDataset` containing 2708 paper nodes and 10556 edges. Each node is characterized by a word frequency vector.You can import this Dataset from dgl using `from dgl.data import CoraGraphDataset`

## Training and Testing

You can run `python GAT.py` to train a model in cmd line and `python GAT.py -h` to get help.

Here are some important parameters:

* `--num_layers`: total layer of the GAT model.
* `--re_zero_lr`: a single learning rate for the parameter used in ReZero Residual connection.
* `--dense_net`: use Densenet.
* `--dense_dim`: the outdim of the Dense Block if using Densenet if `--dense_net` is True.
* `--res_add`: use Residual connection.
* `--re_zero`: use ReZero Residual connection if `--res_add` is True.
* `--num_heads`: number of heads in MultiheadAttention.
* `--l2`: a float if using **weight_decay** else 0.

## Results

You can see my checkpoints and loss/acc during my trainingin in [CheckPoints](./path/checkpoints). Here are the example ![fig1](https://github.com/Polarisjame/GATLearning/blob/main/path/checkpoints/acc-95.40%25_layers-1_lr-5.00e-04_hid_dim-128_res.jpg)
And you can also open [Tensorboardlog](path\log) in Tensorboard to see the Visualization of train.
