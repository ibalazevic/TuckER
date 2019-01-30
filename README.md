
## TuckER: Tensor Factorization for Knowledge Graph Completion

<p align="center">
  <img src="https://raw.githubusercontent.com/ibalazevic/TuckER/master/tucker.png"/ width=400>
</p>

This codebase contains PyTorch implementation of the paper:

> TuckER: Tensor Factorization for Knowledge Graph Completion.
> Ivana Balažević, Carl Allen, and Timothy M. Hospedales.
> arXiv preprint arXiv:1901.09590, 2019.
> [[Paper]](https://arxiv.org/pdf/1901.09590.pdf)

### Link Prediction Results

Dataset | MRR | Hits@10 | Hits@3 | Hits@1
:--- | :---: | :---: | :---: | :---:
FB15k | 0.795 | 0.892 | 0.833 | 0.741
WN18 | 0.953 | 0.958 | 0.955 | 0.949
FB15k-237 | 0.358 | 0.544 | 0.394 | 0.266
WN18RR | 0.470 | 0.526 | 0.482 | 0.443

### Running a model

To run the model, execute the following command:

     CUDA_VISIBLE_DEVICES=0 python main.py --dataset FB15k-237 --num_iterations 500 --batch_size 128
                                           --lr 0.0005 --dr 1.0 --edim 200 --rdim 200 --input_dropout 0.3 
                                           --hidden_dropout1 0.4 --hidden_dropout2 0.5 --label_smoothing 0.1

Available datasets are:
    
    FB15k-237
    WN18RR
    FB15k
    WN18
    
### Requirements

The codebase is implemented in Python 3.6.6. Required packages are:

    numpy      1.14.5
    pytorch    0.4.0
    
### Citation


