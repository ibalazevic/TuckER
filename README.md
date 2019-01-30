
## TuckER: Tensor Factorization for Knowledge Graph Completion

This codebase contains PyTorch implementation of the paper [TuckER: Tensor Factorization for Knowledge Graph Completion](https://arxiv.org/pdf/1901.09590.pdf).

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


