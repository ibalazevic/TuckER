
## TuckER: Tensor Factorization for Knowledge Graph Completion

This repository contains code for the paper [TuckER: Tensor Factorization for Knowledge Graph Completion](https://).

### Running a model

     CUDA_VISIBLE_DEVICES=0 python main.py --dataset FB15k-237 --num_iterations 500 --batch_size 128
                                           --lr 0.0005 --dr 1.0 --edim 200 --rdim 200 --input_dropout 0.3 
                                           --hidden_dropout1 0.4 --hidden_dropout2 0.5 --label_smoothing 0.1

Available datasets are:
    
    FB15k-237
    WN18RR
    FB15k
    WN18
    
### Citation


