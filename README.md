# HGAMLP: A Scalable Training Framework for Heterogeneous Graph Neural Networks
OGBN-MAG result reproduction version
## Requirements

#### Neural network libraries for GNNs
+ Python 3.7.10
+ dgl==0.8.2
+ ogb==1.3.6
+ torch==1.12.1+cu102
+ torch_sparse==0.6.15

#### Data preparation
The pre-trained embeddings we use is the Line[1] method in RpHGNN[2].
To reproduce the results on the OGB Leaderboards (ogbn-mag), follow the steps below:
- For OGBN-MAG, the code will automatically download it via the ogb package.
- Download the pre-trained embeddings mag.p directly from [Google Drive](https://drive.google.com/file/d/1Q7gD1xpmLeFJu5xWWY3nwa46cM8xYClH/view?usp=sharing)

References:
- [1] Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, and Qiaozhu Mei. "Line: Large-scale information network embedding." In Proceedings of the 24th international conference on world wide web, pp. 1067-1077. 2015.
- [2] RpHGNN [https://github.com/CrawlScript/RpHGNN](https://github.com/CrawlScript/RpHGNN)
  
For experiments on the large dataset ogbn-mag, the dataset will be automatically downloaded from OGB challenge.

## Run HGAMLP for OGB Leaderboards (ogbn-mag) 
python main.py --stages 400 400 400 500 --extra-embedding Line --num-hops 2 --label-feats --num-label-hops 2 --n-layers-1 2 --n-layers-2 2 --n-layers-3 3  --act leaky_relu --bns --label-bns --lr 0.002 --weight-decay 0 --threshold 0.75 --patience 100 --gama 10 --amp --seeds 0 1 2 3 4 5 6 7 8 9 --gpu 0
