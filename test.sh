#!/bin/bash  
CUDA_VISIBLE_DEVICES=4 python test.py -r ./checkpoints/0226_asl_lr3e-7_64/ep-29.pth 
# CUDA_VISIBLE_DEVICES=4 python test.py -r ./checkpoints/0226_asl_lr3e-7_64/ep-29.pth -v 

# for((i=1;i<=39;i=i+2));  
# do   
# CUDA_VISIBLE_DEVICES=4 python test.py -r ./checkpoints/0313_signdataset_1e-7_256_crop/ep-$i.pth -v 
# done