#!/bin/bash  
for((i=0;i<=39;i++));  
do   
CUDA_VISIBLE_DEVICES=0 python test.py -r ./checkpoints/0226_asl_lr3e-7_64/ep-$i.pth -v 
done