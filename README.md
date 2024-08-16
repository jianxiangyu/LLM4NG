# Leveraging Large Language Models for Node Generation in Few-Shot Learning on Text-Attributed Graphs

## LLM4NG

## Environment Settings
> python==3.8.0 \
> torch==1.12.0 \
> numpy==1.24.3 \
> scikit_learn==1.1.1 \
> torch-cluster==1.6.0 \
> torch-geometric==2.3.1 \
> torch-scatter==2.1.0 \
> torch-sparse==0.6.16 \
> torch-spline-conv==1.2.1 


## Usage

You can use the following commend to run edge predicitor;  

> python edge.py --dataset cora --model_type Edge

You can use the following commend to get node classifcation result;
If you don't want to use the adjacency matrix you can set the parameter lam to -1.

> python LLM4NG.py --dataset cora --model_type Node

