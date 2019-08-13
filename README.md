# LAMBDA NOTES

#### Download Data

```
wget https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz
tar -xvzf cifar10.tgz
mkdir .fastai/data
mv cifar10 .fastai/data/
```

#### Install Prerequisites

Lambda GPU Cloud uses CUDA 9 and fastai requires minimum Pytorch Version 1.0.0. Hence we choose the conda installation of pytorch built with CUDA 9.
```
curl -O https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
bash Anaconda3-5.0.1-Linux-x86_64.sh

conda create --name ana --python=3.6
source activate ana
conda install -n ana pytorch=1.0.0 torchvision cuda90 -c pytorch
conda install -n ana -c fastai fastai
```

#### Clone Original BaiduNet9P repo

Basically we use the same code as in the original repo, expect removing the average pooling layer that generates erros in our experiments.

```
git clone https://github.com/BAIDU-USA-GAIT-LEOPARD/CIFAR10-Distributed-Training-BaiduNet9P.git
cd CIFAR10-Distributed-Training-BaiduNet9P/

mkdir ~/anaconda3/envs/ana/lib/python3.6/site-packages/fastai/examples
cp BaiduNet.py ~/anaconda3/envs/ana/lib/python3.6/site-packages/fastai/vision/models/
cp train_cifar10_BaiduNet9P.py ~/anaconda3/envs/ana/lib/python3.6/site-packages/fastai/examples/
cp basic_train.py ~/anaconda3/envs/ana/lib/python3.6/site-packages/fastai
cp data.py ~/anaconda3/envs/ana/lib/python3.6/site-packages/fastai/vision/

```

#### Run Training

```
python -m fastai.launch --gpus=0123 train_cifar10_BaiduNet9P.py
```

Results are logged in the `Perf_BaiduNet9P.tsv` file:

```
epoch	hours	top1Accuracy
0	0.00068544	51.91
1	0.00129557	61.67
2	0.00185631	69.33
3	0.00242813	68.15
4	0.00296299	71.69
5	0.00350515	74.32
6	0.00406932	74.65
7	0.00461784	68.43
8	0.00519758	67.71
9	0.00573748	73.23
10	0.00630787	76.88
11	0.00684259	78.40
12	0.00738441	80.79
13	0.00794037	79.08
14	0.00849998	86.78
15	0.00904122	86.95
16	0.00958996	88.26
17	0.01016918	91.03
18	0.01070541	91.54
19	0.01126079	92.81
20	0.01179140	92.99
21	0.01234526	94.01
22	0.01289714	94.14
23	0.01343083	94.18
```

# CIFAR10-Distributed-Training-BaiduNet9P
DAWN CIFAR10 distributed training results by BAIDU USA GAIT LEOPARD team

CIFAR10-Distributed-Training-BaiduNet9P

Codes for DAWN training on CIFAR10 using 8xV100 on Baidu Cloud


Training
----------
Training a small network (BaiduNet9P) to reach 94.0% test accuracy on CIFAR10 data using Baidu Cloud Tesla 8*V100 GPU with 16 GB memory.

To reproduce our results: 

1. Log into Baidu Cloud, install Pytorch, setup fastai environment: https://github.com/fastai/fastai 
2. Add BaiduNet.py to fastai/fastai/vision/models
3. Add training_cifar10_BaiduNet9P.py to fastai/examples
4. Replace basic_train.py and data.py in corresponding fastai library repositories 
5. Issue the following command:

python -m fastai.launch training_cifar10_BaiduNet9P.py

The detailed traning process will be demonstrated in table format on the screen, including data preprocessing time (transformation and normalization). The performance including training time for each epoch will be saved in Perf_BaiduNet9P.tsv

In our tests, we can reach 94.0% test accuracy in about 45s (including data preprocessing time of 0.5s roughly).

