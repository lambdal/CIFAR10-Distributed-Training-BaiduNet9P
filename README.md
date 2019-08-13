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
0	0.00119374	49.59
1	0.00232872	66.02
2	0.00347805	67.05
3	0.00469134	75.55
4	0.00573558	77.61
5	0.00695555	78.60
6	0.00812365	72.88
7	0.00920907	74.86
8	0.01037075	74.25
9	0.01170565	80.51
10	0.01291092	82.47
11	0.01404239	75.84
12	0.01518758	81.48
13	0.01641208	85.68
14	0.01770511	87.64
15	0.01891448	86.36
16	0.02007843	89.91
17	0.02136751	83.58
18	0.02252023	91.47
19	0.02367967	92.52
20	0.02482016	93.24
21	0.02591234	93.84
22	0.02711574	93.99
23	0.02829251	94.02
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

