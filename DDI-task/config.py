import argparse
import numpy as np
import torch
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=str, default=1)
parser.add_argument("--pre-trained", type=str, default="pretrained/pubmed")
parser.add_argument('--train-path', type=str, default='data/train/')
parser.add_argument('--test-path', type=str, default="data/test/")
parser.add_argument("--hidden", type=int, default=64)
parser.add_argument("--n-layers", type=int, default=2)
parser.add_argument("--embed", type=int, default=200)
parser.add_argument("--classes", type=int, default=5)
parser.add_argument("--eval-interval", type=int, default=1)
parser.add_argument('--lr-factor', type=float, default=1)
parser.add_argument("--bi-lstm", type=bool, default=True)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--display-freq", type=int, default=1000)
parser.add_argument('--grad-clip', type=float, default=5)
parser.add_argument("--it", type=int, default=50)
# parser.add_argument()


args = parser.parse_args()
print("current config is ", args)
use_cuda = torch.cuda.is_available()

file_list = os.listdir("save")
