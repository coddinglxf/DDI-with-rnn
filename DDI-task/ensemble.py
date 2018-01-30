# from main import EncoderRNN, Attention, Classifier

import argparse

import config
from config import file_list
import os
import torch
from Datasets import Data
import time
import numpy as np
from torch.autograd import Variable
from MarcoF import *
import sys, os
from sklearn.metrics import classification_report
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=int, default=64)
args = parser.parse_args()

# encoder_rnn = EncoderRNN()
# attention_model = Attention(hidden=config.args.hidden)
# classifier = Classifier(config.args.hidden, classes=config.args.classes)
print(file_list)
model_list = []
for filename in file_list:

    # filter the number

    filter_batch = str(args.batch)
    if str(filename).startswith(filter_batch) is False:
        continue

    current_path = os.path.join("save", filename)
    print("load file from {0}".format(current_path))
    encoder_path = os.path.join(current_path, "encoder")
    attention_path = os.path.join(current_path, "attention")
    classifier_path = os.path.join(current_path, "classifier")

    encoder_rnn = torch.load(encoder_path)
    attention_model = torch.load(attention_path)
    classifier = torch.load(classifier_path)

    model_list.append([encoder_rnn, attention_model, classifier])

print(model_list)

# ensemble here

test = Data(filename="xml/test/")

test_instances = test.features

length = len(test_instances)
batch_num = length / config.args.batch
pred = []
true = []


def generate_batch_data(batch_data):
    input_index = []
    input_mask = []
    input_label = []
    for batch in batch_data:
        input_index.append(batch['all_sequence'])
        input_mask.append(batch['mask'])
        input_label.append(batch['label'])
    return np.array(input_index, dtype="int64"), \
           np.array(input_mask, dtype="float32"), \
           np.array(input_label, dtype="int64")


start_time = time.time()
for i in range(int(batch_num)):
    start = config.args.batch * i
    end = config.args.batch * (i + 1)
    batch_data = test_instances[start:end]

    index, mask, label = generate_batch_data(batch_data)

    for each in label:
        true.append(each)

    index_var = Variable(torch.from_numpy(index))
    mask_var = Variable(torch.from_numpy(mask))
    # label_var = Variable(torch.from_numpy(label))
    if config.use_cuda:
        index_var = index_var.cuda(config.args.gpu)
        mask_var = mask_var.cuda(config.args.gpu)

    ensemble = np.zeros((config.args.batch, config.args.classes))
    for (encoder_rnn, attention_model, classifier) in model_list:
        output, hidden = encoder_rnn(index_var)
        ret, att = attention_model(output, mask_var)
        softmax = classifier(ret)
        softmax = F.softmax(softmax)
        softmax = softmax.data.cpu().numpy()
        ensemble += softmax
    if i % 300 == 0:
        print("current deal with {0}".format(i))
    for each in ensemble.argmax(1):
        pred.append(each)
# eval
true = np.array(true)
pred = np.array(pred)

sys.stdout.flush()
end_time = time.time()
print("test time cost is ", (end_time - start_time))
f = calculateMicroValue(
    y_pred=pred,
    y_true=true,
    labels=[0, 1, 2, 3]
)

print(
    classification_report(
        y_true=true,
        y_pred=pred,
        digits=4
    )
)
sys.stdout.flush()
