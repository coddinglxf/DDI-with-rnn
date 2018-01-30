# encoding=utf8
import torch
import config
import os, sys
from torch import nn, optim
from torch import autograd
import numpy as np
from resRnn import resRnn
from sklearn.metrics import classification_report
from torch.autograd import Variable
import torch.nn.functional as F
import sys
import time

from Datasets import *
from MarcoF import *


class Attention(nn.Module):
    def __init__(self, hidden):
        super(Attention, self).__init__()
        self.hidden = hidden * 2 if config.args.bi_lstm else hidden
        self.linear_att = nn.Linear(self.hidden, self.hidden)
        self.linear_hidden = nn.Linear(self.hidden, self.hidden)

        self.att_trans = nn.Linear(self.hidden, 1, bias=False)

        # self.input_embedding = input_embedding
        self.highway = nn.Linear(config.args.embed, self.hidden, bias=False)
        self.highway_activation = F.tanh
        pass

    def forward(self, input, mask):
        # input = Variable(input)                                          # seq * batch * hidden
        # mask = Variable(mask)                                            # batch * seq
        input_trans = input.view(config.args.batch, -1, self.hidden)  # batch * seq * hidden
        # input_trans = input_trans + highway                                # resnet here

        mask_trans = mask.unsqueeze(dim=2)  # batch * seq * 1

        final = []
        att_ret = []
        for index in range(config.args.batch):
            att = self.linear_att(input_trans[index])
            hidden = self.linear_hidden(input_trans[index])

            att_trans = self.att_trans(att)

            att_trans = att_trans + (mask_trans[index] - 1.0) * float(1e8)

            att_trans = att_trans.view(1, -1)  # 1 * seq
            att_trans = F.softmax(att_trans)
            att_trans_temp = att_trans
            att_trans = att_trans_temp.view(-1, 1)  # seq * 1

            # hidden = F.tanh(hidden)

            final_res = hidden * att_trans.expand_as(hidden)
            final_res = final_res.sum(0)
            final.append(final_res)

            att_ret.append(att_trans_temp)
        ret = torch.cat(final, )
        return ret, att_trans_temp


class Classifier(nn.Module):
    def __init__(self, input_dim, classes, ):
        super(Classifier, self).__init__()
        self.input_dim = input_dim * 2 if config.args.bi_lstm else input_dim
        self.classes = classes

        self.linear_1 = nn.Linear(self.input_dim, self.input_dim)
        self.linear_2 = nn.Linear(self.input_dim, self.classes)

    def forward(self, input):
        # input is batch * hidden
        outputs = self.linear_1(input)
        outputs = F.relu(outputs)
        outputs = self.linear_2(outputs)
        # outputs = F.tanh(outputs)
        # softmax_outputs = F.log_softmax(outputs)
        return outputs


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.n_layers = config.args.n_layers
        self.hidden_size = config.args.hidden
        self.batch = config.args.batch
        self.input_size = len(initial.word2index)
        self.embed_size = config.args.embed

        self.embedding = nn.Embedding(self.input_size, self.embed_size)
        self.embedding.weight.data = torch.from_numpy(np.array(initial.pre_trained, dtype="float32"))

    def forward(self, input):
        return self.embedding(input).view(-1, self.batch, self.embed_size)


# ------------ model define here ---------------
print("build the model")
seq = nn.Sequential()
embed = Embedding()

if config.use_cuda:
    embed = embed.cuda(config.args.gpu)
seq.add_module('embed', embed)

rnn_list = []
for layer in range(config.args.n_layers):
    if layer == 0:
        rnn = resRnn(
            input_dim=config.args.embed,
            hidden=config.args.hidden
        )
    else:
        input_dim = config.args.hidden * 2 if config.args.bi_lstm else config.args.hidden
        rnn = resRnn(
            input_dim=input_dim,
            hidden=config.args.hidden
        )

    if config.use_cuda:
        rnn = rnn.cuda(config.args.gpu)

    rnn_list.append(rnn)
    seq.add_module("rnn_{0}".format(layer), rnn)

attention_model = Attention(hidden=config.args.hidden)
classifier = Classifier(config.args.hidden, classes=config.args.classes)

# add the embed, attention_model, and classifier
if config.use_cuda:
    attention_model = attention_model.cuda(config.args.gpu)
    classifier = classifier.cuda(config.args.gpu)

# add model to seq to generate the parameters
seq.add_module('attention_model', attention_model)
seq.add_module('classifier', classifier)

# ------------ add loss fun and optimizer here ------------
print("build the loss function")
loss_fun = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(seq.parameters(), lr=0.5)
seq_len = 10


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


def main():
    train = Data(filename="xml/train/")
    test = Data(filename="xml/test/")

    train_instances = train.features
    test_instances = test.features
    for it in range(1, config.args.it):
        length = len(train_instances)
        permutation = np.random.permutation(length)
        batch_num = length / config.args.batch

        all_loss = 0
        if it > 1:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * config.args.lr_factor

        start_time = time.time()
        for i in range(int(batch_num)):

            # should zero grad
            optimizer.zero_grad()

            start = config.args.batch * i
            end = config.args.batch * (i + 1)
            batch_data = [train_instances[each] for each in permutation[start:end]]
            index, mask, label = generate_batch_data(batch_data)

            index_var = Variable(torch.from_numpy(index))
            mask_var = Variable(torch.from_numpy(mask))
            label_var = Variable(torch.from_numpy(label))

            if config.use_cuda:
                index_var = index_var.cuda(config.args.gpu)
                mask_var = mask_var.cuda(config.args.gpu)
                label_var = label_var.cuda(config.args.gpu)

            # print(type(embed))
            embed_out = embed(index_var)
            # embed_out = embed_out.cuda(config.args.gpu)
            # print(embed_out.data)
            # print(type(embed_out))

            for layer in range(config.args.n_layers):
                # embed_out = torch.FloatTensor(embed_out)
                # print(type(embed_out))
                rnn = rnn_list[layer]
                # print(type(rnn))
                embed_out, hidden = rnn(embed_out)

            ret, att = attention_model(embed_out, mask_var)
            softmax = classifier(ret)
            loss = loss_fun(softmax, label_var)

            all_loss += loss.data.cpu().numpy()[0]
            if i % config.args.display_freq == 0:
                # print(softmax)
                # print(output.data.cpu()numpy(0))
                # print(ret)
                # print(optimizer.param_groups[0]['lr'])
                # print(att.data.cpu()numpy()[0:20])
                end_time = time.time()
                sys.stdout.flush()
                print("it is in it {0}, and in batch {1}/{2}, the loss is {3}, lr is {4}, time is {5}".format(
                    it, i, batch_num, all_loss / (i + 1), optimizer.param_groups[0]['lr'], (end_time - start_time)
                ))
                # print(encoder_rnn.embedding.weight.data.cpu().numpy()[initial.word2index["the"]])
                sys.stdout.flush()

            # if i > 10:
            #     break
            # backward the function

            loss.backward()
            optimizer.step()
            # print("------------------------------------------------------------------------")
        # start to eval the test data

        if it % config.args.eval_interval == 0:

            sys.stdout.flush()
            print("start to evaluation in it {0}".format(it))
            sys.stdout.flush()

            length = len(test_instances)
            batch_num = length / config.args.batch
            pred = []
            true = []

            start_time = time.time()

            # true_all = []
            feature_all = []

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

                # add the res connection here
                embed_out = embed(index_var)

                for layer in range(config.args.n_layers):
                    rnn = rnn_list[layer]
                    embed_out, hidden = rnn(embed_out)

                ret, att = attention_model(embed_out, mask_var)
                softmax = classifier(ret)

                softmax = softmax.data.cpu().numpy()
                for each in softmax.argmax(1):
                    pred.append(each)

                ret_features = ret.data.cpu().numpy()
                # print(ret_features)
                for i_index in range(ret_features.shape[0]):
                    feature_all.append(list(ret_features[i_index]))
            # eval

            filename = "features/" + str(it)
            writefile = open(filename, 'w')
            for t, fs in zip(true, feature_all):
                fs_str = " ".join([str(each) for each in fs])
                writefile.write(str(t) + "\t" + fs_str + "\n")
            writefile.close()

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
            # save the model here
            # if f > 1:
            #     print("save model in it {}".format(it))
            #     path = "{0}_{1}_{2}_{3}".format(config.args.hidden, config.args.n_layers, it, f)
            #     current = os.path.join("save", path)
            # 
            #     if os.path.exists(current) is False:
            #         os.mkdir(current)
            # 
            #     torch.save(encoder_rnn, os.path.join(current, "encoder"))
            #     torch.save(attention_model, os.path.join(current, "attention"))
            #     torch.save(classifier, os.path.join(current, "classifier"))

            print(
                classification_report(
                    y_true=true,
                    y_pred=pred,
                    digits=4
                )
            )
            sys.stdout.flush()


if __name__ == '__main__':
    main()
