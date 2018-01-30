# encoding=utf8
import torch
import config
import os, sys
from torch import nn, optim
from torch import autograd
import numpy as np

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
        pass

    def forward(self, input, mask):
        # input = Variable(input)                                          # seq * batch * hidden
        # mask = Variable(mask)                                            # batch * seq
        input_trans = input.view(config.args.batch, -1, self.hidden)  # batch * seq * hidden
        mask_trans = mask.unsqueeze(dim=2)  # batch * seq * 1

        final = []
        att_ret = []
        for index in range(config.args.batch):
            att = self.linear_att(input_trans[index])
            hidden = self.linear_hidden(input_trans[index])

            # att_tanh = F.tanh(att)

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


class EncoderRNN(nn.Module):
    def __init__(self):
        super(EncoderRNN, self).__init__()
        self.n_layers = config.args.n_layers
        self.hidden_size = config.args.hidden
        self.batch = config.args.batch
        self.input_size = len(initial.word2index)
        self.embed_size = config.args.embed

        self.embedding = nn.Embedding(self.input_size, self.embed_size)

        # do not initialize the embedding here
        # self.embedding.weight.data = torch.from_numpy(np.array(initial.pre_trained, dtype="float32"))
        # self.embedding.weight.requires_grad = True

        self.gru = nn.GRU(self.embed_size, self.hidden_size, num_layers=self.n_layers,
                          bidirectional=config.args.bi_lstm)

        self.hidden = self.initHidden()

    # All Step of Rnn
    # gruCell is different from gru
    def forward(self, input):
        # seq_len * batch * input_size
        embedded = self.embedding(input).view(-1, self.batch, self.embed_size)
        output = embedded
        output, hidden = self.gru(output, self.hidden)
        return output, hidden

    def initHidden(self):

        num = 2 if config.args.bi_lstm else 1
        result = Variable(torch.zeros(self.n_layers * num, self.batch, self.hidden_size))
        if config.use_cuda:
            return result.cuda(config.args.gpu)
        else:
            return result


# ------------ model define here ---------------
encoder_rnn = EncoderRNN()
attention_model = Attention(hidden=config.args.hidden)
classifier = Classifier(config.args.hidden, classes=config.args.classes)

if config.use_cuda:
    encoder_rnn = encoder_rnn.cuda(config.args.gpu)
    attention_model = attention_model.cuda(config.args.gpu)
    classifier = classifier.cuda(config.args.gpu)

# ------------ model define here ----------------

# ------------ add all parameters ----------------
seq = nn.Sequential()
seq.add_module("encoder", encoder_rnn)
seq.add_module("attention", attention_model)
seq.add_module("classifier", classifier)
# ------------ add all parameters -----------------


# ------------ add loss fun and optimizer here ------------
loss_fun = nn.CrossEntropyLoss()
# para_list = []
# --------- remove the parameters not update here -------------
# for param in seq.parameters():
#     print(type(param), param.requires_grad, type(param.data), param.size(), type(seq.parameters()))
#     if param.requires_grad is True:
#         para_list.append(param)
optimizer = optim.Adadelta(seq.parameters(), lr=0.5)

# optimizer = optim.SGD(seq.parameters(), lr=0.1)
# print("current parameters {0}".format(seq.parameters()))
# models = [encoder_rnn, attention_model, classifier]
# for model in models:
#     name_tuple = model.named_parameters()
#     print(name_tuple)
#     for a, b in name_tuple:
#         print(a)
# ------------ add loss fun and optimizer  ------------

seq_len = 10


def train_test():
    for it in range(config.args.it):
        for batch_time in range(10):
            input_index = np.random.randint(low=0, high=100, size=(config.args.batch, seq_len))
            inputs_var = Variable(torch.from_numpy(np.array(input_index, dtype="int64")))
            # inputs_var = torch.LongTensor()
            masks = Variable(torch.from_numpy(np.array(np.ones(shape=(config.args.batch, seq_len), dtype="float32"))))
            input_class = np.random.randint(low=0, high=config.args.classes,
                                            size=(config.args.batch,))
            inputs_class_var = Variable(torch.from_numpy(np.array(input_class, dtype="int64")))

            if config.use_cuda:
                inputs_var = inputs_var.cuda(1)
                masks = masks.cuda(1)
                inputs_class_var = inputs_class_var.cuda(1)

            # forward the function
            output, hidden = encoder_rnn(inputs_var)
            ret = attention_model(output, masks)
            softmax = classifier(ret)
            loss = loss_fun(softmax, inputs_class_var)
            print("it is in it {0}, and in batch {1}, the loss is {2}".format(
                it, batch_time, loss.data.cpu().numpy()[0]
            ))

            # backward the function
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("------------------------------------------------------------------------")


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

            output, hidden = encoder_rnn(index_var)
            ret, att = attention_model(output, mask_var)
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
                # if i > 100:
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

                output, hidden = encoder_rnn(index_var)
                ret, att = attention_model(output, mask_var)
                softmax = classifier(ret)

                softmax = softmax.data.cpu().numpy()
                for each in softmax.argmax(1):
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
            # save the model here
            if f > 0.7:
                print("save model in it {}".format(it))
                path = "{0}_{1}_{2}_{3}".format(config.args.hidden, config.args.n_layers, it, f)
                current = os.path.join("save", path)

                if os.path.exists(current) is False:
                    os.mkdir(current)

                torch.save(encoder_rnn, os.path.join(current, "encoder"))
                torch.save(attention_model, os.path.join(current, "attention"))
                torch.save(classifier, os.path.join(current, "classifier"))

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
