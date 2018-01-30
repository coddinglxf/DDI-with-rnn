import torch
import config
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F


class resRnn(nn.Module):
    def __init__(self, input_dim, hidden):
        super(resRnn, self).__init__()
        self.hidden = config.args.hidden
        self.input_dim = input_dim
        self.gru = torch.nn.GRU(self.input_dim,
                                hidden_size=self.hidden,
                                bidirectional=config.args.bi_lstm,
                                num_layers=config.args.n_layers)

        # may project to 2*hidden if bi_lstm
        self.hidden_size = self.hidden * 2 if config.args.bi_lstm else hidden
        self.highway = nn.Linear(self.input_dim, self.hidden_size, bias=False)
        self.n_layers = config.args.n_layers
        self.batch = config.args.batch

        self.init_hidden = self.initHidden()

    def forward(self, input):
        # batch_size * seqlen * input_dim
        output = input.view(-1, self.input_dim)
        output = self.highway(output)
        output = F.tanh(output)
        output = output.view(config.args.batch, -1, self.hidden_size)

        # print("type is ------------- ", input)

        bi_output, hidden = self.gru(input, self.init_hidden)
        # resNet here
        bi_output = bi_output + output
        return bi_output, hidden

    def initHidden(self):
        num = 2 if config.args.bi_lstm else 1
        result = Variable(torch.zeros(self.n_layers * num, self.batch, self.hidden))
        if config.use_cuda:
            return result.cuda(config.args.gpu)
        else:
            return result


from torch.autograd import function
