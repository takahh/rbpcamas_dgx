

def main():
    ######################################################################
    # constant
    ######################################################################
    import numpy

    proteinseq_file = "/Users/mac/Documents/transformer_tape_dnabert/data/proteinseqs.csv"
    data_path = "/Users/mac/Documents/transformer_tape_dnabert/data/RNAseq_and_label.csv"

    ######################################################################
    # import
    ######################################################################
    import math, random
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
    import torch
    from torchtext.datasets import WikiText2
    from torchtext.data.utils import get_tokenizer
    from torchtext.vocab import build_vocab_from_iterator

    import argparse
    import glob
    import json
    import logging
    import os
    import re
    import shutil
    import random
    from multiprocessing import Pool
    from typing import Dict, List, Tuple
    from copy import deepcopy

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
    from torch.utils.data.distributed import DistributedSampler
    from tqdm import tqdm, trange

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        from tensorboardX import SummaryWriter


    class TransformerModel(nn.Module):

        def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
            super(TransformerModel, self).__init__()
            self.model_type = 'Transformer'
            self.pos_encoder = PositionalEncoding(ninp, dropout)
            encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
            self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
            self.encoder = nn.Embedding(ntoken, ninp)
            self.ninp = ninp
            self.decoder = nn.Linear(ninp, ntoken)

            self.init_weights()

        def generate_square_subsequent_mask(self, sz):
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            return mask

        def init_weights(self):
            initrange = 0.1
            self.encoder.weight.data.uniform_(-initrange, initrange)
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)

        def forward(self, src, src_mask):
            src = self.encoder(src) * math.sqrt(self.ninp)
            src = self.pos_encoder(src)
            output = self.transformer_encoder(src, src_mask)
            output = self.decoder(output)
            return output


    class PositionalEncoding(nn.Module):

        def __init__(self, d_model, dropout=0.1, max_len=5000):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=dropout)

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer('pe', pe)

        def forward(self, x):
            x = x + self.pe[:x.size(0), :]
            return self.dropout(x)


    ######################################################################
    # Load and batch data
    # -------------------


    train_iter = WikiText2(split='train')
    tokenizer = get_tokenizer('basic_english')
    # vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
    # vocab.set_default_index(vocab["<unk>"])


    def data_process(raw_text_iter):
      data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
      return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


    # train_iter, val_iter, test_iter = WikiText2()
    # train_data = data_process(train_iter)
    # val_data = data_process(val_iter)
    # test_data = data_process(test_iter)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def batchify(data, bsz):
        # Divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.to(device)

    batch_size = 20
    eval_batch_size = 10
    # train_data = batchify(train_data, batch_size)
    # val_data = batchify(val_data, eval_batch_size)
    # test_data = batchify(test_data, eval_batch_size)


    # # make a list of seq of protein on memory as a referenece
    # protseq_dic = {}
    # with open(proteinseq_file) as f:
    #     for lines in f.readlines():
    #         protseq_dic[lines.split(",")[0]] = lines.split(",")[1]


    def batchify2(data_list, max_batch_num=None):
        batchified_data = []
        data_size = len(data_list)
        if not max_batch_num:
            max_batch_num = data_size//batch_size
        for k in range(max_batch_num):
            batchdata = []
            for i in range(3):
                batchdata.append([x[i] for x in data_list[k * batch_size: (k+1) * batch_size]])
            batchified_data.append(batchdata)
        return batchified_data


    # １．一旦３つまとめてリストにしてシャッフルする
    # ２．その後、３つに分ける
    # ３．バッチごとにかためる


    # make a list of batch data
    def make_data_file(max_batch_num=None):
        with open(data_path) as f:
            data_rna = [lines.strip().split(",") for lines in f.readlines()]
            random.shuffle(data_rna)
            data_batch = batchify2(data_rna, max_batch_num)
            # writie_to_cache
            torch.save(data_batch, "./data/batchified_data.pt")


    # make_data_file(10)  # uncomment when data with new batchsize needed
    data_batch2 = torch.load("./data/batchified_data.pt")
    batch_num_total = len(data_batch2)

    ######################################################################
    # Functions to generate input and target sequence
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #


    ######################################################################
    # ``get_batch()`` function generates the input and target sequence for
    # the transformer model. It subdivides the source data into chunks of
    # length ``bptt``. For the language modeling task, the model needs the
    # following words as ``Target``. For example, with a ``bptt`` value of 2,
    # we’d get the following two Variables for ``i`` = 0:
    #
    # .. image:: ../_static/img/transformer_input_target.png
    #
    # It should be noted that the chunks are along dimension 0, consistent
    # with the ``S`` dimension in the Transformer model. The batch dimension
    # ``N`` is along dimension 1.
    #

    bptt = 35
    def get_batch(source, i):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].reshape(-1)
        return data, target


    ######################################################################
    # Initiate an instance
    # --------------------
    #


    ######################################################################
    # The model is set up with the hyperparameter below. The vocab size is
    # equal to the length of the vocab object.
    #

    # ntokens = len(vocab) # the size of vocabulary
    ntokens = 1000
    emsize = 200 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)


    ######################################################################
    # Run the model
    # -------------
    #


    ######################################################################
    # `CrossEntropyLoss <https://pytorch.org/docs/master/nn.html?highlight=crossentropyloss#torch.nn.CrossEntropyLoss>`__
    # is applied to track the loss and
    # `SGD <https://pytorch.org/docs/master/optim.html?highlight=sgd#torch.optim.SGD>`__
    # implements stochastic gradient descent method as the optimizer. The initial
    # learning rate is set to 5.0. `StepLR <https://pytorch.org/docs/master/optim.html?highlight=steplr#torch.optim.lr_scheduler.StepLR>`__ is
    # applied to adjust the learn rate through epochs. During the
    # training, we use
    # `nn.utils.clip_grad_norm\_ <https://pytorch.org/docs/master/nn.html?highlight=nn%20utils%20clip_grad_norm#torch.nn.utils.clip_grad_norm_>`__
    # function to scale all the gradient together to prevent exploding.
    #

    import time
    from DNABERT.examples.run_predict import run_predict

    criterion = nn.CrossEntropyLoss()
    lr = 5.0 # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


    def train(input_data):
        model.train() # Turn on the train mode
        total_loss = 0.
        start_time = time.time()
        src_mask = model.generate_square_subsequent_mask(bptt).to(device)
        for i in range(0, batch_num_total):
            dnabert_inp = input_data[i][1]
            tape_inp = input_data[i][0]  # yet just protein names
            labels = input_data[i][2]

            # DNABERT
            dnabert_out = run_predict(dnabert_inp)
            print("dnabert_out")
            print(dnabert_out)

            # TAPE

            # together


            # # data, targets = get_batch(train_data, i)
            # optimizer.zero_grad()
            # if data.size(0) != bptt:
            #     src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
            # output = model(data, src_mask)
            #
            #
            # loss = criterion(output.view(-1, ntokens), targets)
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            # optimizer.step()
            #
            # total_loss += loss.item()
            # log_interval = 200
            # if batch % log_interval == 0 and batch > 0:
            #     cur_loss = total_loss / log_interval
            #     elapsed = time.time() - start_time
            #     print('| epoch {:3d} | {:5d}/{:5d} batches | '
            #           'lr {:02.2f} | ms/batch {:5.2f} | '
            #           'loss {:5.2f} | ppl {:8.2f}'.format(
            #             epoch, batch, len(train_data) // bptt, scheduler.get_last_lr()[0],
            #             elapsed * 1000 / log_interval,
            #             cur_loss, math.exp(cur_loss)))
            #     total_loss = 0
            #     start_time = time.time()

    def evaluate(eval_model, data_source):
        eval_model.eval() # Turn on the evaluation mode
        total_loss = 0.
        src_mask = model.generate_square_subsequent_mask(bptt).to(device)
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, bptt):
                data, targets = get_batch(data_source, i)
                if data.size(0) != bptt:
                    src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
                output = eval_model(data, src_mask)
                output_flat = output.view(-1, ntokens)
                total_loss += len(data) * criterion(output_flat, targets).item()
        return total_loss / (len(data_source) - 1)

    ######################################################################
    # Loop over epochs. Save the model if the validation loss is the best
    # we've seen so far. Adjust the learning rate after each epoch.

    best_val_loss = float("inf")
    epochs = 3 # The number of epochs
    best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(data_batch2)
        # val_loss = evaluate(model, val_data)
        # print('-' * 89)
        # print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
        #       'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
        #                                  val_loss, math.exp(val_loss)))
        # print('-' * 89)
        #
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     best_model = model
        #
        # scheduler.step()


    ######################################################################
    # Evaluate the model with the test dataset
    # -------------------------------------
    #
    # Apply the best model to check the result with the test dataset.

    # test_loss = evaluate(best_model, test_data)
    # print('=' * 89)
    # print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    #     test_loss, math.exp(test_loss)))
    # print('=' * 89)

if __name__ == "__main__":
    main()