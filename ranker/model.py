import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from operator import setitem
from torch.distributions import Categorical
import numpy as np


class Ranker(nn.Module):
    def __init__(self, args):
        super(Ranker, self).__init__()
        self.args = args
        self.embeddings = None
        self.lstm_enc = LSTMencoder(
            # input_size=args.hidden_size * 2, # without question concatenation
            input_size=4096 * 2,  # with question concatenation
            hidden_size=4096,
            num_layers=1,
            dropout_output=True,
            dropout_rate=args.dropout_rnn
        )
        self.score_net = nn.Sequential(nn.Linear(in_features=args.hidden_size * 4, out_features=1024),
                                       nn.Dropout(0.1),
                                       nn.LeakyReLU(),
                                       nn.Linear(in_features=1024, out_features=1))
        self.focal_loss = FocalLoss(alpha=1, gamma=2, logits=False, reduce=True)

    def set_emb(self, embeddings):
        self.embeddings = torch.FloatTensor(embeddings).detach()

    def forward(self, questions, contexts, masks, labels):
        '''
        :param Q: q_num
        :param A: para_num * max_sen_length
        :param A_mask: para_num * max_sen_length
        :return: scores
        '''
        batch_size, max_seq = contexts.size()
        lengths = masks.data.eq(0).long().sum(1).squeeze()
        Q_emb = torch.FloatTensor(batch_size, self.args.hidden_size).zero_()  # ([num_q,dim])
        for idx in range(batch_size):
            Q_emb[idx].copy_(self.embeddings[questions[idx]])
        A_emb = torch.FloatTensor(batch_size, max_seq,
                                  self.args.hidden_size).zero_()  # ([batchsize, max_length, dim])
        input_emb = torch.FloatTensor(batch_size, max_seq, self.args.hidden_size * 2).zero_()  # 将问题emb拼到文章每句话下面，送入LSTM

        #
        for idx in range(batch_size):
            for l in range(max_seq):
                A_emb[idx][l].copy_(self.embeddings[contexts[idx][l]])
                if masks[idx][l] == 0:
                    input_emb[idx][l].copy_(torch.cat((A_emb[idx][l], Q_emb[idx])))
        Q_emb = Q_emb.to(self.args.device).detach()
        input_emb = input_emb.to(self.args.device).detach()
        output = self.lstm_enc(input_emb, masks)
        # #mean
        P_emb = torch.sum(output, dim=1) / lengths.view(len(lengths), 1).float()

        data = torch.cat((Q_emb, P_emb), 1)
        data = torch.cat((data, Q_emb - P_emb), 1)
        data = torch.cat((data, torch.mul(Q_emb, P_emb)), 1)
        # print(data.size())
        scores = torch.sigmoid(self.score_net(data)).squeeze()
        loss = self.focal_loss(scores, labels.float())
        torch.cuda.empty_cache()
        return loss

    def inference(self, questions, contexts, masks):
        batch_size, max_seq = contexts.size()
        lengths = masks.data.eq(0).long().sum(1).squeeze().to(self.args.device)
        Q_emb = torch.FloatTensor(batch_size, self.args.hidden_size).zero_()  # ([num_q,dim])
        for idx in range(batch_size):
            Q_emb[idx].copy_(self.embeddings[questions[idx]])
        A_emb = torch.FloatTensor(batch_size, max_seq,
                                  self.args.hidden_size).zero_()  # ([batchsize, max_length, dim])
        input_emb = torch.FloatTensor(batch_size, max_seq, self.args.hidden_size * 2).zero_()  # 将问题emb拼到文章每句话下面，送入LSTM

        #
        for idx in range(batch_size):
            for l in range(max_seq):
                A_emb[idx][l].copy_(self.embeddings[contexts[idx][l]])
                if masks[idx][l] == 0:
                    input_emb[idx][l].copy_(torch.cat((A_emb[idx][l], Q_emb[idx])))
        Q_emb = Q_emb.to(self.args.device).detach()
        input_emb = input_emb.to(self.args.device).detach()
        masks = masks.to(self.args.device).detach()
        output = self.lstm_enc(input_emb, masks)
        # #mean
        P_emb = torch.sum(output, dim=1) / lengths.view(len(lengths), 1).float()

        data = torch.cat((Q_emb, P_emb), 1)
        data = torch.cat((data, Q_emb - P_emb), 1)
        data = torch.cat((data, torch.mul(Q_emb, P_emb)), 1)
        # print(data.size())
        scores = torch.sigmoid(self.score_net(data))
        return scores

class Jumper(nn.Module):
    def __init__(self, args):
        super(Jumper, self).__init__()
        self.lstm = nn.LSTMCell(args.hidden_size * 2, args.hidden_size)
        self.linear = nn.Linear(args.hidden_size, args.K + 1)
        self.baseline = nn.Linear(args.hidden_size, 1)
        self.mse_loss = nn.MSELoss()
        self.args = args
        self.embeddings = None
        self.score_net = nn.Sequential(nn.Linear(in_features=args.hidden_size * 4, out_features=256),
                                       nn.Dropout(0.1),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Linear(in_features=256, out_features=1))
        self.focal_loss = FocalLoss(alpha=1, gamma=2, logits=False, reduce=True)

    def forward(self, questions, contexts, masks, labels):
        torch.cuda.empty_cache()
        batch_size, max_seq = contexts.size()
        lengths = masks.data.eq(0).long().sum(1).squeeze()
        Q_emb = torch.FloatTensor(batch_size, self.args.hidden_size).zero_()  # ([num_q,dim])
        for idx in range(batch_size):
            Q_emb[idx].copy_(self.embeddings[questions[idx]])
        A_emb = torch.FloatTensor(batch_size, max_seq,
                                  self.args.hidden_size).zero_()  # ([batchsize, max_length, dim])
        input_embs = torch.FloatTensor(batch_size, max_seq, self.args.hidden_size * 2).zero_()  # 将问题emb拼到文章每句话前面，送入LSTM

        #
        for idx in range(batch_size):
            for l in range(max_seq):
                A_emb[idx][l].copy_(self.embeddings[contexts[idx][l]])
                if masks[idx][l] == 0:
                    input_embs[idx][l].copy_(torch.cat((A_emb[idx][l], Q_emb[idx])))

        input_embs = input_embs.transpose(0, 1)
        state = None
        rows = torch.LongTensor(batch_size).zero_().to(self.args.device)  # 每个段落的第n个句子
        columns = torch.LongTensor(range(batch_size)).to(self.args.device)
        log_probs = []
        baselines = []
        jump_masks = []
        hiddens = [None] * batch_size
        reward_a = 0
        last_rows = lengths - 1
        # jump
        for n in range(self.args.N):
            # for _ in range(self.args.R):
            #     feed_previous = rows >= lengths
            #     rows = feed_previous.long() * last_rows + (1 - feed_previous.long()) * rows
            #     emb = input_embs[rows, columns].to(self.args.device).detach()
            #     if feed_previous.any():
            #         for i,v in enumerate(feed_previous):
            #             if v == 1 and hiddens[i] is None:
            #                 hiddens[i] = h[i,:].cpu().detach().numpy().tolist()
            #         # [setitem(hiddens, i, h[:, i, :]) for i, v in enumerate(feed_previous) if v == 1]
            #     if state is None:
            #         h, state = self.lstm(emb)
            #     else:
            #         h, state = self.lstm(emb, (h,state))
            #     reward_a = [i+1 for i in reward_a]
            #     rows = rows + 1
            #     if (rows >= lengths).all():
            #         break


            feed_previous = rows >= lengths
            rows = feed_previous.long() * last_rows + (1 - feed_previous.long()) * rows
            if feed_previous.any():
                for i, v in enumerate(feed_previous):
                    if v == 1 and hiddens[i] is None:
                        hiddens[i] = h[i, :].cpu().detach().numpy().tolist()
            # print(input_embs[rows,columns])
            emb = input_embs[rows, columns].to(self.args.device).detach()
            torch.cuda.empty_cache()
            if state is None:
                h, state = self.lstm(emb)
            else:
                h, state = self.lstm(emb, (h, state))
            reward_a +=1
            rows = rows + 1
            p = F.softmax(self.linear(h.squeeze(0)), dim=1)
            m = Categorical(p)
            jump = m.sample()
            log_prob = m.log_prob(jump)
            log_probs.append(log_prob[:, None])
            jump_masks.append(feed_previous[:, None])
            baselines.append(self.baseline(h.squeeze(0)))
            # is_stopping = (jump.data == 0).long()
            # row1 = is_stopping * (rows ) + (1 - is_stopping) * (rows + jump.data )
            rows = rows+jump.data
            if (rows >= lengths).all():
                break

        if any(x is None for x in hiddens):
            [setitem(hiddens, i, h[ i, :].cpu().detach().numpy().tolist()) for i, v in enumerate(hiddens) if v is None]
        P_emb = torch.FloatTensor(hiddens).to(self.args.device).detach()
        Q_emb = Q_emb.to(self.args.device).detach()
        # P_emb = torch.sum(output, dim=1) / lengths.view(len(lengths), 1).float()
        data = torch.cat((Q_emb, P_emb), 1)
        data = torch.cat((data, Q_emb - P_emb), 1)
        data = torch.cat((data, torch.mul(Q_emb, P_emb)), 1)
        # print(data.size())
        scores = torch.sigmoid(self.score_net(data)).squeeze()
        log_probs = torch.cat(log_probs, dim=1)
        baselines = torch.cat(baselines, dim=1)

        reward_r = scores.gt(0.5).eq(labels.data).float().unsqueeze(1)
        # a = correct.masked_fill_(correct == 0., -1)
        # reward_r = Variable(correct.masked_fill_(correct == 0., -1)).unsqueeze(1)
        reward_a = torch.Tensor([-reward_a/max_seq] * batch_size).unsqueeze(1).to(self.args.device)
        rewards = reward_a + reward_r
        rewards = rewards.repeat(1,baselines.size(1))
        # filling with 0
        mask = torch.cat(jump_masks, dim=1)
        log_probs.masked_fill_(mask, 0)
        baselines.masked_fill_(mask, 0)
        rewards.masked_fill_(mask, 0)
        torch.cuda.empty_cache()
        focal_loss = self.focal_loss(scores, labels.float())
        reinforce_loss = torch.mean((rewards - baselines) * log_probs)
        mse_loss = self.mse_loss(baselines, rewards)
        loss  = focal_loss - reinforce_loss + mse_loss
        return loss

    def set_emb(self, embeddings):
        self.embeddings = torch.FloatTensor(embeddings).detach()

    def inference(self, questions, contexts, masks):
        batch_size, max_seq = contexts.size()
        lengths = masks.data.eq(0).long().sum(1).squeeze().to(self.args.device)
        Q_emb = torch.FloatTensor(batch_size, self.args.hidden_size).zero_()  # ([num_q,dim])
        for idx in range(batch_size):
            Q_emb[idx].copy_(self.embeddings[questions[idx]])
        A_emb = torch.FloatTensor(batch_size, max_seq,
                                  self.args.hidden_size).zero_()  # ([batchsize, max_length, dim])
        input_embs = torch.FloatTensor(batch_size, max_seq, self.args.hidden_size * 2).zero_()  # 将问题emb拼到文章每句话下面，送入LSTM

        #
        for idx in range(batch_size):
            for l in range(max_seq):
                A_emb[idx][l].copy_(self.embeddings[contexts[idx][l]])
                if masks[idx][l] == 0:
                    input_embs[idx][l].copy_(torch.cat((A_emb[idx][l], Q_emb[idx])))
        input_embs = input_embs.transpose(0, 1)
        state = None
        rows = torch.LongTensor(batch_size).zero_().to(self.args.device)  # 每个段落的第n个句子
        columns = torch.LongTensor(range(batch_size)).to(self.args.device)
        log_probs = []
        baselines = []
        jump_masks = []
        hiddens = [None] * batch_size
        last_rows = lengths - 1
        # jump
        for n in range(self.args.N):
            for _ in range(self.args.R):
                feed_previous = rows >= lengths
                rows = feed_previous.long() * last_rows + (1 - feed_previous.long()) * rows
                emb = input_embs[rows, columns].to(self.args.device).detach()
                if feed_previous.any():
                    for i, v in enumerate(feed_previous):
                        if v == 1 and hiddens[i] is None:
                            hiddens[i] = h[i, :].cpu().detach().numpy().tolist()
                    # [setitem(hiddens, i, h[:, i, :]) for i, v in enumerate(feed_previous) if v == 1]
                if state is None:
                    h, state = self.lstm(emb)
                else:
                    h, state = self.lstm(emb, (h, state))
                rows = rows + 1
                if (rows >= lengths).all():
                    break
            feed_previous = rows >= lengths
            # TODO: replace where function when it is added
            rows = feed_previous.long() * last_rows + (1 - feed_previous.long()) * rows
            if feed_previous.any():
                for i, v in enumerate(feed_previous):
                    if v == 1 and hiddens[i] is None:
                        hiddens[i] = h[i, :].cpu().detach().numpy().tolist()
            # print(input_embs[rows,columns])
            emb = input_embs[rows, columns].to(self.args.device).detach()
            torch.cuda.empty_cache()
            h, state = self.lstm(emb, (h, state))
            p = F.softmax(self.linear(h.squeeze(0)), dim=1)
            m = Categorical(p)
            jump = m.sample()
            log_prob = m.log_prob(jump)
            log_probs.append(log_prob[:, None])
            jump_masks.append(feed_previous[:, None])
            baselines.append(self.baseline(h.squeeze(0)))
            is_stopping = (jump.data == 0).long()
            rows = is_stopping * (rows + 1) + (1 - is_stopping) * (rows + jump.data + 1)
            if (rows >= lengths).all():
                break

        if any(x is None for x in hiddens):
            [setitem(hiddens, i, h[i, :].cpu().detach().numpy().tolist()) for i, v in enumerate(hiddens) if v is None]
        P_emb = torch.FloatTensor(hiddens).to(self.args.device).detach()
        Q_emb = Q_emb.to(self.args.device).detach()
        # P_emb = torch.sum(output, dim=1) / lengths.view(len(lengths), 1).float()
        data = torch.cat((Q_emb, P_emb), 1)
        data = torch.cat((data, Q_emb - P_emb), 1)
        data = torch.cat((data, torch.mul(Q_emb, P_emb)), 1)
        # print(data.size())
        scores = torch.sigmoid(self.score_net(data))
        return scores

class LSTMencoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0,
                 dropout_output=False, padding=False):
        super(LSTMencoder, self).__init__()
        self.padding = padding
        self.dropout_rate = dropout_rate
        self.dropout_output = dropout_output
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1)

    def forward(self, x, x_mask):
        """Slower (significantly), but more precise, encoding that handles
        padding.
        """
        # Compute sorted sequence lengths
        # Compute sorted sequence lengths
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)

        # Sort x
        x = x.index_select(0, idx_sort)
        x = x.transpose(0, 1)
        # Pack it up
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        if self.dropout_rate > 0:
            dropout_input = F.dropout(rnn_input.data, p=self.dropout_rate, training=self.training)
            rnn_input = nn.utils.rnn.PackedSequence(dropout_input, rnn_input.batch_sizes)
        output, hidden_state = self.lstm(rnn_input)

        # Unpack everything
        output = nn.utils.rnn.pad_packed_sequence(output)[0]

        # Transpose and unsort
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)

        # Pad up to original batch sequence length
        if output.size(1) != x_mask.size(1):
            padding = torch.zeros(output.size(0),
                                  x_mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, Variable(padding)], 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.4, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.sum(F_loss)
        else:
            return F_loss
