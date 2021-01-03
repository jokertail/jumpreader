from operator import setitem

import numpy
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional

from ranker.data import Dictionary
from ranker.model import Jumper, LSTMencoder, FocalLoss


class RankWordJumper(nn.Module):

    def __init__(self, args):
        super(RankWordJumper, self).__init__()
        self.lstm = nn.LSTMCell(args.glove_hidden_size, args.glove_hidden_size)
        self.linear = nn.Linear(args.glove_hidden_size, args.K + 1)
        self.baseline = nn.Linear(args.glove_hidden_size, 1)
        self.mse_loss = nn.MSELoss()
        self.args = args
        self.embeddings = None
        self.score_net = nn.Sequential(nn.Linear(in_features=args.glove_hidden_size * 4, out_features=256),
                                       nn.Dropout(0.1),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Linear(in_features=256, out_features=1))
        self.focal_loss = FocalLoss(alpha=1, gamma=2, logits=False, reduce=True)
        self.Q_encoder = LSTMencoder(
            # input_size=args.hidden_size * 2, # without question concatenation
            input_size=args.glove_hidden_size,  # with question concatenation
            hidden_size=args.glove_hidden_size,
            num_layers=1,
            dropout_output=True,
            dropout_rate=args.dropout_rnn
        )

    # 一个 GPU 中的一个 batch 只训练一个问题样本，对应固定个段落，其中只有一个正样本，返回一个问题样本分类的 loss
    # 分类即找出唯一正确的段落样本
    def forward(self, q_seq, q_mask, p_seq, p_mask, labels, q_length, group_size):
        torch.cuda.empty_cache()
        batch_size, max_q_seq = q_seq.size()
        q_seq_emb = torch.FloatTensor(batch_size, max_q_seq, self.args.glove_hidden_size).zero_()
        for i in range(batch_size):
            for j in range(max_q_seq):
                if q_mask[i][j] == 0:
                    q_seq_emb[i][j].copy_(self.embeddings[q_seq[i][j]])
        tem_q_emb = self.Q_encoder(q_seq_emb.cuda(), q_mask)
        q_emb = torch.FloatTensor(batch_size, self.args.glove_hidden_size).zero_().cuda()
        for i in range(batch_size):
            q_emb[i].copy_(tem_q_emb[i][q_length[i]-1])

        _, max_seq = p_seq.size()
        seq_emb = torch.FloatTensor(batch_size, max_seq, self.args.glove_hidden_size).zero_()
        for i in range(batch_size):
            for j in range(max_seq):
                if p_mask[i][j] == 0:
                    seq_emb[i][j].copy_(self.embeddings[p_seq[i][j]])

        p_mask = torch.cat((torch.ByteTensor([[0]] * batch_size).cuda(), p_mask), 1)
        max_seq += 1
        seq_emb = torch.cat((torch.unsqueeze(q_emb, 1), seq_emb.cuda()), 1)
        lengths = p_mask.data.eq(0).long().sum(1).squeeze()

        seq_emb = seq_emb.transpose(0, 1)
        state = None
        rows = torch.LongTensor(batch_size).zero_().to(self.args.device)
        columns = torch.LongTensor(range(batch_size)).to(self.args.device)
        log_probs = []
        baselines = []
        jump_masks = []
        hiddens = [None] * batch_size
        reward_a = 0
        last_rows = lengths - 1
        for n in range(self.args.N):
            feed_previous = rows >= lengths
            rows = feed_previous.long() * last_rows + (1 - feed_previous.long()) * rows
            if feed_previous.any():
                for i, v in enumerate(feed_previous):
                    if v == 1 and hiddens[i] is None:
                        hiddens[i] = h[i, :].cpu().detach().numpy().tolist()
            # print(input_embs[rows,columns])
            emb = seq_emb[rows, columns].to(self.args.device).detach()
            torch.cuda.empty_cache()
            if state is None:
                h, state = self.lstm(emb)
            else:
                h, state = self.lstm(emb, (h, state))
            reward_a += 1
            rows = rows + 1
            p = functional.softmax(self.linear(h.squeeze(0)), dim=1)
            m = Categorical(p)
            jump = m.sample()
            log_prob = m.log_prob(jump)
            log_probs.append(log_prob[:, None])
            jump_masks.append(feed_previous[:, None])
            baselines.append(self.baseline(h.squeeze(0)))
            # is_stopping = (jump.data == 0).long()
            # row1 = is_stopping * (rows ) + (1 - is_stopping) * (rows + jump.data )
            rows = rows + jump.data
            if (rows >= lengths).all():
                break

        if any(x is None for x in hiddens):
            [setitem(hiddens, i, h[i, :].cpu().detach().numpy().tolist()) for i, v in enumerate(hiddens) if v is None]

        p_emb = torch.FloatTensor(hiddens).to(self.args.device).detach()
        q_emb = q_emb.to(self.args.device).detach()
        # P_emb = torch.sum(output, dim=1) / lengths.view(len(lengths), 1).float()
        data = torch.cat((q_emb, p_emb), 1)
        data = torch.cat((data, q_emb - p_emb), 1)
        data = torch.cat((data, torch.mul(q_emb, p_emb)), 1)
        # print(data.size())
        sn = self.score_net(data)
        scores = torch.sigmoid(sn).squeeze()
        log_probs = torch.cat(log_probs, dim=1)
        baselines = torch.cat(baselines, dim=1)

        reward_r = scores.gt(0.5).eq(labels.data).float().unsqueeze(1)
        # a = correct.masked_fill_(correct == 0., -1)
        # reward_r = Variable(correct.masked_fill_(correct == 0., -1)).unsqueeze(1)
        reward_a = torch.Tensor([-reward_a / max_seq] * batch_size).unsqueeze(1).to(self.args.device)
        rewards = reward_a + reward_r
        rewards = rewards.repeat(1, baselines.size(1))
        # filling with 0
        mask = torch.cat(jump_masks, dim=1)
        log_probs.masked_fill_(mask, 0)
        baselines.masked_fill_(mask, 0)
        rewards.masked_fill_(mask, 0)
        torch.cuda.empty_cache()
        loss = torch.FloatTensor(1).zero_().cuda()
        margin_ranking_loss = torch.FloatTensor(1).zero_().cuda()
        target = torch.FloatTensor(1).fill_(1).cuda()
        for i in range((int)(batch_size/group_size)):
            score_softmax = torch.softmax(sn[i*group_size:(i+1)*group_size], 0).squeeze()
            margin_ranking_loss.fill_(0)
            for j in range(group_size-1):
                margin_ranking_loss += functional.margin_ranking_loss(sn[i*group_size], sn[i*group_size+j+1], target)
            # focal_loss += self.focal_loss(score_softmax, labels[i*group_size:(i+1)*group_size].float())
            reinforce_loss = torch.mean((rewards - baselines) * log_probs)
            mse_loss = self.mse_loss(baselines, rewards)
            loss += margin_ranking_loss - reinforce_loss + mse_loss
        return loss

    def set_emb(self, embeddings):
        self.embeddings = torch.FloatTensor(embeddings).detach()

    def inference(self, q_seq, q_mask, p_seq, p_mask, q_length):
        torch.cuda.empty_cache()
        batch_size, max_q_seq = q_seq.size()
        q_seq_emb = torch.FloatTensor(batch_size, max_q_seq, self.args.glove_hidden_size).zero_()
        for i in range(batch_size):
            for j in range(max_q_seq):
                if q_mask[i][j] == 0:
                    q_seq_emb[i][j].copy_(self.embeddings[q_seq[i][j]])
        tem_q_emb = self.Q_encoder(q_seq_emb.cuda(), q_mask.cuda())
        q_emb = torch.FloatTensor(batch_size, self.args.glove_hidden_size).zero_().cuda()
        for i in range(batch_size):
            q_emb[i].copy_(tem_q_emb[i][q_length[i]-1])

        _, max_seq = p_seq.size()
        seq_emb = torch.FloatTensor(batch_size, max_seq, self.args.glove_hidden_size).zero_()
        for i in range(batch_size):
            for j in range(max_seq):
                if p_mask[i][j] == 0:
                    seq_emb[i][j].copy_(self.embeddings[p_seq[i][j]])

        p_mask = torch.cat((torch.ByteTensor([[0]] * batch_size).cuda(), p_mask.cuda()), 1)
        max_seq += 1
        seq_emb = torch.cat((torch.unsqueeze(q_emb, 1), seq_emb.cuda()), 1)
        lengths = p_mask.data.eq(0).long().sum(1).squeeze()

        seq_emb = seq_emb.transpose(0, 1)
        state = None
        rows = torch.LongTensor(batch_size).zero_().to(self.args.device)
        columns = torch.LongTensor(range(batch_size)).to(self.args.device)
        log_probs = []
        baselines = []
        jump_masks = []
        hiddens = [None] * batch_size
        reward_a = 0
        last_rows = lengths - 1
        for n in range(self.args.N):
            feed_previous = rows >= lengths
            rows = feed_previous.long() * last_rows + (1 - feed_previous.long()) * rows
            if feed_previous.any():
                for i, v in enumerate(feed_previous):
                    if v == 1 and hiddens[i] is None:
                        hiddens[i] = h[i, :].cpu().detach().numpy().tolist()
            # print(input_embs[rows,columns])
            emb = seq_emb[rows, columns].to(self.args.device).detach()
            torch.cuda.empty_cache()
            if state is None:
                h, state = self.lstm(emb)
            else:
                h, state = self.lstm(emb, (h, state))
            reward_a += 1
            rows = rows + 1
            p = functional.softmax(self.linear(h.squeeze(0)), dim=1)
            m = Categorical(p)
            jump = m.sample()
            log_prob = m.log_prob(jump)
            log_probs.append(log_prob[:, None])
            jump_masks.append(feed_previous[:, None])
            baselines.append(self.baseline(h.squeeze(0)))
            # is_stopping = (jump.data == 0).long()
            # row1 = is_stopping * (rows ) + (1 - is_stopping) * (rows + jump.data )
            rows = rows + jump.data
            if (rows >= lengths).all():
                break

        if any(x is None for x in hiddens):
            [setitem(hiddens, i, h[i, :].cpu().detach().numpy().tolist()) for i, v in enumerate(hiddens) if v is None]

        p_emb = torch.FloatTensor(hiddens).to(self.args.device).detach()
        q_emb = q_emb.to(self.args.device).detach()
        # P_emb = torch.sum(output, dim=1) / lengths.view(len(lengths), 1).float()
        data = torch.cat((q_emb, p_emb), 1)
        data = torch.cat((data, q_emb - p_emb), 1)
        data = torch.cat((data, torch.mul(q_emb, p_emb)), 1)
        # print(data.size())
        sn = self.score_net(data)
        return sn
