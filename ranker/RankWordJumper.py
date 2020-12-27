from operator import setitem

import numpy
import torch
from torch.distributions import Categorical
from torch.nn import functional

from ranker.data import Dictionary
from ranker.model import Jumper, LSTMencoder

class RankWordJumper(Jumper):

    def __init__(self, args):
        super(RankWordJumper, self).__init__(args)
        self.Q_encoder = LSTMencoder(
            # input_size=args.hidden_size * 2, # without question concatenation
            input_size=300,  # with question concatenation
            hidden_size=300,
            num_layers=1,
            dropout_output=True,
            dropout_rate=args.dropout_rnn
        )

    # 一个 GPU 中的一个 batch 只训练一个问题样本，对应固定个段落，其中只有一个正样本，返回一个问题样本分类的 loss
    # 分类即找出唯一正确的段落样本
    def forward(self, q_seq, q_mask, p_seq, p_mask, labels):
        torch.cuda.empty_cache()
        batch_size, max_q_seq = q_seq.size()
        q_seq_emb = torch.FloatTensor(batch_size, max_q_seq, self.args.glove_hidden_size).zero_()
        for i in range(batch_size):
            for j in range(max_q_seq):
                if q_mask[i][j] == 0:
                    q_seq_emb[i][j].copy_(self.embeddings[q_seq[i][j]])
        q_emb = self.Q_encoder(q_seq_emb, q_mask)

        _, max_seq = p_seq.size()
        seq_emb = torch.FloatTensor(batch_size, max_seq, self.args.glove_hidden_size).zero_()
        for i in range(batch_size):
            for j in range(max_seq):
                if p_mask[i][j] == 0:
                    seq_emb[i][j].copy_(self.embeddings[p_seq[i][j]])

        p_mask = torch.cat((torch.ByteTensor([[0] for i in range(batch_size)]), p_mask), 1)
        max_seq += 1
        seq_emb = torch.cat((torch.unsqueeze(q_emb, 1), seq_emb), 1)
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
        score_softmax = torch.softmax(sn).squeeze()
        focal_loss = self.focal_loss(score_softmax, labels.float())
        reinforce_loss = torch.mean((rewards - baselines) * log_probs)
        mse_loss = self.mse_loss(baselines, rewards)
        loss = focal_loss - reinforce_loss + mse_loss
        return loss

    def inference(self, q_seq, q_mask, p_seq, p_mask):
        batch_size, max_q_seq = q_seq.size()
        q_seq_emb = torch.FloatTensor(batch_size, max_q_seq, self.args.glove_hidden_size).zero_()
        for i in range(batch_size):
            for j in range(max_q_seq):
                if p_mask[i][j] == 0:
                    q_seq_emb[i][j].copy_(self.embeddings[q_seq[i][j]])
        q_emb = self.Q_encoder(q_seq_emb, q_mask)

        _, max_seq = p_seq.size()
        seq_emb = torch.FloatTensor(batch_size, max_seq, self.args.glove_hidden_size).zero_()
        for i in range(batch_size):
            for j in range(max_seq):
                if p_mask[i][j] == 0:
                    seq_emb[i][j].copy_(self.embeddings[p_seq[i][j]])

        p_mask = torch.cat((torch.ByteTensor([[0] for i in range(batch_size)]), p_mask), 1)
        max_seq += 1
        seq_emb = torch.cat((torch.unsqueeze(q_emb, 1), seq_emb), 1)
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
        scores = torch.softmax(self.score_net(data)).squeeze()
        return scores
