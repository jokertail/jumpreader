from operator import setitem

import numpy
import torch
from torch.distributions import Categorical
from torch.nn import functional

from ranker.data import Dictionary
from ranker.model import Jumper, LSTMencoder


class WordJumper(Jumper):

    def __init__(self, args):
        super(WordJumper, self).__init__(args)
        self.Q_encoder = LSTMencoder(
            # input_size=args.hidden_size * 2, # without question concatenation
            input_size=300,  # with question concatenation
            hidden_size=300,
            num_layers=1,
            dropout_output=True,
            dropout_rate=args.dropout_rnn
        )
        self.word_dict = None
        self.glove_dim = args.glove_dim
        self.glove_path = args.glove_path

    def forward(self, q_seq, sequences, masks, mask_q, labels):
        torch.cuda.empty_cache()
        batch_size, max_q_seq = q_seq.size()
        q_seq_emb = torch.FloatTensor(batch_size, max_q_seq, self.args.glove_hidden_size).zero_()
        for i in range(batch_size):
            for j in range(max_q_seq):
                if masks[i][j] == 0:
                    q_seq_emb[i][j].copy_(self.embeddings[q_seq[i][j]])
        q_emb = self.Q_encoder(q_seq_emb, mask_q)

        _, max_seq = sequences.size()
        seq_emb = torch.FloatTensor(batch_size, max_seq, self.args.glove_hidden_size).zero_()
        for i in range(batch_size):
            for j in range(max_seq):
                if masks[i][j] == 0:
                    seq_emb[i][j].copy_(self.embeddings[sequences[i][j]])

        masks = torch.cat((torch.ByteTensor([[0] for i in range(batch_size)]), masks), 1)
        max_seq += 1
        seq_emb = torch.cat((torch.unsqueeze(q_emb, 1), seq_emb), 1)
        lengths = masks.data.eq(0).long().sum(1).squeeze()

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
        scores = torch.sigmoid(self.score_net(data)).squeeze()
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
        focal_loss = self.focal_loss(scores, labels.float())
        reinforce_loss = torch.mean((rewards - baselines) * log_probs)
        mse_loss = self.mse_loss(baselines, rewards)
        loss = focal_loss - reinforce_loss + mse_loss
        return loss

    def set_glove_path(self, glove_path):
        self.glove_path = glove_path

    def generate_word_dict(self, sentences, tokenize=True):
        word_dict = Dictionary()
        if tokenize:
            from nltk.tokenize import word_tokenize

        sentences = [s.split() if not tokenize else word_tokenize(s)
                     for s in sentences]

        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict.add(word)

        self.word_dict = word_dict

    def generate_word_embbedding(self, word_dict):
        assert hasattr(self, 'glove_path'), \
            'warning : you need to set_glove_path(glove_path)'
        # create word_vec with glove vectors
        word_emb_np = numpy.zeros((len(word_dict), self.glove_dim), dtype=float)
        count = 0
        with open(self.glove_path) as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word_dict[word] > 1:
                    word_emb_np[word_dict[word]] = numpy.fromstring(vec, sep=' ')
                    count += 1
        self.embeddings = torch.from_numpy(word_emb_np)
