import numpy as np
import torch
import logging
import unicodedata

from torch.utils.data import Dataset

from ranker.data import Dictionary


class WJDataSet(Dataset):

    def __init__(self, exs, word_dict, mode='train'):
        from nltk.tokenize import word_tokenize
        self.word_dict = word_dict
        self.examples = []
        self.mode = mode

        for ex in exs:
            question = ex["question"];
            q_token = word_tokenize(question)
            for w in q_token:
                if w not in self.word_dict:
                    self.word_dict.add(w)

            for p in ex["ct"]:
                p_token = word_tokenize(p)
                self.examples.add({
                    "q_token": q_token,
                    "p_token": p_token,
                    "label": 1
                })
                for w in p_token:
                    if w not in self.word_dict:
                        self.word_dict.add(w)

            for p in ex["cf"]:
                p_token = word_tokenize(p)
                self.examples.add({
                    "q_token": q_token,
                    "p_token": p_token,
                    "label": 0
                })
                for w in p_token:
                    if w not in self.word_dict:
                        self.word_dict.add(w)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        try:
            outputs = self.vectorize(self.examples[index])
            # print(len(self.examples),index)
            return outputs
        except IndexError:
            print(len(self.examples), index)

        # if self.mode == 'train':
        #     return vectorize(self.examples[index], self.sentence_dict)
        # elif self.mode == 'dev':
        #     return vectorize_dev(self.examples[index], self.sentence_dict)
        # elif self.mode == 'test':
        #     return vectorize_test(self.examples[index], self.sentence_dict)

    def lengths(self):
        return [(len(ex['question']['str']), len(ex['ct']['str']), len(ex['cf'][:]['str']))
                for ex in self.examples]

    def get_word_dict(self):
        return self.word_dict

    def vectorize(self, ex):
        question = torch.LongTensor([self.word_dict[i] for i in ex["q_token"]])
        context = torch.LongTensor([self.word_dict[i] for i in ex["p_token"]])
        label = ex["label"]

        return question, context, label

    @staticmethod
    def batchify(batch, mode='train'):
        batch_size = len(batch)
        questions = []
        contexts = []
        labels = []
        for ex in batch:
            questions += [ex[0]]
            contexts += [ex[1]]
            labels += [ex[2]]

        max_q_len = max([len(q) for q in questions])
        max_p_len = max([len(p) for p in contexts])

        q_seq = torch.LongTensor(batch_size, max_q_len).zero_()
        q_mask = torch.ByteTensor(batch_size, max_q_len).fill_(1)
        for i, q in enumerate(questions):
            q_seq[i, :len(q)].copy_(torch.LongTensor(q))
            q_mask[i, :len(q)].fill_(0)

        p_seq = torch.LongTensor(batch_size, max_p_len).zero_()
        p_mask = torch.LongTensor(batch_size, max_p_len).fill_(1)
        for i, p in enumerate(contexts):
            p_seq[i, :len(p)].copy_(torch.LongTensor(p))
            p_mask[i, len(p)].fill_(0)

        labels = torch.ByteTensor(labels)

        return q_seq, q_mask, p_seq, p_mask, labels
