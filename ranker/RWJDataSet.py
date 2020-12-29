import numpy as np
import torch
import logging
import unicodedata
from tqdm import tqdm
from torch.utils.data import Dataset

from ranker.data import Dictionary


class RWJDataSet(Dataset):

    def __init__(self, exs, word_dict, f_num=5, max_t_sample_num=4, mode='train'):
        from nltk.tokenize import word_tokenize
        self.word_dict = word_dict
        self.examples = []
        self.mode = mode

        # 一组样本中 正样本永远只有一个
        self.size = (1, f_num)

        for ex in tqdm(exs,desc="forming dataset "):
            question = ex["question"]
            q_token = word_tokenize(question)
            ex_token = {
                "q_token": q_token,
                "ct_token": [],
                "cf_token": []
            }

            for p in ex["ct"]:
                p_token = word_tokenize(p)
                ex_token["ct_token"].append(p_token)

            for p in ex["cf"]:
                p_token = word_tokenize(p)
                ex_token["cf_token"].append(p_token)

            ex_token["cf_token"].sort(key=len)

            if self.mode == "test":
                self.examples.append(ex_token)
                continue

            sample_num = min(max_t_sample_num, len(ex_token["ct_token"]))
            if sample_num == 0 or len(ex_token["cf_token"]) < sample_num+f_num:
                continue


            stride = (int)((len(ex_token["cf_token"]) - f_num) / sample_num)

            for i in range(min(max_t_sample_num, len(ex_token["ct_token"]))):
                fb = stride * i
                self.examples.append({
                    "q_token": q_token,
                    "ct_token": ex_token["ct_token"][i],
                    "cf_token": ex_token["cf_token"][fb:fb + f_num]
                })

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

    def vectorize(self, ex):
        question = [[self.word_dict[i] for i in ex["q_token"]]]
        context = [[self.word_dict[i] for i in ex["ct_token"]]]
        for cf in ex["cf_token"]:
            context.append([self.word_dict[i] for i in cf])
            question.append(question[0])

        # 一个训练样本中段落样本数是固定的
        assert len(context) == sum(self.size)
        # 有且只有第一个段落样本是正样本
        label = [1] + [0] * self.size[1]

        return question, context, label

    # 一个 batch 中的一个 GPU 训练一个样本，batch_size 始终等于使用 GPU 的数量
    @staticmethod
    def batchify(batch, mode='train'):
        batch_size = len(batch)
        # batch_size * size 个问题，对应到 GPU 中只有 size 个重复的问题
        questions = []
        # batch_size * size 个段落，对应到 GPU 中只有一个问题的段落
        contexts = []
        # batch_size 个标签，对应到 GPU 中只有一个问题的标签
        labels = []
        for ex in batch:
            questions += ex[0]
            contexts += ex[1]
            labels += ex[2]

        size = len(questions)
        max_q_len = max([len(q) for q in questions])
        max_p_len = max([len(p) for p in contexts])

        q_seq = torch.LongTensor(size, max_q_len).zero_()
        q_mask = torch.ByteTensor(size, max_q_len).fill_(1)
        for i, q in enumerate(questions):
            q_seq[i, :len(q)].copy_(torch.LongTensor(q))
            q_mask[i, :len(q)].fill_(0)

        p_seq = torch.LongTensor(size, max_p_len).zero_()
        p_mask = torch.ByteTensor(size, max_p_len).fill_(1)
        for i, p in enumerate(contexts):
            p_seq[i, :len(p)].copy_(torch.LongTensor(p))
            p_mask[i, :len(p)].fill_(0)

        labels = torch.ByteTensor(labels)

        return q_seq, q_mask, p_seq, p_mask, labels
