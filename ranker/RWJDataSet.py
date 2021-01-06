import numpy as np
import torch
import logging
import unicodedata
from tqdm import tqdm
from torch.utils.data import Dataset

from ranker.data import Dictionary


class RWJDataSet(Dataset):

    def __init__(self, exs, word_dict, f_num=5, max_t_sample_num=4, max_train_ex_num=20000, mode='train'):
        from nltk.tokenize import word_tokenize
        self.word_dict = word_dict
        self.examples = []
        self.mode = mode

        def get_c_length(x):
            return len(x[0])

        if self.mode == "test":
            for ex in tqdm(exs, desc="forming dataset "):
                question = ex["question"]
                q_token = word_tokenize(question)
                ex_token = {
                    "q_token": q_token,
                    "c_token": [],
                }
                for p in ex["ct"]:
                    p_token = word_tokenize(p)
                    ex_token["c_token"].append((p_token, 1))

                for p in ex["cf"]:
                    p_token = word_tokenize(p)
                    ex_token["c_token"].append((p_token, 0))

                ex_token["c_token"].sort(key=get_c_length)
                self.examples.append(ex_token)
            return

        self.max_train_ex_num = max_train_ex_num

        # 一组样本中 正样本永远只有一个
        self.size = (1, f_num)

        count = 0
        for ex in tqdm(exs, desc="forming dataset "):
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

            sample_num = min(max_t_sample_num, len(ex_token["ct_token"]))
            if sample_num == 0 or len(ex_token["cf_token"]) < sample_num + f_num:
                continue

            stride = (int)((len(ex_token["cf_token"]) - f_num) / sample_num)

            for i in range(min(max_t_sample_num, len(ex_token["ct_token"]))):
                fb = stride * i
                self.examples.append({
                    "q_token": q_token,
                    "ct_token": ex_token["ct_token"][i],
                    "cf_token": ex_token["cf_token"][fb:fb + f_num]
                })
                count += 1
                # if count == self.max_train_ex_num:
                #     return

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
        if self.mode == "test":
            q = [self.word_dict[i] for i in ex["q_token"]]
            question = []
            context = []
            label = []
            for c in ex["c_token"]:
                question.append(q)
                context.append([self.word_dict[i] for i in c[0]])
                label.append(c[1])
            return question, context, label

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

        group_size = len(batch[0][0])

        q_length = []
        for ex in batch:
            q_length += [len(ex[0][0])] * len(ex[0])
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

        return q_seq, q_mask, p_seq, p_mask, labels, q_length, group_size

    # 对于测试的 batchify，一个 batch 只取一个问题的所有段落，并根据实际进入网络的数据进行分片
    @staticmethod
    def batchify_test(batch, mode='test'):

        batch_size = len(batch)


        q_slide = []
        q_all = []
        c_all = []
        l_all = []
        q_idx = 0
        for i in range(batch_size):
            q_all += batch[i][0]
            c_all += batch[i][1]
            l_all += batch[i][2]
            q_slide.append((q_idx, len(q_all)))
            q_idx = len(q_all)


        max_slide = 4096

        slides = []
        bidx = eidx = 0
        while bidx < len(q_all):
            eidx = min(len(q_all), bidx + max_slide)
            slides.append((bidx, eidx))
            bidx = eidx

        slide_question = []

        slide_question_mask = []

        slide_context = []

        slide_context_mask = []

        slide_q_length = []

        for s in slides:
            slide_size = s[1] - s[0]

            question = []
            for i in range(s[0],s[1]):
                question.append(q_all[i])
            max_q_len = max([len(q) for q in question])
            q_seq = torch.LongTensor(slide_size, max_q_len).zero_()
            q_mask = torch.ByteTensor(slide_size, max_q_len).fill_(1)
            q_length = []
            for i, q in enumerate(question):
                q_seq[i, :len(q)].copy_(torch.LongTensor(q))
                q_mask[i, :len(q)].fill_(0)
                q_length.append(len(q))
            slide_question.append(q_seq)
            slide_question_mask.append(q_mask)
            slide_q_length.append(q_length)


            context = []
            for i in range(s[0], s[1]):
                context.append(c_all[i])
            max_p_len = max([len(p) for p in context])
            p_seq = torch.LongTensor(slide_size, max_p_len).zero_()
            p_mask = torch.ByteTensor(slide_size, max_p_len).fill_(1)
            for i, p in enumerate(context):
                p_seq[i, :len(p)].copy_(torch.LongTensor(p))
                p_mask[i, :len(p)].fill_(0)

            slide_context.append(p_seq)
            slide_context_mask.append(p_mask)

        return slide_question, slide_question_mask, slide_context, slide_context_mask, l_all, slide_q_length, q_slide
