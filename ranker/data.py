import numpy as np
import torch
import logging
import unicodedata

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class Dictionary(object):
    NULL = '<NULL>'
    UNK = '<UNK>'
    START = 2

    @staticmethod
    def normalize(sen):
        return unicodedata.normalize('NFD', sen)

    def __init__(self):
        self.sen2ind = {self.NULL: 0, self.UNK: 1}
        self.ind2sen = {0: self.NULL, 1: self.UNK}

    def __len__(self):
        return len(self.sen2ind)

    def __iter__(self):
        return iter(self.sen2ind)

    def __contains__(self, key):
        if type(key) == int:
            return key in self.ind2sen
        elif type(key) == str:
            return self.normalize(key) in self.sen2ind

    def __getitem__(self, key):
        if type(key) == int:
            return self.ind2sen.get(key, self.UNK)
        if type(key) == str:
            return self.sen2ind.get(self.normalize(key),
                                    self.sen2ind.get(self.UNK))

    def __setitem__(self, key, item):
        if type(key) == int and type(item) == str:
            self.ind2sen[key] = item
        elif type(key) == str and type(item) == int:
            self.sen2ind[key] = item
        else:
            raise RuntimeError('Invalid (key, item) types.')

    def add(self, sentence):
        sentence = self.normalize(sentence)
        if sentence not in self.sen2ind:
            index = len(self.sen2ind)
            self.sen2ind[sentence] = index
            self.ind2sen[index] = sentence

    def add_index(self, sentence,idx):
        sentence = self.normalize(sentence)
        self.sen2ind[sentence] = idx
        self.ind2sen[idx] = sentence

    def delete(self,index):
        self.sen2ind.pop(self.ind2sen[index])
        self.ind2sen.pop(index)


    def sentences(self):
        """Get dictionary tokens.

        Return all the words indexed by this dictionary, except for special
        tokens.
        """
        senences = [k for k in self.sen2ind.keys()
                    if k not in {'<NULL>', '<UNK>'}]
        return senences


class RankerDataset(Dataset):

    def __init__(self, examples, sentence_dict, mode='train'):
        self.sentence_dict = sentence_dict
        self.examples = examples
        self.mode = mode

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        try:
            outputs= vectorize(self.examples[index],self.sentence_dict)
            # print(len(self.examples),index)
            return outputs
        except IndexError:
            print(len(self.examples),index)


        # if self.mode == 'train':
        #     return vectorize(self.examples[index], self.sentence_dict)
        # elif self.mode == 'dev':
        #     return vectorize_dev(self.examples[index], self.sentence_dict)
        # elif self.mode == 'test':
        #     return vectorize_test(self.examples[index], self.sentence_dict)

    def lengths(self):
        return [(len(ex['question']['str']), len(ex['ct']['str']), len(ex['cf'][:]['str']))
                for ex in self.examples]


def vectorize(ex, sentence_dict):
    question = torch.LongTensor([sentence_dict[ex['question']]])
    contexts = []
    para_num = len(ex['contexts'])
    for i in range(para_num):
        ct_ex = torch.LongTensor([sentence_dict[s] for s in ex['contexts'][i]['split_para']])
        contexts.append(ct_ex)
    labels = ex['labels']

    return question, contexts, labels,para_num


def batchify(batch, mode='train'):
    questions = []
    contexts = []  # paragraphs
    labels = []
    para_num = []
    for ex in batch:
        questions += [ex[0]] * (len(ex[1]))
        contexts += ex[1]
        labels += ex[2]
        para_num += [ex[3]]
    para_num = torch.ByteTensor(para_num)
    assert len(questions) == len(contexts) == len(labels)
    lengths = torch.ByteTensor([len(ct) for ct in contexts])
    questions = torch.LongTensor(questions)
    labels = torch.ByteTensor(labels)
    max_sen = max(lengths)  # max sentence number of paragraph
    mask = torch.ByteTensor(len(contexts), max_sen).fill_(1)  # [batch_size,max_sen]
    xcts = torch.LongTensor(len(contexts), max_sen).zero_()
    for i, ct in enumerate(contexts):
        xcts[i, :len(ct)].copy_(ct)
        mask[i, :len(ct)].fill_(0)
    return questions, xcts, mask, labels,lengths,para_num
