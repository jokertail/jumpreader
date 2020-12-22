import json
import time
import logging
from .data import Dictionary
import torch
from tqdm import tqdm
import gc
logger = logging.getLogger(__name__)


def load_data(para_num, filename, mode):  # Load the data exists in sentence embeddings file
    examples = []
    num = 0
    num_true = 0
    num_false = 0
    with open(filename) as f:
        for line in tqdm(f, desc="load examples"):
            l = json.loads(line)
            context = []
            label = []
            if mode == "train" or mode == "dev":
                count = 0
                if len(l['context_true']) == 0 or len(l['context_false']) == 0:
                    continue
                else:
                    num += 1
                for idx_t in range(len(l['context_true'])):
                    if len(l['context_true'][idx_t]) != 0:
                        if len(l['context_true'][idx_t]['split_para']) < 50:
                            if len(l['context_true'][idx_t]['split_para']) > 40:
                                l['context_true'][idx_t]['split_para'] = l['context_true'][idx_t]['split_para'][:40]
                            context.append(l['context_true'][idx_t])
                            label.append(1)
                            count += 1
                            num_true += 1
                            if count == 3:
                                break
                        else:
                            continue

                for idx_f in range(min(len(l['context_false']), para_num - count)):
                # for idx_f in range(len(l['context_false'])):
                    if len(l['context_false'][idx_f]) != 0:
                        if len(l['context_false'][idx_f]['split_para']) < 50:
                            if len(l['context_false'][idx_f]['split_para']) > 40:
                                l['context_false'][idx_f]['split_para'] = l['context_false'][idx_f]['split_para'][:40]
                            context.append(l['context_false'][idx_f])
                            label.append(0)
                            num_false += 1
                        else:
                            continue
                    else:
                        continue

                if context == []:
                    continue
                if num == 20000:
                    break

            elif mode == "test":
                for idx_t in range(len(l['context_true'])):
                    if l['context_true'][idx_t]['length'] != 0:
                        context.append(l['context_true'][idx_t])
                        label.append(1)
                        num_true += 1

                for idx_f in range(len(l['context_false'])):
                    if l['context_false'][idx_f]['length'] != 0:
                        context.append(l['context_false'][idx_f])
                        label.append(0)
                        num_false += 1
                num+=1
            examples.append({
                'qid': l['uid'],
                'question': l['question'],
                'contexts': context,
                'labels': label,
                'answer': l['answer'],
            })
        logger.info('deal with %d paragraph_true \n %d paragraph_false \n %d questions \n total %d paragraphs' % (num_true, num_false, num,num_true+num_false))
    return examples


def load_sen_emb(filename, dict):
    '''load pretrained paragraph embedding with dataset'''
    embeddings = [0] * len(dict)
    with open(filename) as f:
        for i,line in tqdm(enumerate(f)):
            l = json.loads(line)
            for sen in dict.sen2ind.keys():
                if sen in l.keys():
                    embeddings[dict.sen2ind[sen]] =l[sen]
    # for emb in embeddings:
    #     assert len(emb) == 4096
    return embeddings


def build_sentence_dict(examples):
    sentence_dict = Dictionary()
    for ex in tqdm(examples, desc="build sentence dict"):
        sentence_dict.add(ex['question'])
        for i in range(len(ex['contexts'])):
            for sentence in ex['contexts'][i]['split_para']:
                sentence_dict.add(sentence)
    return sentence_dict


def gener_sen_emb(sentence_dict, model_embed):
    '''generate sentence embeddings using pre-trained model '''
    GLOVE_PATH = '/home/wxy/xsy/Glove/glove.840B.300d.txt'
    model_embed.set_glove_path(GLOVE_PATH)
    model_embed.build_vocab(sentence_dict)
    model_embed.to('cuda')
    print("2")
    with torch.no_grad():
        embeddings = torch.Tensor(model_embed.encode(
            sentence_dict, bsize=512, tokenize=False, verbose=True)).detach()  # (sentence_num,4096)

    return embeddings


def output_emb(emb, out_file, l, sentence_dict, off_set):
    '''write compressed sentence embeddings to file'''
    emb = emb.cpu().numpy()
    sentence_emb = open(out_file, 'a+')
    tmp_emb = {}
    for i in range(l, l + len(emb)):
        tmp_emb[sentence_dict.ind2sen[i + off_set]] = emb[i - l].tolist()
    sentence_emb.write(json.dumps(tmp_emb) + '\n')


def output_emb_batch(output_file, embeddings, sentence_dict):
    '''compress and write sentence embeddings to file in batches'''

    skip = 10240
    length = len(embeddings)
    b_size = length // skip
    off_set = list(sentence_dict.ind2sen.keys())[0]
    for l in range(0,length,skip):
        emb = embeddings[l:min(l + skip, length), :]
        output_emb(emb, output_file, l, sentence_dict, off_set)


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """Computes elapsed time."""

    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total
