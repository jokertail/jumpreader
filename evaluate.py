import torch
import os
import time
import json
import argparse
import logging

logger = logging.getLogger()

PREDICT_DIR = 'result/long'

def add_predict_args(parser):
    parser.add_argument('--predict-file', type=str, default='jump-N2-K10_checkpoint-190000jump-N2-K10.preds',
                    help='Prediction files')
    parser.add_argument('--predict-dir',type=str, default=PREDICT_DIR,
                    help='Directory for the prediction file')
    parser.add_argument('--top-k', type=int, default=5,
                    help='Number of the selected paragraphs')
    # parser.add_argument('--log-dir', type=str, default='test.txt')


def set_defaults(args):
    args.predict_file = os.path.join(args.predict_dir, args.predict_file)
    if not os.path.isfile(args.predict_file):
        raise IOError('No such file: %s' % args.predict_file)
    args.log_file = args.predict_file + '.log'

def load_prediction(args):
    with open(args.predict_file) as f:
        dataset = []
        for line in f:
            dataset.append(json.loads(line))
    return dataset


def calculate_recall(args,dataset):
    for line in dataset:
        line['rank'].sort(key=lambda x: x['score'], reverse=True)
        # line['rank'].sort(key=lambda x: (-x[1]))
    n_q = 0
    hit = 0
    for line in dataset:
        if len(line['rank'])>=args.top_k:
            n_q += 1
            for i in range(args.top_k):
                if line['rank'][i]['type']==1:
                    hit += 1
                    break
    recall = hit/n_q
    return recall

def multiple_scores(args,dataset):
    for line in dataset:
        line['rank'].sort(key=lambda x: x['score'], reverse=True)
        # line['rank'].sort(key=lambda x: (-x[1]))
    score_list = []
    n_q = 0
    hit = 0
    for line in dataset:
        score = []
        if len(line['rank'])>=args.top_k:
            n_q += 1
            for i in range(args.top_k):
                score.append(line['rank'][i]['score'])
        if score[0] == score[1]:
           score_list.append(score)
    # print(score_list)
    return score_list



def main(args):
    logger.info('-' * 100)
    logger.info('Load predict files')
    dataset=load_prediction(args)
    logger.info('-' * 100)
    logger.info('Calculate recall of %s' %args.predict_file)
    recall=calculate_recall(args, dataset)
    if args.top_k > 1:
        multiple_scores(args,dataset)
    logger.info('Recall of top %d = %.5f' % (args.top_k,recall))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_predict_args(parser)
    args = parser.parse_args()
    set_defaults(args)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if args.log_file:
        logfile = logging.FileHandler(args.log_file)
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    main(args)