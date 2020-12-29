import argparse
import torch
import numpy as np
import random
import sys
import logging
import os
import json
from torch import optim
import subprocess
from ranker import utils, data
from ranker.RankWordJumper import RankWordJumper
from ranker.RWJDataSet import RWJDataSet
from ranker.data import Dictionary, RankerDataset
from ranker.model import Ranker, Jumper
from ranker.WordJumper import WordJumper
from ranker.utils import gener_sen_emb, output_emb_batch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
import heapq
from collections import OrderedDict
from torch.optim.lr_scheduler import LambdaLR

from ranker.utils4WordJump import load_data, load_glove
from run import set_seed

logger = logging.getLogger()
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def add_train_args(parser):
    """Adds commandline arguments pertaining to training a model. These
    are different from the arguments dictating the model architecture.
    """
    # Runtime environment
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--no-cuda', action="store_true",
                         help='Train on CPU, even if GPUs are available.')
    runtime.add_argument('--gpu', type=int, default=-1,
                         help='Run on a specific GPU')
    runtime.add_argument('--data-workers', type=int, default=12,
                         help='Number of subprocesses for data loading')

    runtime.add_argument('--random-seed', type=int, default=1013,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    runtime.add_argument('--num-epochs', type=int, default=50,
                         help='Train data iterations')
    runtime.add_argument('--batch-size', type=int, default=64,
                         help='Batch size for training')
    runtime.add_argument('--test-batch-size', type=int, default=32,
                         help='Batch size during validation/testing')
    runtime.add_argument("--do-train", action="store_true", help="Whether to run training.")
    runtime.add_argument("--do-test", action="store_true", help="Whether to run eval on the test set.")
    runtime.add_argument("--generate-emb", action="store_true", help="Whether to generate sentence embeddings")
    runtime.add_argument("--generate-mode", type=str, default='train')

    runtime.add_argument('--model-type', type=str, default='jump',
                         choices=('base', 'jump'))
    runtime.add_argument('--para-num', type=int, default=6,
                         help='number of paragraphs for one question')
    runtime.add_argument(
        "--overwrite-cache", action="store_true", help="Overwrite the content of the output directory"
    )
    runtime.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    runtime.add_argument("--init-step", type=int, default=0, help="the start step")
    parser.add_argument("--warmup-steps", default=2000, type=int, help="Linear warmup over warmup_steps.")

    # Files
    files = parser.add_argument_group('Filesystem')
    files.add_argument('--output-dir', type=str, default='output/long',
                       help='Directory for saved ranker/checkpoints/logs')
    files.add_argument('--model-name', type=str, default='test',
                       help='Unique model identifier (.mdl, .txt, .checkpoint)')
    files.add_argument('--data-dir', type=str, default='data/quasar/long',
                       help='Directory of training/validation data')
    files.add_argument('--train-file', type=str, default='demo',
                       help='Preprocessed train file')
    files.add_argument('--dev-file', type=str, default='demo',
                       help='Preprocessed dev file')
    files.add_argument('--test-file', type=str, default='demo',
                       help='Preprocessed dev file')
    files.add_argument('--embed-dir', type=str, default='/home/wxy/xsy/Glove',
                       help='Directory of pre-trained embedding files')
    files.add_argument('--embedding-file', type=str,
                       default='glove.840B.300d.txt',
                       help='Space-separated pretrained embeddings file')
    files.add_argument("--ckpt-file", type=str, default="test_checkpoint-4",
                       help="the model checkpoint file.", )
    files.add_argument("--result_path", type=str, default="result/long",
                       help="The output directory where the model checkpoints and predictions will be written.", )
    # Saving + loading
    save_load = parser.add_argument_group('Saving/Loading')
    save_load.add_argument("--load-ckpt", action='store_true',
                           help="Whether to load ckpt", )
    save_load.add_argument('--pretrained', type=str, default='',
                           help='Path to a pretrained model to warm-start with')

    # General
    general = parser.add_argument_group('General')
    general.add_argument('--logging-steps', type=int, default=50,
                         help='Log state after every <display_iter> epochs')

    # Model architecture
    config = parser.add_argument_group('Model Config')
    config.add_argument('--embedding-dim', type=int, default=300,
                        help='Embedding size if embedding_file is not given')
    config.add_argument('--hidden-size', type=int, default=4096,
                        help='Hidden size of RNN units')
    config.add_argument('--num-layers', type=int, default=2,
                        help='Number of encoding layers for document')
    config.add_argument('--question-layers', type=int, default=2,
                        help='Number of encoding layers for question')
    config.add_argument('--rnn-type', type=str, default='lstm',
                        help='RNN type: LSTM, GRU, or RNN')
    config.add_argument('--dropout-emb', type=float, default=0.1,
                        help='Dropout rate for word embeddings')
    config.add_argument('--dropout-rnn', type=float, default=0.1,
                        help='Dropout rate for RNN states')
    config.add_argument('--dropout-rnn-output', action="store_true",
                        help='Whether to dropout the RNN output')
    config.add_argument('--optimizer', type=str, default='adamax',
                        help='Optimizer: sgd or adamax')
    config.add_argument('--learning-rate', type=float, default=0.01,
                        help='Learning rate for SGD only')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    config.add_argument('--grad-clipping', type=float, default=10,
                        help='Gradient clipping')
    config.add_argument('--weight-decay', type=float, default=0.001,
                        help='Weight decay factor')
    config.add_argument('--momentum', type=float, default=0.5,
                        help='Momentum factor')
    config.add_argument('--fix-embeddings', action="store_true",
                        help='Keep word embeddings fixed (use pretrained)')
    config.add_argument('--tune-partial', type=int, default=0,
                        help='Backprop through only the top N question words')
    config.add_argument('--max-len', type=int, default=15,
                        help='The max span allowed during decoding')
    config.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    config.add_argument("--R", default=2, type=int, help="read R sentences before jumping")
    config.add_argument("--N", default=10, type=int, help="max jump times ")
    config.add_argument("--K", default=10, type=int, help="max skip number for each jump")

    config.add_argument("--sample-num", default=6, type=int, help="sampling number of paragraphs to every quesiton")
    config.add_argument("--glove-hidden-size", default=300, type=int, help="Glove word embedding dimension")


def set_defaults(args):
    """Make sure the commandline arguments are initialized properly."""
    # Check critical files exist
    args.train_path = os.path.join(args.data_dir, args.train_file + '.txt')
    if not os.path.isfile(args.train_path):
        raise IOError('No such file: %s' % args.train_path)
    args.dev_path = os.path.join(args.data_dir, args.dev_file + '.txt')
    if not os.path.isfile(args.dev_path):
        raise IOError('No such file: %s' % args.dev_path)
    args.test_path = os.path.join(args.data_dir, args.test_file + '.txt')
    if not os.path.isfile(args.test_path):
        raise IOError('No such file: %s' % args.test_path)
    # if args.embedding_file:
    #     args.embedding_file = os.path.join(args.embed_dir, args.embedding_file)
    #     if not os.path.isfile(args.embedding_file):
    #         raise IOError('No such file: %s' % args.embedding_file)
    subprocess.call(['mkdir', '-p', args.output_dir])

    # Set log + model file names
    args.log_file = os.path.join(args.output_dir, args.model_name + '.txt')
    args.model_file = os.path.join(args.output_dir, args.model_name + '.mdl')
    args.ckpt_path = os.path.join(args.output_dir, args.ckpt_file)
    return args


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def train(args, data_loader, model, global_stats, scheduler):
    loss = 0
    for step, batch in tqdm(enumerate(data_loader), desc="training"):
        model.train()
        torch.cuda.empty_cache()
        inputs = {
            "q_seq": batch[0],
            "q_mask": batch[1],
            "p_seq": batch[2],
            "p_mask": batch[3],
            "labels": batch[4].to(args.device),
            "q_length": batch[5],
            "group_size": batch[6]
        }
        loss_tem = model(**inputs)
        loss_tem = loss_tem.sum()
        loss += loss_tem
        # if args.gradient_accumulation_steps > 1:
        #     loss = loss / args.gradient_accumulation_steps
        # torch.cuda.empty_cache()
        global_stats['total_loss'] += loss_tem.item()
        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            loss.backward()
            args.optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            loss = 0
            global_stats['global_step'] += 1
            global_stats['tr_loss'] = global_stats['total_loss'] / (global_stats['global_step'] - args.init_step)
            if args.logging_steps > 0 and global_stats['global_step'] % args.logging_steps == 0:
                logger.info('train: Epoch = %d | iter = %d/%d | ' %
                            (global_stats['epoch'], global_stats['global_step'], len(batch)) +
                            'loss = %.4f | elapsed time = %.2f (min) | ' %
                            (global_stats['tr_loss'], global_stats['timer'].time() / 60) +
                            'learning_rate: %.6f' % (scheduler.get_lr()[0])
                            )


def evaluate(data_loader, model, global_stats, mode):
    eval_time = utils.Timer()
    exact_match = utils.AverageMeter()
    exact_match_top3 = utils.AverageMeter()
    # Make predictions
    examples = 0
    model.eval()
    for batch in tqdm(data_loader, desc="evaluating"):
        with torch.no_grad():
            inputs = {
                "q_seq": batch[0],
                "q_mask": batch[1],
                "p_seq": batch[2],
                "p_mask": batch[3],
                "q_length": batch[5]
            }
            labels = batch[4]
            scores = model.module.inference(**inputs).squeeze().cpu().detach().numpy().tolist()
            pred_label = []
            target_label = []
            count = 0
            for i in range((int)(len(labels)/args.sample_num)):
                pred_label.append(scores[count:count+args.sample_num])
                target_label.append(labels[count:count+args.sample_num])
                count += args.sample_num

            batch_size = len(pred_label)
            for i in range(batch_size):
                if pred_label[i].index(max(pred_label[i])) == 0:
                    exact_match.update(1)
                else:
                    exact_match.update(0)

    logger.info('%s valid unofficial: Epoch = %d | ' %
                (mode, global_stats['epoch']) +
                'recall_1 = %.4f | examples = %d | ' %
                (exact_match.avg, examples) +
                'valid time = %.2f (s)' % eval_time.time())

    return {'exact_match': exact_match.avg}



def main(args):
    if args.model_type == "jump":
        model = RankWordJumper(args)

        # multi-gpu training (should be after apex fp16 initialization)
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.to(args.device)

    glove_file_path = os.path.join(args.embed_dir, args.embedding_file)
    word_dict, word_emb = load_glove(glove_file_path)
    # why module.set_emb() ???
    model.module.set_emb(word_emb)


    if args.do_train:
        logger.info('-' * 100)
        logger.info('Load data files')
        train_exs = load_data(args.train_path)
        train_dataset = RWJDataSet(train_exs, word_dict)
        train_sampler = SequentialSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.data_workers,
            collate_fn=train_dataset.batchify,
            pin_memory=True,
        )

        dev_exs = load_data(args.dev_path)
        dev_dataset = RWJDataSet(dev_exs, word_dict)
        dev_sampler = SequentialSampler(dev_dataset)
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=args.test_batch_size,
            sampler=dev_sampler,
            num_workers=args.data_workers,
            collate_fn=dev_dataset.batchify,
            pin_memory=True,
        )

        # False default
        if args.load_ckpt:
            pass

        # -------------------------------------------------------------------------
        # PRINT CONFIG
        logger.info('-' * 100)
        logger.info("Training/evaluation parameters %s", args)
        # --------------------------------------------------------------------------
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]

        args.optimizer = optim.Adamax(optimizer_grouped_parameters, lr=args.learning_rate)
        t_total = len(train_loader) * args.num_epochs
        scheduler = get_linear_schedule_with_warmup(args.optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=t_total)
        # Train!
        logger.info("*************** Running training ***************")
        logger.info("  train examples = %d", len(train_dataset))
        logger.info("  eval examples = %d", len(dev_dataset))
        logger.info("  Num Epochs = %d", args.num_epochs)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        model.zero_grad()
        stats = {'timer': utils.Timer(), 'epoch': 0, 'global_step': args.init_step, 'best_valid': 0, 'tr_loss': 0,
                 'total_loss': 0}
        set_seed(args)  # Added here for reproductibility
        start_epoch = 0
        train_time = utils.Timer()
        for epoch in range(start_epoch, args.num_epochs):
            stats['epoch'] = epoch

            train(args, train_loader, model, stats, scheduler)
            evaluate(train_loader, model, stats, mode="train")

            result = evaluate(dev_loader, model, stats, mode="dev")

            # Save best valid
            if result['exact_match'] > stats['best_valid']:
                logger.info('Best valid: exact_match = %.2f (epoch %d, %d updates)' %
                            (result['exact_match'],
                             stats['epoch'], stats['global_step']))
                output_path = os.path.join(args.output_dir,
                                           (args.model_name + "_checkpoint-{}".format(stats['global_step'])))
                torch.save(model.module.state_dict(), output_path)
                stats['best_valid'] = result['exact_match']

        logger.info('-' * 100)
        logger.info('Time for training = %.2f (s)' % train_time.time())
        logger.info('-' * 100)


if __name__ == '__main__':
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        'LSTM-jump-ranker',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_train_args(parser)
    args = parser.parse_args()
    set_defaults(args)

    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if args.log_file:
        logfile = logging.FileHandler(args.log_file, 'a')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))

    # Run!
    main(args)
