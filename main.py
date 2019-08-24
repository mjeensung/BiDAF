from model.model import BIDAF_Model, EMA
from model.data import SQuAD
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
from util import *
import os

def train(model, args, data, optimizer, criterion, ema):
    model.train()
    loss, total = 0, 0

    iterator = data.train_iter
    for i, batch in tqdm(enumerate(iterator), total=len(iterator)):
        p1, p2 = model(context_words=batch.c_word[0],
                    context_chars=batch.c_char,
                    query_words=batch.q_word[0],
                    query_chars=batch.q_char)
        optimizer.zero_grad()
        batch_loss = criterion(p1,batch.start_idx) + criterion(p2,batch.end_idx)
        if i%args.print_freq == 0 :
            print("step={}/{}, batch_loss={}".format(i+1, len(iterator), batch_loss))
        loss += batch_loss.item()
        total += 1
        batch_loss.backward()
        optimizer.step()

        for name, param in model.named_parameters():
            if param.requires_grad:
                ema.update(name, param.data)
    loss /= total
    return loss, model

def test(model, args, data, criterion, ema):
    loss, total = 0,0
    answers = dict()
    model.eval()

    temp_ema = EMA(0)
    for name, param in model.named_parameters():
        if param.requires_grad:
            temp_ema.register(name, param.data)
            param.data.copy_(ema.get(name))
    iterator = data.dev_iter
    with torch.no_grad():
        for i, batch in tqdm(enumerate(iterator), total=len(iterator)):
            p1, p2 = model(context_words=batch.c_word[0],
                    context_chars=batch.c_char,
                    query_words=batch.q_word[0],
                    query_chars=batch.q_char)
            batch_loss = criterion(p1,batch.start_idx) + criterion(p2,batch.end_idx)
            loss += batch_loss.item()
            total += 1

            # (batch, c_len, c_len)
            batch_size, c_len = p1.size()
            ls = nn.LogSoftmax(dim=1)
            mask = (torch.ones(c_len, c_len) * float('-inf')).cuda().tril(-1).unsqueeze(0).expand(batch_size, -1, -1)
            score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
            score, s_idx = score.max(dim=1)
            score, e_idx = score.max(dim=1)
            s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()

            for i in range(batch_size):
                id = batch.qid[i]
                answer = batch.c_word[0][i][s_idx[i]:e_idx[i]+1]
                answer = ' '.join([data.WORD.vocab.itos[idx] for idx in answer])
                answers[id] = answer
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data.copy_(temp_ema.get(name))

    loss /= total
    results = evaluate(args, answers)
    return loss, results['exact_match'], results['f1']

def init_seed(seed=None):
    if seed is None:
        seed = int(round(time.time() * 1000)) % 10000

    import os
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True

def main():
    parser = argparse.ArgumentParser()

    # run
    parser.add_argument('--seed',  type=int, default=0)

    # model
    parser.add_argument('--char_dim', default=100, type=int)
    parser.add_argument('--char_channel_width', default=5, type=int)
    parser.add_argument('--char_channel_size', default=100, type=int)
    parser.add_argument('--word_dim', default=100, type=int)
    parser.add_argument('--hidden_size', default=100, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--print_freq', default=250, type=int)
    
    # data
    parser.add_argument('--max_token_len', default=400, type=int)

    # train
    parser.add_argument('--epoch', default=12, type=int)
    parser.add_argument('--exp_decay_rate', default=0.999, type=float)
    parser.add_argument('--learning_rate', default=0.5, type=float)
    parser.add_argument('--train_batch_size', default=60, type=int)
    parser.add_argument('--train_file', default='train-v1.1.json')
    parser.add_argument('--GPU', default=0, type=int)

    # dev
    parser.add_argument('--dev_batch_size', default=100, type=int)
    parser.add_argument('--dev_file', default='dev-v1.1.json')

    args = parser.parse_args()

    init_seed(args.seed)
    
    print('loading SQuAD data...')
    data = SQuAD(args)
    setattr(args, 'char_vocab_size', len(data.CHAR.vocab))
    setattr(args, 'word_vocab_size', len(data.WORD.vocab))
    print('data loading complete!')

    print('loading model')
    model = BIDAF_Model(
                args,
                pretrained = data.WORD.vocab.vectors)
    print('loading model complete!')

    print('training start!')
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, lr=args.learning_rate)
    ema = EMA(args.exp_decay_rate)
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)
    criterion = nn.NLLLoss()
    
    best_f1 = 0
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    write_tsv("result",[['train_loss','val_loss', 'exact_match', 'f1']],append=False)
    for epoch in range(args.epoch):
        train_loss, model = train(model, args, data, optimizer, criterion, ema)
        val_loss, exact_match, f1 = test(model, args, data, criterion, ema)
        
        print("epoch={}/{} train_loss={}, val_loss={}, exact_match={}, f1={}".format(epoch+1,args.epoch,train_loss, val_loss, exact_match, f1))
        write_tsv("result",[[train_loss,val_loss, exact_match, f1]],append=True)
        if best_f1 < f1:
            best_f1 = f1
            model.save_checkpoint({'state_dict':model.state_dict()},"./","model.ckpt")
    print('training finished!')


if __name__ == '__main__':
    main()