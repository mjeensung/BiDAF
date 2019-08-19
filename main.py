from model import BIDAF_Model
from prepro import READ
import torch
import torch.optim as optim
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()

# Run settings
parser.add_argument('--max_token_len', 
                    default=200, type=int)
parser.add_argument('--word_dim', 
                    default=100, type=int)
parser.add_argument('--char_dim', 
                    default=100, type=int)
parser.add_argument('--learning_rate', 
                    default=0.5, type=float)
parser.add_argument('--epoch', 
                    default=10, type=int)
parser.add_argument('--batch', 
                    default=10, type=int)
parser.add_argument('--dropout', 
                    default=0.2, type=float)

args = parser.parse_args()

# load data
reader = READ({"Train_File":"train-v1.1.json",
        "Dev_File":"dev-v1.1.json",
        "Max_Token_Length":args.max_token_len,
        "Word_Dim":args.word_dim,
        "GPU":0,
        "Batch_Size":args.batch
})

train_iter = reader.train_iter
val_iter = reader.dev_iter
# load model
print("len(reader.WORD.vocab)=",len(reader.WORD.vocab))
print("len(reader.CHAR.vocab)=",len(reader.CHAR.vocab))
weight_matrix = reader.WORD.vocab.vectors
model = BIDAF_Model(char_size=len(reader.CHAR.vocab),
                    vocab_size=len(reader.WORD.vocab),
                    char_dim = args.char_dim,
                    word_dim = args.word_dim, 
                    dropout=args.dropout)
model.word_embedding.weight.data.copy_(weight_matrix)
optimizer = torch.optim.Adadelta(model.parameters(), lr=args.learning_rate)

# train
for epoch in range(args.epoch):
    train_loss = 0
    val_loss = 0
    train_total = 0
    val_total = 0
    model.train()
    for i, data in tqdm(enumerate(train_iter),total=len(train_iter)):
        pred_start, pred_end = model(context_words=data.c_word[0],
                                     context_chars=data.c_char,
                                     query_words=data.q_word[0],
                                     query_chars=data.q_char)
        loss = model.get_loss(start_idx=data.start_idx,
                              end_idx=data.end_idx,
                              p1=pred_start,
                              p2=pred_end)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_total += 1
        # print(loss)
        # print(i)
        # print(data.qid)
        # print(data.start_idx)
        # print(data.end_idx)
        # print(data.c_word)
        # print(data.c_char)
        # print(data.q_word)
        # print(data.q_char)
    #     break
    # break
    train_loss /= train_total
    print("train_loss=",train_loss)

    model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(val_iter),total=len(val_iter)):
            pred_start, pred_end = model(context_words=data.c_word[0],
                                     context_chars=data.c_char,
                                     query_words=data.q_word[0],
                                     query_chars=data.q_char)
            loss = model.get_loss(start_idx=data.start_idx,
                                end_idx=data.end_idx,
                                p1=pred_start,
                                p2=pred_end)
            val_loss += loss.item()
            val_total += 1
            # print(loss)
            # print(i)
            # print(data.qid)
            # print(data.start_idx)
            # print(data.end_idx)
            # print(data.c_word)
            # print(data.c_char)
            # print(data.q_word)
            # print(data.q_char)
            # break
        val_loss /= val_total
        print("val_loss=",val_loss)