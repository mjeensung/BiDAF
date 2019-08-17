from model import BIDAF_Model
from prepro import READ

# load data
args = {"Train_File":"train-v1.1.json",
        "Dev_File":"dev-v1.1.json",
        "Max_Token_Length":200,
        "Word_Dim":100,
        "GPU":0,
        "Batch_Size":1
    }
reader = READ(args)

train_iter = reader.train_iter
dev_iter = reader.dev_iter

for i, data in enumerate(train_iter):
    print(i)
    print(data.qid)
    print(data.start_idx)
    print(data.end_idx)
    print(data.c_word)
    print(data.c_char)
    print(data.q_word)
    print(data.q_char)
    break