from transformer import Chatter, Criterion, NoamOpt
import random
import torch
import numpy as np
from copy import deepcopy
import pickle
import matplotlib.pyplot as plt

batch_n = 1000
batch_size = 20
max_context_size = 10
max_sentence_size = 10
vocab_size = 20
word_embed_size = 16
sentence_embed_size = 32

padding_char = 0
start_char = 1
end_char = 2


def next_char(cur_char, step):
    assert cur_char > 2 and cur_char < vocab_size
    return (cur_char + step - 3) % (vocab_size - 3) + 3


class Sample():
    def __init__(self, src, tgt, sol, ctx_size):
        self.src = src
        self.tgt = tgt
        self.sol = sol
        self.wrd_mask = (src.view(src.size(0), -1) == padding_char).unsqueeze(-2)
        ttt = (tgt == padding_char)
        self.tgt_mask = ttt.unsqueeze(-2) | ttt.unsqueeze(-1) | torch.from_numpy(
            np.triu(np.ones(shape=(max_sentence_size, max_sentence_size), dtype=bool), k=1))
        tmp = np.full((src.size(0), 1, max_context_size), True)
        tmp[:, :, :ctx_size] = False
        self.sts_mask = torch.from_numpy(tmp)


def data_generator(batch_n, batch_size, max_ctx_size, max_sts_size, vocab_size):
    for i in range(batch_n):
        ctx_size = random.randint(3, max_ctx_size)
        sts_size = random.randint(3, max_sts_size)
        src = torch.zeros((batch_size, max_ctx_size, max_sts_size), dtype=int)
        tgt = torch.zeros((batch_size, max_sts_size), dtype=int)
        for i in range(batch_size):
            cur = random.randint(3, vocab_size - 1)
            for j in range(ctx_size):
                src[i, j, :sts_size] = cur
                cur = next_char(cur, 1)
            tgt[i, :sts_size] = cur
        src[:, :, sts_size - 1] = end_char
        sol = deepcopy(tgt)
        sol[:, sts_size - 1] = end_char
        tgt[:, 0] = start_char
        yield Sample(src, tgt, sol, ctx_size)


#####################################################################
# loss_record = np.zeros(batch_n)
# chatter = Chatter(vocab_size, word_embed_size, sentence_embed_size, max_sentence_size, max_context_size, layer_n=2)
# chatter.train()
# gene = data_generator(batch_n, batch_size, max_context_size, max_sentence_size, vocab_size)
# chatter_opt = NoamOpt(512, 1, 100, torch.optim.Adam(chatter.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
# criterion = Criterion(vocab_size, chatter_opt)
# for i, sample in enumerate(gene):
#     out = chatter(sample.src, sample.tgt, sample.wrd_mask, sample.sts_mask, sample.tgt_mask)
#     loss_record[i] = criterion(out, sample.sol, i)

# f = open('checkpoint_chat.pkl', 'wb')
# pickle.dump(chatter, f)

# idx = np.arange(batch_n)
# plt.plot(idx, loss_record)
# plt.show()
#####################################################################
f1 = open('checkpoint_chat.pkl', 'rb')
chatter = pickle.load(f1)
chatter.eval()


def simple_decode(out):
    max_idx = torch.max(out, 1)
    return max_idx.indices


gene = data_generator(1, batch_size, max_context_size, max_sentence_size, vocab_size)
print("####################")
test_sample = next(gene)
rand_idx = random.randint(0, batch_size - 1)
test_src = test_sample.src[rand_idx]
print(test_src)
wrd_mask = test_sample.wrd_mask[rand_idx]
sts_mask = test_sample.sts_mask[rand_idx]
test_tgt = torch.from_numpy(np.full(max_sentence_size, 0))
test_tgt[0] = start_char
tgt_mask = torch.from_numpy(np.full(max_sentence_size, True))
tgt_mask[0] = False
i = 1
while i < max_sentence_size:
    print(test_tgt[:i])
    out = chatter(test_src.unsqueeze(0), test_tgt.unsqueeze(0), wrd_mask.unsqueeze(0), sts_mask.unsqueeze(0),
                  tgt_mask.unsqueeze(0))
    predict = simple_decode(out.squeeze(0))
    if predict[i - 1] == end_char:
        break
    test_tgt[i] = predict[i - 1]
    tgt_mask[i] = False
    i += 1
print(test_tgt)
