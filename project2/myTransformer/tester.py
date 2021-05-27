import torch
import numpy as np
from transformer import Translater, Criterion, NoamOpt
import pickle
import matplotlib.pyplot as plt

batch_n = 1000
batch_size = 100
max_seq_len = 15
vocab_size = 100
src_embed_size = 26
tgt_embed_size = 28
padding_char = 0
start_char = 1
end_char = 2


class Sample():
    def __init__(self, src, tgt):
        super(Sample, self).__init__()
        self.src = src
        self.tgt = tgt
        self.sol = torch.cat((tgt[:, 1:], torch.from_numpy(np.full((tgt.size(0), 1), end_char))), 1)
        self.src_mask = (src == padding_char).unsqueeze(-2)
        self.tgt_mask = (tgt == padding_char).unsqueeze(-2) | torch.from_numpy(
            np.triu(np.ones(shape=(max_seq_len, max_seq_len), dtype=bool), k=1))


def data_generator(batch_n, batch_size, max_seq_len, vocab_size):
    for i in range(batch_n):
        tmp = torch.from_numpy(np.random.randint(3, vocab_size, (batch_size, max_seq_len + 1)))
        src = tmp[:, 1:]
        src[:, -1] = end_char
        tgt = tmp[:, :-1]
        tgt[:, 0] = start_char
        yield Sample(src, tgt)


########################################################################
# translater = Translater(vocab_size, src_embed_size, tgt_embed_size, max_seq_size=max_seq_len, layer_n=2)
# loss_record = np.zeros(batch_n)
# translater_opt = NoamOpt(512, 1, 100, torch.optim.Adam(translater.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
# criterion = Criterion(vocab_size, translater_opt)
# sample_gene = data_generator(batch_n, batch_size, max_seq_len, vocab_size)
# translater.train()
# for i, sample in enumerate(sample_gene):
#     out = translater(sample.src, sample.tgt, sample.src_mask, sample.tgt_mask)
#     loss_record[i] = criterion(out, sample.sol, i).item()

# f = open('checkpoint_test.pkl', 'wb')
# pickle.dump(translater, f)

# idx = np.arange(batch_n)
# plt.plot(idx, loss_record)
# plt.show()
#######################################################################
f1 = open('checkpoint_test.pkl', 'rb')
translater = pickle.load(f1)


def simple_decode(out):
    max_idx = torch.max(out, 1)
    return max_idx.indices


print("####################")
test_src = torch.from_numpy(np.random.randint(3, vocab_size, (max_seq_len)))
test_src[-1] = end_char
print(test_src)
src_mask = (test_src == padding_char)
test_tgt = torch.from_numpy(np.full(max_seq_len, 0))
test_tgt[0] = start_char
tgt_mask = torch.from_numpy(np.full(src_mask.size(), True))
print(tgt_mask.shape)
tgt_mask[0] = False
i = 1
while i < max_seq_len:
    print(test_tgt[:i])
    out = translater(test_src.unsqueeze(0), test_tgt.unsqueeze(0), src_mask.unsqueeze(0), tgt_mask.unsqueeze(0))
    predict = simple_decode(out.squeeze(0))
    test_tgt[i] = predict[i - 1]
    if predict[i] == end_char:
        break
    tgt_mask[i] = False
    i += 1
print(test_tgt)