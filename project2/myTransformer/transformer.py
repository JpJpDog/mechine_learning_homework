import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from copy import deepcopy
import numpy as np


def clones(module, n):
    return nn.ModuleList([deepcopy(module) for _ in range(n)])


class Chatter(nn.Module):
    def __init__(self, vocab_size, wrd_embed_size, sts_embed_size, max_sts_size, max_ctx_size, layer_n=6):
        super(Chatter, self).__init__()
        self.embeddings = Embeddings(vocab_size, wrd_embed_size)
        self.wrd_position = PositionEncoder(wrd_embed_size, max_sts_size)
        self.wrd_encoder = Encoder(wrd_embed_size, layer_n)
        self.wrd2sts = nn.Linear(max_sts_size * wrd_embed_size, sts_embed_size)
        self.sts_position = PositionEncoder(sts_embed_size, max_ctx_size)
        self.sts_encoder = Encoder(sts_embed_size, layer_n)
        self.decoder = Decoder(wrd_embed_size, sts_embed_size, layer_n)
        self.debeddings = Debeddings(wrd_embed_size, vocab_size)
        for para in self.parameters():
            if para.dim() > 1:
                nn.init.xavier_uniform_(para)

    def wrd_encode(self, x, wrd_mask):
        x = self.wrd_position(x)
        origin_shape = x.size()
        x = x.view(origin_shape[0], -1, origin_shape[-1])
        x = self.wrd_encoder(x, wrd_mask)
        return x.view(origin_shape)

    def sts_encode(self, x, sts_mask):
        origin_shape = x.size()
        x = x.view(origin_shape[0], origin_shape[1], -1)
        x = self.wrd2sts(x)
        x = self.sts_position(x)
        return self.sts_encoder(x, sts_mask)

    def decode(self, mem, tgt, mem_mask, tgt_mask):
        tgt = self.wrd_position(tgt)
        return self.decoder(mem, tgt, mem_mask, tgt_mask)

    def forward(self, input, tgt, wrd_mask, sts_mask, tgt_mask):  #input: [batch_size, ctx_size, sts_size]
        input_embedding = self.embeddings(input)  #[...,...,...,wrd_embed_size]
        wrd_mem = self.wrd_encode(input_embedding, wrd_mask)
        sts_mem = self.sts_encode(wrd_mem, sts_mask)
        tgt_embedding = self.embeddings(tgt)
        predict_embedding = self.decode(sts_mem, tgt_embedding, sts_mask, tgt_mask)
        return self.debeddings(predict_embedding)


class Translater(nn.Module):  #util seq2seq model
    def __init__(self, vocab_size, src_embed_size, tgt_embed_size, max_seq_size=5000, layer_n=6):
        super(Translater, self).__init__()
        self.encoder = Encoder(src_embed_size, layer_n)
        self.decoder = Decoder(tgt_embed_size, src_embed_size, layer_n)
        self.enc_position = PositionEncoder(src_embed_size, max_seq_size)
        self.tgt_position = PositionEncoder(tgt_embed_size, max_seq_size)
        self.src_embeds = Embeddings(vocab_size, src_embed_size)
        self.tgt_embeds = Embeddings(vocab_size, tgt_embed_size)
        self.debeddings = Debeddings(tgt_embed_size, vocab_size)
        for para in self.parameters():
            if para.dim() > 1:
                nn.init.xavier_uniform_(para)

    def encode(self, src, src_mask):
        src = self.enc_position(src)  #encode the position, x size: max_seq_size*embed_size
        return self.encoder(src, src_mask)

    def decode(self, memory, tgt, mem_mask, tgt_mask):
        tgt = self.tgt_position(tgt)
        return self.decoder(memory, tgt, mem_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        src_embedding = self.src_embeds(src)
        tgt_embedding = self.tgt_embeds(tgt)
        memory = self.encode(src_embedding, src_mask)
        predict_embedding = self.decode(memory, tgt_embedding, src_mask, tgt_mask)
        return self.debeddings(predict_embedding)


class Embeddings(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(Embeddings, self).__init__()
        self.embeds = nn.Embedding(vocab_size, embed_size)
        self.embed_size = embed_size

    def forward(self, x):
        return self.embeds(x) * math.sqrt(self.embed_size)


class Debeddings(nn.Module):
    def __init__(self, embed_size, vocab_size):
        super(Debeddings, self).__init__()
        self.debeds = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.debeds(x), dim=-1)


class Encoder(nn.Module):
    def __init__(self, embed_size, layer_n):
        super(Encoder, self).__init__()
        self.self_attns = clones(MultiHeadAttn(embed_size, embed_size, embed_size), layer_n)
        self.feed_forwards = clones(FeedForward(embed_size), layer_n)
        self.normss = clones(nn.LayerNorm(embed_size), layer_n * 2)

    def forward(self, src, mask):  # x: [batch_size, seq_size, embed_size]
        for i in range(len(self.self_attns)):  #residual connection
            src = self.normss[2 * i](src + self.self_attns[i](src, src, src, mask))
            src = self.normss[2 * i + 1](src + self.feed_forwards[i](src))
        return src


class Decoder(nn.Module):
    def __init__(self, embed_size, other_embed_size, layer_n):
        super(Decoder, self).__init__()
        self.self_attns = clones(MultiHeadAttn(embed_size, embed_size, embed_size), layer_n)
        self.other_attns = clones(MultiHeadAttn(embed_size, other_embed_size, embed_size), layer_n)
        self.feed_forwards = clones(FeedForward(embed_size), layer_n)
        self.normss = clones(nn.LayerNorm(embed_size), layer_n * 3)

    def forward(self, memory, tgt, mem_mask, tgt_mask):  # x/mem: [batch_size, seq_size, embed_size1/embed_size2]
        for i in range(len(self.self_attns)):  #residual connection
            tgt = self.normss[3 * i](tgt + self.self_attns[i](tgt, tgt, tgt, tgt_mask))
            tgt = self.normss[3 * i + 1](tgt + self.other_attns[i](tgt, memory, memory, mem_mask))
            tgt = self.normss[3 * i + 2](tgt + self.feed_forwards[i](tgt))
        return tgt


class MultiHeadAttn(nn.Module):
    def __init__(self, q_size, kv_size, vec_size=64, head_n=8, drop_out=0.1):
        super(MultiHeadAttn, self).__init__()
        self.vec_size = vec_size
        self.head_n = head_n
        multi_size = vec_size * head_n
        self.linear_qkv = nn.ModuleList([
            nn.Linear(q_size, multi_size),
            nn.Linear(kv_size, multi_size),  #q and kv size are embed size, not necessarily the same
            nn.Linear(kv_size, multi_size)  #multihead in one mat
        ])
        self.linear_out = nn.Linear(multi_size, q_size)
        self.dropout = nn.Dropout(drop_out)

    def attention(self, query, key, value, mask, dropout):
        # qkv: [batch_size, head_n, seq_size_q/seq_size_kv, vec_size]
        score = torch.matmul(query, key.transpose(-2, -1))
        # score: [..., ..., seq_size_q, seq_size_kv], score[..., ..., i , j] is ith word in q query the key of jth word in k
        score /= math.sqrt(query.size(-1))
        if mask is not None:
            mask = mask.unsqueeze(1)  #all headers same mask
            score = score.masked_fill(mask != False, -1e10)
        p = F.softmax(score, dim=-1)  #softmax all the key that one query has. multihead work here!
        if dropout is not None:
            p = dropout(p)
        tt = torch.matmul(p, value)
        return torch.matmul(p, value)  #shape like kv, return[..., ..., i, ...] is ith word in query's likely vec in key

    def forward(self, q, k, v, mask):  # x: [batch_size, seq_size, q_size/kv_size]
        batch_size = q.size(0)
        query, key, value = [
            linear_x(x).view(batch_size, -1, self.head_n, self.vec_size).transpose(1, 2)
            for linear_x, x in zip(self.linear_qkv, (q, k, v))
        ]  #multihead by metadata operation, qkv: [batch_size, head_n, seq_size, vec_size]
        x = self.attention(query, key, value, mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1,
                                                self.head_n * self.vec_size)  #recover dimension and memory contiguous
        return self.linear_out(x)  #return: [..., ..., q_size]


class FeedForward(nn.Module):
    def __init__(self, in_out_size, mid_size=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear_in = nn.Linear(in_out_size, mid_size)
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(mid_size, in_out_size)

    def forward(self, x):
        return self.linear_out(self.dropout(F.relu(self.linear_in(x))))  #return [..., ..., out_size]


class PositionEncoder(nn.Module):
    def __init__(self, embed_size, max_seq_size):
        super(PositionEncoder, self).__init__()
        assert (embed_size % 2 == 0)
        v1 = torch.arange(0, max_seq_size).unsqueeze(1)  #size:(max_seq_size*1)
        v2 = torch.exp(-torch.arange(0, embed_size, 2) / embed_size * math.log(10000)).unsqueeze(
            0)  # 10000^a = exp(a*log(10000)), size:(1,embed_size/2)
        pos = torch.zeros(max_seq_size, embed_size)
        mat = v1 * v2
        pos[:, 0::2] = torch.sin(mat)
        pos[:, 1::2] = torch.cos(mat)
        #pos[i,2*j]=sin(i/(10000^(2i/embed_size))),pos[i,2*j+1]=cos(i/(10000^(2i/embed_size)))
        pos = pos.unsqueeze(0)
        self.register_buffer("position", pos)

    def forward(self, x):  #add the position mat
        return x + self.position[:, :x.size(1)]


###############################################


class NoamOpt(object):
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size**(-0.5) * min(step**(-0.5), step * self.warmup**(-1.5)))


class Criterion(nn.Module):
    def __init__(self, vocab_size, opt):
        super(Criterion, self).__init__()
        self.loss_compute = nn.KLDivLoss(reduction="sum")
        self.vocab_size = vocab_size
        self.opt = opt

    def get_true_dist(self, tgt):  #tgt: [batch_n, max_seq_n]
        true_dist = torch.zeros(tgt.size(0), self.vocab_size)
        true_dist.scatter_(1, tgt.unsqueeze(1), 1)
        return true_dist

    def forward(self, predict, tgt, i):
        assert self.vocab_size == predict.size(-1)
        predict = predict.contiguous().view(-1, self.vocab_size)
        tgt = tgt.contiguous().view(-1)
        true_dist = self.get_true_dist(tgt)
        loss = self.loss_compute(predict, true_dist)
        loss.backward(retain_graph=True)
        self.opt.step()
        self.opt.optimizer.zero_grad()
        print("%d, %s" % (i, loss))
        return loss