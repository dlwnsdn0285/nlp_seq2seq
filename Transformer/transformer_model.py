# transformer model for seq2seq

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import datasets
import tqdm
import evaluate
from Transformer.layer.encoder import encoder_layer
from Transformer.layer.decoder import decoder_layer
from Transformer.positional_embedding import PositionalEmbedding
from Transformer.Attention import Attention

# dataset = datasets.load_dataset()
# data loader

# 1. one layer, one num_head, without masked attention (사용 유무에 따른 test 성능 비교)
# 2. masked attention
# 3. multi layer, multi head attention (num_heads > 1)

###TODO###
# masked attention for decoder
# dataset > dataloader

class Transformer(nn.Module):
    # 파라미터 추가해주기
    def __init__(self, d_model, max_len, device, num_embeddings, embedding_dim, hidden, num_heads):
        super(Transformer, self).__init__()
        self.encoder = encoder_layer(d_model=d_model, 
                                           hidden=hidden, 
                                           max_len=max_len, 
                                           device=device, 
                                           num_embeddings=num_embeddings, 
                                           embedding_dim=embedding_dim, 
                                           num_heads=num_heads)
        self.decoder = decoder_layer(d_model=d_model, 
                                           hidden=hidden, 
                                           max_len=max_len, 
                                           device=device, 
                                           num_embeddings=num_embeddings, 
                                           embedding_dim=embedding_dim, 
                                           num_heads=num_heads)
        
    def forward(self, src, trg):
        encoder_res = self.encoder(src)
        decoder_res = self.decoder(trg, encoder_res)
        return decoder_res
        

# A = PositionalEmbedding(d_model = 3, max_len = 5, device = device, num_embddings=5, embedding_dim=3)














