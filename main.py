from torch import nn, optim
from torch.optim import Adam
from tqdm import tqdm
from data import *


import math, time

from seed_holder import my_seed_everywhere
from Transformer.transformer_model import Transformer
from Transformer.positional_embedding import PositionalEmbedding

my_seed_everywhere(seed=42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
# parameters of transformer #
d_model=
max_len=
device=
num_embeddings=
embedding_dim=
hidden=
num_heads=

learning_rate = 1e-3 # 논문에서는 learning rate를 계속 바꿈. >> lr_scheduler 사용하면 learning rate 변경 가능
#############################

model = Transformer(d_model=d_model, 
                        max_len=max_len,
                        device=device,   # 'cuda' or 'cpu' 
                        num_embeddings=num_embeddings, 
                        embedding_dim=embedding_dim,
                        hidden=hidden,
                        num_heads=num_heads)
optimizer = Adam(lr = learning_rate)
criterion = nn.CrossEntropyLoss()   # ignore하는 pad가 존재 >> mask에 사용?

# dataloader
# 

def idx_to_word():
    #

def get_bleu():
    #


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()
        output = model(src, trg[:, :-1]) # 마지막 하나는 예측
        # output_reshape
        # trg =

        loss = criterion()  
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())
    return epoch_loss/len(iterator)



def evaluate(model, iterator, criterion, batch_size):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            total_bleu = []
            for j in range(batch_size):
                try:
                    trg_words = idx_to_word(batch.trg[j], loader.target.vocab)
                    output_words = output[j].max(dim=1)[1]
                    output_words = idx_to_word(output_words, loader.target.vocab)
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                except:
                    pass

            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / len(iterator), batch_bleu
    



if __name__ == "__main__":
    