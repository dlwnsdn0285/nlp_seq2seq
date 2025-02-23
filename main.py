from torch import nn, optim
from torch.optim import Adam

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

def train():
    model.train()
    epoch_loss = 0



def evaluate():



if __name__ == "__main__":