from Transformer.layer_norm import residual_norm
from Transformer.Attention import Attention
from Transformer.feed_forward import feed_forward
from Transformer.positional_embedding import PositionalEmbedding

class encoder_layer(nn.Module):
    def __init__(self, d_model, hidden, max_len, device, num_embeddings, embedding_dim, num_heads):
        super(encoder_layer, self).__init__()
        self.self_attention = Attention(d_model = d_model, num_heads=1)
        #self.norm = 
        self.feed_forward = feed_forward(d_model = d_model, hidden = hidden)
        self.positional_embedding = PositionalEmbedding(d_model = d_model, max_len=max_len, device=device, num_embddings=num_embeddings, embedding_dim=embedding_dim)

    def forward(self, x):
        x = PositionalEmbedding(x)
        first_x = x
        x = self.self_attention(q=x, k=x, v=x) # q=k=v = x in encoder self attention
        # x normalize
        x = first_x + x # residual learning
        first_x = x
        x = feed_forward(x)
        
        return x + first_x
        

