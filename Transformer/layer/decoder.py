from Transformer.layer_norm import residual_norm
from Transformer.Attention import Attention
from Transformer.feed_forward import feed_forward
from Transformer.positional_embedding import PositionalEmbedding

class decoder_layer(nn.Module):
    def __init__(self, d_model, hidden, max_len, device, num_embeddings, embedding_dim, num_heads):
        super(decoder_layer, self).__init__()
        self.self_attention = Attention(d_model = d_model, num_heads=1)
        #self.norm = 
        #self.masked_attention = ~~
        self.encoder_decoder_attention = Attention(d_model = d_model, num_heads=1)
        self.feed_forward = feed_forward(d_model = d_model, hidden = hidden)
        self.positional_embedding = PositionalEmbedding(d_model = d_model, max_len=max_len, device=device, num_embddings=num_embeddings, embedding_dim=embedding_dim)
        self.linear = nn.Linear(d_model)
        self.softmax = nn.Softmax(d_model)

    def forward(self, x, encoder_state):
        x = PositionalEmbedding(x)
        first_x = x
        x = self.self_attention(q=x, k=x, v=x) # q=k=v = x in decoder self attention
        # x normalize
        x = first_x + x # residual learning
        
        first_x = x
        # q=x (decoder input), k=v = encoder's state in decoder enc-dec-attention
        x = self.encoder_decoder_attention(q=x, k=encoder_state, v=encoder_state)
        x = first_x + x

        first_x = x
        x = feed_forward(x)
        x = first_x + x

        x = self.linear(x)
        return self.softmax(x)
        
