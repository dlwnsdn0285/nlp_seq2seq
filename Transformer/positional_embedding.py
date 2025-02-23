class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len, device, num_embddings, embedding_dim):
        super(PositionalEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embddings, embedding_dim=embedding_dim) 

        self.encoding = torch.zeros(max_len, d_model, device = device) # size : max_len x d_model

        pos = torch.arange(0, max_len, device = device)
        pos = pos.unsqueeze(dim=1)  # size : max_len x 1

        step = torch.arange(0, d_model, step=2, device=device).float() # size : (d_model//2)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (step/d_model))) # 0: :2 >> 0부터 2개씩 건너뛰기
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (step/d_model))) # 1: :2 >> 1부터 2개씩 건너뛰기

    def forward(self, x):
        batch_size, seq_len = x.size()

        return self.encoding[:seq_len, :] + self.embedding


