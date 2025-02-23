class Attention(nn.Module):
    def __init__(self, d_model, num_heads=1): # 일단 num_head = 1
        super(Attention, self).__init__()

        self.w_q = nn.Linear(d_model, d_model // num_heads)
        self.w_k = nn.Linear(d_model, d_model // num_heads)
        self.w_v = nn.Linear(d_model, d_model // num_heads)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        
        batch_size, head, length, d_tensor = k.size() # ?
        k_t = k.transpose(2, 3) # batch_size, head, length, d_tensor 중 length와 d_tensor 바꿈
        d_k = d_model // num_heads
        score = (q @ k_t) / math.sqrt(d_k) # @ : matmul

        attention_res = self.softmax(score) @ v

        return attention_res