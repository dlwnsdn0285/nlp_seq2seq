class feed_forward(nn.Module):
    def __init__(self, d_model, hidden): # d_model : 임베딩 사이즈, hidden : hidden unit size
        super(feed_forward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        # dropout?
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        # dropout?

        return x