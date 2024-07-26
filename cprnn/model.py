from torch import nn

from cprnn.data import StressDataset


class CP_RNN(nn.Module):
    def __init__(self, state_size, dropout=0.0, hidden_size=128, n_layers=2):
        super().__init__()

        self.s_in_size = StressDataset.in_size
        self.s_ou_size = StressDataset.out_size
        self.s_hs_size = hidden_size
        self.s_n_layer = n_layers

        self.h_in_size = state_size
        self.h_ou_size = self.s_hs_size * self.s_n_layer

        self.sstate1 = nn.GRU(
            self.s_in_size,
            self.s_hs_size,
            num_layers=self.s_n_layer,
            dropout=dropout,
            batch_first=True,
        )
        self.sstate2 = nn.Linear(self.s_hs_size, self.s_ou_size)
        self.hstate = nn.Linear(self.h_in_size, self.h_ou_size)

    def forward(self, X, H0):
        S0 = (
            self.hstate(H0)
            .reshape(H0.shape[0], self.s_hs_size, self.s_n_layer)
            .permute((2, 0, 1))
            .contiguous()
        )
        Y, _ = self.sstate1(X, S0)
        Y = self.sstate2(Y)
        return Y
