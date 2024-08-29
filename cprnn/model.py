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


class CP_RNNSingleStep(CP_RNN):
    def __init__(self, state_size, dropout=0, hidden_size=128, n_layers=2):
        super().__init__(state_size, dropout, hidden_size, n_layers)

    def forward(self, X, H):
        """_summary_

        Args:
            X (torch.Tensor, (n, s_in_size)): Input
            H (torch.Tensor, (s_n_layer, n, s_hs_size)): Current state
            note n/batch axis can be ommitted from all

        Return:
            Y (torch.Tensor, (n, s_ou_size)): Output
            H (torch.Tensor, (s_n_layer, n, s_hs_size)): New state
        """
        H_all, H_final = self.sstate1(X[..., None, :], H)
        Y = self.sstate2(H_all)
        return Y[..., 0, :], H_final
