from typing import Optional

import torch
from torch import nn

from cprnn.data import StressDataset


class CP_RNN(nn.Module):
    def __init__(
        self,
        state_size: int,
        dropout: float = 0.0, 
        hidden_size: int = 128, 
        n_layers: int = 2,
        rnn_type: str = 'GRU'
    ):
        super().__init__()

        self.s_in_size = StressDataset.in_size
        self.s_ou_size = StressDataset.out_size
        self.s_hs_size = hidden_size
        self.s_n_layer = n_layers

        self.h_in_size = state_size

        self.rnn_type = rnn_type

        if rnn_type == 'GRU':
            self.h_ou_size = self.s_hs_size * self.s_n_layer
            self.sstate1 = nn.GRU(
                self.s_in_size,
                self.s_hs_size,
                num_layers=self.s_n_layer,
                dropout=dropout,
                batch_first=True,
            )
        elif rnn_type == 'LMSC':
            self.h_ou_size = self.s_hs_size
            self.sstate1 = LMSC(
                self.s_in_size,
                self.s_hs_size,
                num_layers=self.s_n_layer,
                batch_first=True,
            )
        else:
            raise ValueError(f'Unknown rnn_type: {rnn_type}')

        self.sstate2 = nn.Linear(self.s_hs_size, self.s_ou_size)
        self.hstate = nn.Linear(self.h_in_size, self.h_ou_size)

    def forward(self, X, H0):
        S0 = self.hstate(H0)
        if self.rnn_type == 'GRU':
            S0 = (
                S0.reshape(H0.shape[0], self.s_hs_size, self.s_n_layer)
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


class LMSC(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        num_layers: int = 3, 
        batch_first: bool = True
    ) -> None:
        super().__init__()

        if not batch_first:
            raise NotImplementedError()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.cell = LMSCCell(self.input_size, self.hidden_size, self.num_layers)

    def forward(
        self, 
        X: torch.Tensor, 
        H0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """_summary_

        Args:
            X (torch.Tensor): 
                (N, seq_size, input_size), input vectors
            H0 (Optional[torch.Tensor], optional):
                (N, hidden_size), initial hidden state. Defaults to None.

        Returns:
            torch.Tensor: 
                (N, seq_size, hid_size), hidden state throughout sequence.
        """
        if H0 is None:
            raise NotImplementedError()
        
        out = torch.empty(X.shape[:2] + (self.hidden_size, ))
        H_prev = H0
        for i in range(X.shape[1]):
            out[:, i, :] = self.cell(X[:, i], H_prev)
            H_prev = out[:, i, :]

        return out


class LMSCCell(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        num_layers: int = 3
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layer_size = self.input_size + self.hidden_size

        self.layers = []
        for i in range(self.num_layers):
            self.layers.append((
                nn.Linear(self.layer_size, self.layer_size),
                nn.Linear(self.layer_size, self.layer_size)
            ))
        self.layers.append((
            nn.Linear(self.layer_size, self.hidden_size),
            nn.Linear(self.layer_size, self.hidden_size)
        ))

    def forward(self, X: torch.Tensor, H: Optional[torch.Tensor] = None) -> torch.Tensor:
        """_summary_

        Args:
            X (torch.Tensor): 
                (N, input_size) input vectors
            H (Optional[torch.Tensor], optional): 
                (N, hidden_size), previous hidden state. Defaults to None.

        Returns:
            Tensor: (N, hid_size), updated hidden state.
        """

        if H is None:
            raise NotImplementedError()
        
        X_norm = torch.norm(X, dim=1)
        X_dir = X.clone()
        X_dir /= X_norm[:, None]

        L = torch.cat((X_dir, H), dim=1)
        for layer in self.layers[:-1]:
            L = torch.tanh(layer[0](L)) * torch.tanh(layer[1](L))

        layer = self.layers[-1]
        alpha = torch.exp(layer[0](L))
        beta  = torch.tanh(layer[1](L))
        H_new = torch.exp(-alpha * X_norm[:, None]) * (H - beta) + beta

        return H_new
