# models/lstm.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    """
    Pures LSTM für tabellarische Zeitserien.
    Erwartet:
      - x.shape == (B, F) -> wird auf (B, seq_len, input_dim) reshaped
      - oder (B, T, C) mit T==seq_len, C==input_dim
    """
    def __init__(
        self,
        input_dim: int,
        seq_len: int = 64,
        hidden_size: int = 64,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.1,
        use_last_timestep: bool = True,  # alternativ: Mean-Pooling über Zeit
        **kwargs
    ):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.bidirectional = bidirectional
        self.use_last_timestep = use_last_timestep

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        out_dim = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        # Einfache Architektur für besseres Lernen
        self.fc1 = nn.Linear(out_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        
        # Bessere Initialisierung für LSTM-Stabilität
        self._init_weights()

    def _ensure_seq(self, x: torch.Tensor) -> torch.Tensor:
        # Bringt x auf (B, T, C) mit T=self.seq_len, C=self.input_dim
        if x.dim() == 2:
            b, f = x.size(0), x.size(1)
            target = self.seq_len * self.input_dim
            if f >= target:
                x = x[:, :target]
            else:
                pad = torch.zeros(b, target - f, device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=1)
            x = x.view(b, self.seq_len, self.input_dim)
        elif x.dim() == 3:
            b, t, c = x.shape
            assert c == self.input_dim, f"input_dim mismatch: got {c}, expected {self.input_dim}"
            if t >= self.seq_len:
                x = x[:, :self.seq_len, :]
            else:
                pad = torch.zeros(b, self.seq_len - t, c, device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=1)
        else:
            raise ValueError(f"Unsupported x.dim(): {x.dim()}")
        return x

    def forward(self, x):
        x = self._ensure_seq(x)  # (B, T, C)
        output, (h_n, c_n) = self.lstm(x)

        if self.bidirectional:
            # letzte Layer forward/backward konkatten
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, 2H)
        else:
            h_last = h_n[-1]  # (B, H)

        if not self.use_last_timestep:
            # optional: Mean-Pooling über Zeit (robuster bei Rauschen)
            h_last = output.mean(dim=1)  # (B, H*dirs)

        x = self.dropout(h_last)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # (B, 1) - Keine Aktivierung für Regression
        return x
    
    def _init_weights(self):
        """Xavier/Glorot Initialisierung für bessere LSTM-Konvergenz"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    torch.nn.init.xavier_uniform_(param)
                else:
                    torch.nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                torch.nn.init.constant_(param, 0)
