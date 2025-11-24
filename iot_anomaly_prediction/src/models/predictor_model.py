import torch
import torch.nn as nn

class GRUPredictor(nn.Module):
    """
    Defines the GRU (Gated Recurrent Unit) model architecture for time-series prediction.
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout_prob=0.3):
        super(GRUPredictor, self).__init__()
        
        # The GRU layer processes the input sequence.
        # batch_first=True means the input tensor shape is (batch, sequence_length, features)
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )
        
        # A dropout layer for regularization to prevent overfitting
        self.dropout = nn.Dropout(dropout_prob)
        
        # The final fully connected layer that makes the binary prediction
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # The GRU returns the output for each time step and the final hidden state.
        # We only need the output of the very last time step in the sequence.
        gru_out, _ = self.gru(x)
        
        # Pass the output of the last time step through the dropout and linear layers
        out = self.dropout(gru_out[:, -1, :])
        out = self.fc(out)
        return out

