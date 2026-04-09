"""
LSTM forecasting model for time-series prediction.

Supports both next-step (teacher-forced) and autoregressive multi-step prediction.
"""
import numpy as np
import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """LSTM model for time-series forecasting.

    Args:
        input_dim: Number of input features.
        output_dim: Number of output features.
        hidden_size: Number of hidden units per LSTM layer.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout rate between LSTM layers (only applied if num_layers > 1).
    """

    def __init__(self, input_dim, output_dim, hidden_size=64, num_layers=1,
                 dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x, hidden=None):
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).
            hidden: Optional (h_0, c_0) tuple.

        Returns:
            output: Predictions of shape (batch, seq_len, output_dim).
            hidden: Updated hidden state tuple.
        """
        lstm_out, hidden = self.lstm(x, hidden)
        output = self.fc(lstm_out)
        return output, hidden

    def detach_hidden(self, hidden):
        """Detach hidden states from the computation graph for TBPTT."""
        h, c = hidden
        return (h.detach(), c.detach())


def _make_sequences(inputs, outputs, seq_len):
    """Create overlapping (input_seq, target_seq) pairs for training.

    Args:
        inputs: Array of shape (num_samples, input_dim).
        outputs: Array of shape (num_samples, output_dim).
        seq_len: Length of each subsequence.

    Returns:
        X: Tensor of shape (num_seqs, seq_len, input_dim).
        Y: Tensor of shape (num_seqs, seq_len, output_dim).
    """
    n = len(inputs)
    X, Y = [], []
    for i in range(0, n - seq_len + 1, seq_len):
        end = i + seq_len
        X.append(inputs[i:end])
        Y.append(outputs[i:end])
    return (torch.tensor(np.array(X), dtype=torch.float32),
            torch.tensor(np.array(Y), dtype=torch.float32))


def train_lstm(model, train_in, train_out, val_in, val_out,
               seq_len=50, lr=1e-3, weight_decay=0.0,
               epochs=200, patience=20, device='cpu',
               input_noise=0.0):
    """Train the LSTM model with early stopping on validation loss.

    Args:
        model: LSTMForecaster instance.
        train_in: Training inputs, shape (num_samples, input_dim).
        train_out: Training targets, shape (num_samples, output_dim).
        val_in: Validation inputs, shape (num_samples, input_dim).
        val_out: Validation targets, shape (num_samples, output_dim).
        seq_len: Subsequence length for training.
        lr: Learning rate.
        weight_decay: L2 regularization.
        epochs: Maximum training epochs.
        patience: Early stopping patience.
        device: 'cpu' or 'cuda'.
        input_noise: Std of Gaussian noise added to training inputs each
            epoch. Simulates the perturbed inputs seen during autoregressive
            rollout, improving robustness to error compounding.

    Returns:
        best_val_loss: Best validation MSE achieved.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    criterion = nn.MSELoss()

    X_train, Y_train = _make_sequences(train_in, train_out, seq_len)
    X_val, Y_val = _make_sequences(val_in, val_out, seq_len)
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_val, Y_val = X_val.to(device), Y_val.to(device)

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    n_train_chunks = X_train.shape[0]
    n_val_chunks = X_val.shape[0]

    for epoch in range(epochs):
        # --- Training with TBPTT (sequential hidden state carry-over) ---
        model.train()
        optimizer.zero_grad()
        hidden = None
        total_loss = 0.0
        for i in range(n_train_chunks):
            chunk = X_train[i:i+1]
            if input_noise > 0:
                chunk = chunk + torch.randn_like(chunk) * input_noise
            pred, hidden = model(chunk, hidden)
            total_loss = total_loss + criterion(pred, Y_train[i:i+1])
            hidden = model.detach_hidden(hidden)
        total_loss = total_loss / n_train_chunks
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # --- Validation with sequential hidden state carry-over ---
        model.eval()
        with torch.no_grad():
            hidden = None
            # Warm up on training data
            for i in range(n_train_chunks):
                _, hidden = model(X_train[i:i+1], hidden)
            # Evaluate on validation chunks
            val_loss = 0.0
            for i in range(n_val_chunks):
                val_pred, hidden = model(X_val[i:i+1], hidden)
                val_loss += criterion(val_pred, Y_val[i:i+1]).item()
            val_loss /= n_val_chunks

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)
    return best_val_loss


def predict_lstm(model, test_in, device='cpu'):
    """Next-step (teacher-forced) prediction.

    Args:
        model: Trained LSTMForecaster.
        test_in: Test inputs, shape (num_samples, input_dim).
        device: 'cpu' or 'cuda'.

    Returns:
        predictions: numpy array of shape (num_samples, output_dim).
    """
    model.eval()
    model = model.to(device)
    x = torch.tensor(test_in, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred, _ = model(x)
    return pred.squeeze(0).cpu().numpy()


def predict_lstm_autoregressive(model, initial_input, num_steps, device='cpu',
                                warmup_data=None):
    """Autoregressive multi-step prediction.

    Feeds the model's own output back as the next input for num_steps.

    Args:
        model: Trained LSTMForecaster.
        initial_input: First input, shape (input_dim,).
        num_steps: Number of steps to predict.
        device: 'cpu' or 'cuda'.
        warmup_data: Optional array of shape (num_samples, input_dim) to
            drive through the model first (teacher-forced) to warm up the
            hidden state before autoregressive prediction.

    Returns:
        predictions: numpy array of shape (num_steps, output_dim).
    """
    model.eval()
    model = model.to(device)
    hidden = None

    # Warm up hidden state by teacher-forcing through warmup data
    if warmup_data is not None:
        x = torch.tensor(warmup_data, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            _, hidden = model(x, hidden)

    predictions = []
    current = torch.tensor(initial_input, dtype=torch.float32).reshape(1, 1, -1).to(device)

    with torch.no_grad():
        for _ in range(num_steps):
            out, hidden = model(current, hidden)
            pred = out[:, -1, :]
            predictions.append(pred.cpu().numpy().flatten())
            current = pred.unsqueeze(1)

    return np.array(predictions)
