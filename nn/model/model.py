import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(x.size(0), self.hidden_dim).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[-1])
        out = torch.sigmoid(out)
        return out


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers


def train(model, train_dl, num_epochs, opt, loss_fn):
    losses = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        loss_per_epoch = []
        for i, (xb, yb) in enumerate(train_dl):
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            opt.zero_grad()
            loss = loss_fn(pred, yb)
            loss_per_epoch.append(loss.detach().item())
            loss.backward()
            opt.step()

        losses.append(loss_per_epoch.sum() / len(loss_per_epoch))
        print(f'Epoch: {epoch} | Loss: {losses[-1]}')
    return losses
