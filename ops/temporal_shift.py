
import torch
import torch.nn as nn

class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=12, n_div=8):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div

    def forward(self, x):
        x = self.shift(x, self.n_segment, self.fold_div)
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment

        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        out = x.clone()

        # Unidirectional shift (real-time friendly)
        out[:, 1:, :fold] = x[:, :-1, :fold]

        return out.view(nt, c, h, w)
