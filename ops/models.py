
import torch
import torch.nn as nn
from torchvision import models
from .temporal_shift import TemporalShift

class TSN(nn.Module):
    def __init__(self, num_class, num_segments=12):
        super(TSN, self).__init__()
        self.num_segments = num_segments

        self.base_model = models.mobilenet_v2(pretrained=True)

        # Apply TSM to selected layers (not all)
        for i, m in enumerate(self.base_model.features):
            if isinstance(m, nn.Sequential):
                self.base_model.features[i] = TemporalShift(
                    m,
                    n_segment=num_segments,
                    n_div=8
                )

        # Replace classifier
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.base_model.last_channel, num_class)
        )

    def forward(self, input):
        b, s, c, h, w = input.size()

        input = input.view(-1, c, h, w)
        base_out = self.base_model(input)

        base_out = base_out.view(b, s, -1)

        # Temporal consensus (average)
        return base_out.mean(dim=1)
