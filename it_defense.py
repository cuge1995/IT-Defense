import torch.nn as nn

class Infer(nn.Module):
    def __init__(self, model):
        super(Infer, self).__init__()
        self.model = model
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        x.data = x[:, :, torch.randperm(x.size()[2])].data
        x = self.model(x)
        return x
model = Infer(model)
