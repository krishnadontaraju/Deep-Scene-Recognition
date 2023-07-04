import torch
import torch.nn as nn

from torchvision.models import alexnet


class MyAlexNet(nn.Module):
  def __init__(self):
    '''
    Init function to define the layers and loss function

    Note: Use 'sum' reduction in the loss_criterion. Ready Pytorch documention
    to understand what it means

    Note: Do not forget to freeze the layers of alexnet except the last one

    Download pretrained alexnet using pytorch's API (Hint: see the import
    statements)
    '''
    super().__init__()

    self.cnn_layers = nn.Sequential()
    self.fc_layers = nn.Sequential()
    self.loss_criterion = None

    model = alexnet(pretrained=True)
    self.cnn_layers = nn.Sequential(*list(model.features.children())[:])
    for param in self.cnn_layers.parameters():
        param.requires_grad = False
    self.fc_layers = nn.Sequential(*list(model.classifier.children())[:-1],
                                   nn.Linear(4096, 15))
    for i in [1, 4]:
        for param in self.fc_layers[i].parameters():
            param.requires_grad = False
    self.loss_criterion = nn.CrossEntropyLoss()



  def forward(self, x: torch.tensor) -> torch.tensor:
    '''
    Perform the forward pass with the net

    Args:
    -   x: the input image [Dim: (N,C,H,W)]
    Returns:
    -   y: the output (raw scores) of the net [Dim: (N,15)]
    '''

    model_output = None
    x = x.repeat(1, 3, 1, 1) # as AlexNet accepts color images


    x = self.cnn_layers(x)
    x = torch.flatten(x, 1)
    model_output = self.fc_layers(x)

    return model_output
