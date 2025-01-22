from torch import nn
import torch
class Softmax(nn.Module):
    def __init__(self,inputNeurons, hlayer1,outputs):
        super(Softmax,self).__init__()
        self.inputNeurons = inputNeurons
        self.inputLayer = nn.Linear(inputNeurons , hlayer1)
        self.Hlayer1 = nn.Linear(hlayer1 , 300)
        self.Hlayer2 = nn.Linear(300 , 50)
        self.fc = nn.Linear(50,outputs)
    def forward(self,x):
        x = x.view(-1,self.inputNeurons)
        x  = torch.nn.functional.relu(self.inputLayer(x))
        x = torch.nn.functional.relu(self.Hlayer1(x))
        x = torch.nn.functional.relu(self.Hlayer2(x))
        x = self.fc(x)
        return x