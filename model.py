from torch import nn
from torch.utils.data import DataLoader
import torch


class DownScalingCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=37, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        a1 = self.conv1(X)
        o1 = self.relu(a1)
        a2 = self.conv2(o1)
        o2 = self.relu(a2)
        a3 = self.conv3(o2)
        o3 = self.relu(a3)
        a4 = self.conv4(o3)
        o4 = self.relu(a4)
        return o4.squeeze(axis=1)
    
def train(model: nn.Module,
    dataloader: DataLoader,
    loss_func: nn.MSELoss,
    optimizer: torch.optim,
    num_epoch: int,
    print_info: bool = True,
) -> list[float]:
    
    loss_per_epoch = []
    model.train()
    
    for epoch in range(num_epoch):
        
        epoch_loss_sum = 0
        
        for X, Y in dataloader:
            X = X.float()
            Y = Y.float()

            # run a forward pass on the neural net to get the output for this step
            output = model.forward(X)

            # set all gradients from last step to zero
            optimizer.zero_grad()
            
            # compute the loss on this step 
            loss = loss_func(output, Y)
            
            # compute backward pass and update weight for this step
            loss.backward()
            optimizer.step()
            
            # add to the loss for this epoch
            n_samples = X.shape[0]
            epoch_loss_sum += loss.item()*n_samples

        loss_per_epoch.append(epoch_loss_sum / len(dataloader.dataset))

        if print_info:
            print("Epoch " + str(epoch+1) + ", Loss: " + str(epoch_loss_sum / len(dataloader.dataset)))

    return loss_per_epoch

def test(
    model: nn.Module,
    dataloader: DataLoader,
    loss_func: nn.MSELoss,
) -> float:
    
    model.eval()
    torch.no_grad()
    
    loss_sum = 0
            
    for X,Y in dataloader:
        X = X.float()
        Y = Y.float()
        
        # run forward pass on model to get the output for this step
        output = model.forward(X)
        
        # calculate the loss for this step
        loss = loss_func(output, Y)
        
        # add to the total loss among all samples
        n_samples = X.shape[0]
        loss_sum += loss.item()*n_samples

    return loss_sum / len(dataloader.dataset)

def predict(
    model: nn.Module,
    dataloader: DataLoader,
) -> torch.Tensor:
    
    model.eval()
    torch.no_grad()
    
    all_outputs = []
            
    for X in dataloader:
        X = X.float()
        
        output = model.forward(X)
        
        all_outputs.append(output)

    return torch.cat(all_outputs, dim=0).numpy()
