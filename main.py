import torch
from utils import get_proxys_dataloader, get_prediction_dataloader
from model import DownScalingCNN, train, test
import matplotlib.pyplot as plt
import numpy as np


def main():
    training_dataloader, testing_dataloader = get_proxys_dataloader()
    prediction_samples = get_prediction_dataloader()

    model = train_model(training_dataloader, testing_dataloader)
    save_model(model, "downscaling_cnn.pth")

    #model = load_model("downscaling_cnn.pth")

    visualize_upscaled_results(model, training_dataloader)
    visualize_predictions(model, prediction_samples)
    
    
    
def train_model(training_dataloader, testing_dataloader) -> torch.nn.Module:
    print("Training CNN")
    
    num_epochs = 10
    learning_rate = 0.001
    
    model = DownScalingCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = torch.nn.MSELoss()
    
    training_losses = train(model, training_dataloader, loss_func, optimizer, num_epochs, True)
    testing_loss = test(model, testing_dataloader, loss_func)
    
    print("Testing Loss: ", testing_loss)
    
    return model
    
def save_model(model: torch.nn.Module, filepath: str):
    torch.save(model.state_dict(), filepath)

def load_model(filepath: str):
    print("Loading Model")
    model = DownScalingCNN()
    model.load_state_dict(torch.load(filepath))
    model.eval()
    return model

def visualize_upscaled_results(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader):
    model.eval()
    torch.no_grad()
    
    model_outputs = []
    raw_irradiance = []
    original_irradiance = []
    
    for X, Y in dataloader:
        X = X.float()
        Y = Y.float()

        output = model.forward(X)

        model_outputs.append(output.detach().numpy().squeeze())
        raw_irradiance.append(Y.detach().numpy().squeeze())
        original_irradiance.append(X[:, -1, :, :].detach().numpy().squeeze())

    fig, axs = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(8):
        h = axs[i//4, i%4].imshow(original_irradiance[i][200:400, 300:500], cmap = 'viridis', vmin=0, vmax=100)
        fig.colorbar(h, ax=axs[i//4, i%4])
        axs[i//4, i%4].set_title('Sample ' + str(i+1))
    fig.suptitle("Original Upscaled Irradiance Data")
    fig.tight_layout()
    plt.show()

    fig, axs = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(8):
        h = axs[i//4, i%4].imshow(model_outputs[i][200:400, 300:500], cmap = 'viridis', vmin=0, vmax=100)
        fig.colorbar(h, ax=axs[i//4, i%4])
        axs[i//4, i%4].set_title('Sample ' + str(i+1))
    fig.suptitle("Model Predictions Using Proxies")
    fig.tight_layout()
    plt.show()
    
    fig, axs = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(8):
        diff = model_outputs[i] - raw_irradiance[i]
        h = axs[i//4, i%4].imshow(diff, cmap = 'viridis', vmin=-1, vmax=1)
        fig.colorbar(h, ax=axs[i//4, i%4])
        axs[i//4, i%4].set_title('Sample ' + str(i+1))
    fig.suptitle("Difference Between Model Predictions and Raw Irradiance Data for Training Samples")
    fig.tight_layout()
    plt.show()

def visualize_predictions(model, dataloader):
    model.eval()
    torch.no_grad()
    
    model_outputs = []
    original_irradiance = []
    for X in dataloader:
        X = X.float()

        output = model.forward(X)

        model_outputs.append(output.detach().numpy().squeeze())
        original_irradiance.append(X[:, -1, :, :].detach().numpy().squeeze())

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    for i in range(2):
        h = axs[i%2, i//2].imshow(original_irradiance[i+4][135:335, 275:475], cmap = 'viridis', vmin=0, vmax=100)
        fig.colorbar(h, ax=axs[i%2, i//2], label='Irradiance (W/m^2)')
        axs[i%2, i//2].set_title('Original Irradiance Data')
        
        axs[i%2, i//2].set_xticks([])
        axs[i%2, i//2].set_yticks([])

        h = axs[(i+2)%2, (i+2)//2].imshow(model_outputs[i+4][135:335, 275:475], cmap = 'viridis', vmin=0, vmax=100)
        fig.colorbar(h, ax=axs[(i+2)%2, (i+2)//2], label='Irradiance (W/m^2)')
        axs[(i+2)%2, (i+2)//2].set_title('Downscaling Model Predictions')
        axs[(i+2)%2, (i+2)//2].set_xticks([])
        axs[(i+2)%2, (i+2)//2].set_yticks([])
        
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
