import time
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms
from maraboupy import Marabou

# Define the same MLP model structure
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # Flatten the input tensor (e.g., 28x28 image to 784 vector)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
def export_to_onnx(pytorch_path, onnx_path):
    # Load PyTorch model
    model = MLP()
    model.load_state_dict(torch.load(pytorch_path))
    model.eval()
    
    # Create a dummy input tensor matching the MNIST image shape
    dummy_input = torch.randn(1, 1, 28, 28)
    
    # Export to ONNX format
    torch.onnx.export(model, dummy_input, onnx_path, 
                      export_params=True, 
                      opset_version=11, 
                      do_constant_folding=True, 
                      input_names=['input'], 
                      output_names=['output'])
    print(f"Model successfully exported to {onnx_path}")
    
def build_model(model_weights_path=None, **kwargs):
    # Initialize the model
    model = MLP()
    
    # Load weights if a valid path is provided
    if model_weights_path:
        model.load_state_dict(torch.load(model_weights_path))
        
    return model

def prepare_data():
    #mnist dataset
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Create DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_dataloader, test_dataloader

if __name__ == "__main__":
    # train the model here
    train_loader, test_loader = prepare_data()
    
    import os
    model_weights_path = os.path.join(os.path.dirname(__file__), 'mlp_mnist.pth')
    
    if os.path.exists(model_weights_path):
        print(f"Weights file found! Loading weights from: {model_weights_path}")
        # Build model with pre-trained weights
        model = build_model(model_weights_path=model_weights_path)
    else:
        print("Weights file not found. Initializing a new model from scratch.")
        # Build a blank model
        model = build_model(model_weights_path=None)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        
        model.train()
        for epoch in range(10):  # number of epochs
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                accuracy = (output.argmax(dim=1) == target).float().mean()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch [{epoch+1}/10], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}')
        
        model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            correct += (output.argmax(dim=1) == target).sum().item()
            
    print(f'Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {correct/len(test_loader.dataset):.4f}')
    
    torch.save(model.state_dict(), 'mlp_mnist.pth')