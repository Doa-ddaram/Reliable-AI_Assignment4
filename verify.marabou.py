import time
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms
from maraboupy import Marabou, MarabouCore

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

def verify_with_marabou(onnx_path, epsilon=0.05):
    # Load MNIST test dataset to get a sample image
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Take the very first image and its label
    image, true_label = test_dataset[0]
    image_np = image.view(1, -1).numpy() # Flatten to match Marabou input
    
    print(f"Testing image 0. True label: {true_label}")
    
    # Target other classes to see if the model can be fooled
    other_classes = [i for i in range(10) if i != true_label]
    
    start_time = time.time()
    
    # Check vulnerability against each incorrect class
    for target_class in other_classes:
        print(f"Checking if model can misclassify as class {target_class}...")
        
        # Read the ONNX model into Marabou
        network = Marabou.read_onnx(onnx_path)
        inputVars = network.inputVars[0].flatten()
        outputVars = network.outputVars[0].flatten()
        
        # Set input constraints (L-infinity norm bound)
        for i in range(784):
            val = float(image_np[0][i])
            lower_bound = max(0.0, val - epsilon)
            upper_bound = min(1.0, val + epsilon)
            network.setLowerBound(int(inputVars[i]), lower_bound)
            network.setUpperBound(int(inputVars[i]), upper_bound)
            
        # Set output constraints: We want to find a case where Target Class > True Class
        # The inequality: -1 * Output(target) + 1 * Output(true) <= -0.001
        # This mathematically means: Output(target) - Output(true) >= 0.001
        network.addInequality([int(outputVars[target_class]), int(outputVars[true_label])], [-1, 1], -0.001)
        
        # Solve the query using Marabou
        # Disable verbose output to make the console cleaner
        options = Marabou.createOptions(verbosity=0)
        exitCode, vals, stats = network.solve(options=options)
        
        if exitCode == "sat":
            # "sat" means the constraints are satisfied -> The model CAN be fooled!
            print(f"[Result] UNSAFE: Found an adversarial example for class {target_class}!")
            end_time = time.time()
            print(f"Verification Time: {end_time - start_time:.4f} seconds")
            return
            
    # If all incorrect classes return "unsat"
    print("[Result] SAFE: Verified robust against all target classes.")
    end_time = time.time()
    print(f"Total Verification Time: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    pth_model_path = 'mlp_mnist.pth'
    onnx_model_path = 'mlp_mnist.onnx'
    
    # 1. Export the trained model to ONNX
    export_to_onnx(pth_model_path, onnx_model_path)
    
    # 2. Run Marabou Verification
    print("Starting Marabou Verification...")
    verify_with_marabou(onnx_model_path, epsilon=0.01)