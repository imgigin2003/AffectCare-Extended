import torch
import sys
import os

# Add the src directory to the Python path to import model.py
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from model import ElderCareModel

def export_model_to_onnx(model_path="src/best_model.pth", output_path="best_model.onnx"):
    # Load the PyTorch model
    model = ElderCareModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval() # Set the model to evaluation mode

    # Create a dummy input tensor with the expected shape (batch_size, channels, height, width)
    # Based on predict.py, the input shape is (1, 1, 13, 173)
    dummy_input = torch.randn(1, 1, 13, 173, requires_grad=True)

    # Export the model to ONNX
    torch.onnx.export(
        model,                       # model being run
        dummy_input,                 # model input (or a tuple for multiple inputs)
        output_path,                 # where to save the model (file or file-like object)
        export_params=True,          # store the trained parameter weights inside the model file
        opset_version=11,            # the ONNX version to export the model to
        do_constant_folding=True,    # whether to execute constant folding for optimization
        input_names=['input'],       # the name to assign to the input node
        output_names=['output'],     # the name to assign to the output node
        dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                      'output' : {0 : 'batch_size'}})

    print(f"Model successfully exported to {output_path}")

if __name__ == "__main__":
    # Ensure the model path is correct
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(base_dir, "src", "best_model.pth")
    onnx_output_file = os.path.join(base_dir, "best_model.onnx")

    if not os.path.exists(model_file):
        print(f"Error: Model file not found at {model_file}. Please ensure you have trained the model and placed 'best_model.pth' in the 'ML-Backend/src/' directory.")
        sys.exit(1)

    export_model_to_onnx(model_file, onnx_output_file)
