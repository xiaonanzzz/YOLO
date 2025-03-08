import sys
from pathlib import Path
import torch
import argparse

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from yolo.tools.solver import BaseModel
from yolo.config.config import Config
import hydra


def convert_checkpoint_to_weights(cfg, checkpoint_path, output_path, verbose=True):
    """
    Convert a Lightning checkpoint to standard weights format
    
    Args:
        cfg: Configuration object
        checkpoint_path: Path to the Lightning checkpoint file
        output_path: Path where to save the exported weights
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    # Load the checkpoint

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if verbose:
        print('checkpoint keys:', checkpoint.keys())

    # Create a model instance
    model = BaseModel(cfg)
    
    # Extract model state dict from the checkpoint
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        # Remove 'model.' prefix if it exists in the keys
        clean_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        
        # Get the base model (without Lightning wrapper)
        base_model = model.model.model # this one is the YOLO.model part
        
        # Load the cleaned state dict
        base_model.load_state_dict(clean_state_dict, strict=False)
        # calculate the base_model keys and clean_state_dict keys, show the ones in base_model but not in clean_state_dict
        base_model_keys = set(base_model.state_dict().keys())
        clean_state_dict_keys = set(clean_state_dict.keys())
        print('base_model keys: ', len(base_model_keys))
        print('clean_state_dict keys:', len(clean_state_dict_keys))
        lack_set = base_model_keys - clean_state_dict_keys
        print('base_model keys not in clean_state_dict:', len(lack_set))
        print('lack keys:', lack_set)
        
        # Save the model weights
        to_save_dict = base_model.state_dict()
        print('to_save_dict keys:', len(to_save_dict.keys()))
        torch.save(to_save_dict, output_path)
        print(f"Weights successfully exported to {output_path}")
    else:
        print("Error: Invalid checkpoint format")


def export_onnx(cfg, checkpoint_path, output_path, input_shape=(1, 3, 640, 640)):
    """
    Export model to ONNX format
    
    Args:
        cfg: Configuration object
        checkpoint_path: Path to the checkpoint file
        output_path: Path where to save the ONNX model
        input_shape: Input shape for the model (batch_size, channels, height, width)
    """
    print(f"Exporting model to ONNX format from {checkpoint_path}")
    
    # Create a model instance and load weights
    model = TrainModel(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        clean_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        base_model = model.model
        base_model.load_state_dict(clean_state_dict, strict=False)
        
        # Set model to evaluation mode
        base_model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        
        # Export to ONNX
        torch.onnx.export(
            base_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        print(f"ONNX model successfully exported to {output_path}")
    else:
        print("Error: Invalid checkpoint format")


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: Config):
    task = getattr(cfg.task, "export_type", "weights")
    checkpoint_path = getattr(cfg.task, "checkpoint_path", None)
    
    if checkpoint_path is None:
        print("Error: checkpoint_path must be specified for export")
        return
    
    if task == "weights":
        output_path = getattr(cfg.task, "output_path", "exported_weights.pt")
        convert_checkpoint_to_weights(cfg, checkpoint_path, output_path)
    elif task == "onnx":
        output_path = getattr(cfg.task, "output_path", "exported_model.onnx")
        input_shape = getattr(cfg.task, "input_shape", (1, 3, 640, 640))
        export_onnx(cfg, checkpoint_path, output_path, input_shape)
    else:
        print(f"Unsupported export type: {task}")


if __name__ == "__main__":
    main()
