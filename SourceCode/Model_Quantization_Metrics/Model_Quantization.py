import torch
import torch.nn as nn
import torch.quantization
import numpy as np
import os
from typing import Dict, Tuple
import json
from collections import OrderedDict
import matplotlib.pyplot as plt

# Import your RNN model architecture
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, h):
        out, h = self.rnn(x, h)
        out = self.fc(out.reshape(out.size(0) * out.size(1), out.size(2)))
        return out, h

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

def load_model() -> Tuple[RNNModel, Dict, Dict]:
    """Load the trained model and mappings"""
    char_to_idx = torch.load(R'SourceCode\Algorithms\RNN\RNN_Modules\char_to_idx.pth')
    idx_to_char = torch.load(R'SourceCode\Algorithms\RNN\RNN_Modules\idx_to_char.pth')
    
    input_size = len(char_to_idx)
    hidden_size = 64
    num_layers = 2
    model = RNNModel(input_size, hidden_size, num_layers, input_size)
    
    model.load_state_dict(torch.load(R'SourceCode\Algorithms\RNN\RNN_Modules\rnn_model.pth'))
    return model, char_to_idx, idx_to_char

def analyze_model_size(model: nn.Module) -> Dict:
    """Analyze the model size and parameter statistics"""
    total_params = sum(p.numel() for p in model.parameters())
    total_size = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # Analyze parameter distribution
    param_stats = {
        'total_parameters': total_params,
        'model_size_mb': total_size / (1024 * 1024),
        'parameter_types': {}
    }
    
    for name, param in model.named_parameters():
        param_type = str(param.dtype)
        if param_type not in param_stats['parameter_types']:
            param_stats['parameter_types'][param_type] = {
                'count': 0,
                'size_mb': 0
            }
        
        param_stats['parameter_types'][param_type]['count'] += param.numel()
        param_stats['parameter_types'][param_type]['size_mb'] += (param.numel() * param.element_size()) / (1024 * 1024)
    
    return param_stats

def plot_parameter_distribution(param_stats: Dict, save_path: str):
    """Plot parameter distribution statistics"""
    plt.figure(figsize=(15, 10), dpi=300)
    
    # Parameter count by type
    types = list(param_stats['parameter_types'].keys())
    counts = [stats['count'] for stats in param_stats['parameter_types'].values()]
    
    plt.subplot(2, 1, 1)
    plt.bar(types, counts)
    plt.title('Parameter Count by Data Type')
    plt.ylabel('Number of Parameters')
    plt.xticks(rotation=45)
    
    # Size by type
    sizes = [stats['size_mb'] for stats in param_stats['parameter_types'].values()]
    
    plt.subplot(2, 1, 2)
    plt.bar(types, sizes)
    plt.title('Memory Usage by Data Type')
    plt.ylabel('Size (MB)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'parameter_distribution.png'))
    plt.close()

def quantize_model(model: nn.Module) -> nn.Module:
    """Perform static quantization on the model"""
    # Configure model for quantization
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # Prepare model for quantization
    model_prepared = torch.quantization.prepare(model)
    
    # Convert to quantized model
    model_quantized = torch.quantization.convert(model_prepared)
    
    return model_quantized

def compare_models(original_model: nn.Module, quantized_model: nn.Module) -> Dict:
    """Compare original and quantized models"""
    original_stats = analyze_model_size(original_model)
    quantized_stats = analyze_model_size(quantized_model)
    
    comparison = {
        'original_size_mb': original_stats['model_size_mb'],
        'quantized_size_mb': quantized_stats['model_size_mb'],
        'size_reduction': (1 - quantized_stats['model_size_mb'] / original_stats['model_size_mb']) * 100,
        'original_params': original_stats['total_parameters'],
        'quantized_params': quantized_stats['total_parameters']
    }
    
    return comparison

def plot_model_comparison(comparison: Dict, save_path: str):
    """Plot model size comparison"""
    plt.figure(figsize=(12, 6), dpi=300)
    
    models = ['Original', 'Quantized']
    sizes = [comparison['original_size_mb'], comparison['quantized_size_mb']]
    
    plt.bar(models, sizes)
    plt.title('Model Size Comparison')
    plt.ylabel('Size (MB)')
    
    # Add size reduction text
    plt.text(0.5, max(sizes) * 0.5, 
             f"Size Reduction: {comparison['size_reduction']:.2f}%",
             horizontalalignment='center',
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'model_comparison.png'))
    plt.close()

def main():
    output_dir = "SourceCode/Model_Evaluation_Metrics"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading model...")
    model, char_to_idx, idx_to_char = load_model()
    
    print("Analyzing original model...")
    original_stats = analyze_model_size(model)
    plot_parameter_distribution(original_stats, output_dir)
    
    print("Quantizing model...")
    quantized_model = quantize_model(model)
    
    print("Comparing models...")
    comparison = compare_models(model, quantized_model)
    plot_model_comparison(comparison, output_dir)
    
    # Save statistics
    with open(os.path.join(output_dir, 'quantization_analysis.json'), 'w') as f:
        json.dump({
            'original_model_stats': original_stats,
            'model_comparison': comparison
        }, f, indent=4)
    
    print(f"Analysis complete. Results saved to {output_dir}")
    
    # Print summary
    print("\nModel Quantization Summary:")
    print(f"Original Model Size: {comparison['original_size_mb']:.2f} MB")
    print(f"Quantized Model Size: {comparison['quantized_size_mb']:.2f} MB")
    print(f"Size Reduction: {comparison['size_reduction']:.2f}%")
    print(f"Original Parameters: {comparison['original_params']:,}")
    print(f"Quantized Parameters: {comparison['quantized_params']:,}")

if __name__ == "__main__":
    main() 