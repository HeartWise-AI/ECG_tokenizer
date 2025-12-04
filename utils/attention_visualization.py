#!/usr/bin/env python3
"""
Attention visualization utilities for ECG-LLM models.
Captures and visualizes cross-modal attention between ECG tokens and text tokens.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class AttentionHook:
    """
    Hook for capturing attention weights from transformer layers.
    """
    
    def __init__(self, name: str = "attention"):
        """
        Initialize attention hook.
        
        Args:
            name: Name identifier for this hook
        """
        self.name = name
        self.attention_weights: List[torch.Tensor] = []
        self.enabled = True
    
    def __call__(self, module: nn.Module, input: Tuple, output: Tuple) -> None:
        """
        Hook function to capture attention weights.
        
        Args:
            module: The module being hooked
            input: Input to the module
            output: Output from the module (includes attention weights)
        """
        if not self.enabled:
            return
            
        # Extract attention weights from output
        # Format depends on the specific model architecture
        if isinstance(output, tuple) and len(output) > 1:
            # Usually attention weights are the second element
            if output[1] is not None:
                self.attention_weights.append(output[1].detach().cpu())
    
    def clear(self):
        """Clear stored attention weights."""
        self.attention_weights = []
    
    def disable(self):
        """Disable the hook."""
        self.enabled = False
    
    def enable(self):
        """Enable the hook."""
        self.enabled = True


class ECGAttentionVisualizer:
    """
    Visualizes attention patterns between ECG tokens and text tokens.
    """
    
    def __init__(self, 
                 num_ecg_tokens: int = 128,
                 save_dir: Optional[str] = None,
                 use_wandb: bool = False):
        """
        Initialize the attention visualizer.
        
        Args:
            num_ecg_tokens: Number of ECG tokens
            save_dir: Directory to save visualizations
            use_wandb: Whether to log to Weights & Biases
        """
        self.num_ecg_tokens = num_ecg_tokens
        self.save_dir = Path(save_dir) if save_dir else Path("attention_visualizations")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        self.attention_hooks: Dict[str, AttentionHook] = {}
    
    def register_hooks(self, model: nn.Module, layer_indices: Optional[List[int]] = None) -> None:
        """
        Register attention hooks on specified layers.
        
        Args:
            model: The model to hook
            layer_indices: Indices of layers to hook (None = all layers)
        """
        # Clear existing hooks
        self.clear_hooks()
        
        # Find attention layers
        for name, module in model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                if layer_indices is None or any(str(idx) in name for idx in layer_indices):
                    hook = AttentionHook(name)
                    module.register_forward_hook(hook)
                    self.attention_hooks[name] = hook
    
    def clear_hooks(self):
        """Clear all registered hooks."""
        for hook in self.attention_hooks.values():
            hook.clear()
        self.attention_hooks.clear()
    
    def visualize_attention(self, 
                           attention_weights: torch.Tensor,
                           input_tokens: Optional[List[str]] = None,
                           layer_name: str = "",
                           step: Optional[int] = None) -> plt.Figure:
        """
        Create attention heatmap visualization.
        
        Args:
            attention_weights: Attention weights tensor [batch, heads, seq_len, seq_len]
            input_tokens: List of input token strings
            layer_name: Name of the layer
            step: Training step number
            
        Returns:
            Matplotlib figure
        """
        # Average over batch and heads
        if attention_weights.dim() == 4:
            attention_weights = attention_weights.mean(dim=[0, 1])  # [seq_len, seq_len]
        elif attention_weights.dim() == 3:
            attention_weights = attention_weights.mean(dim=0)  # [seq_len, seq_len]
        
        # Convert to numpy
        attention_matrix = attention_weights.numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot heatmap
        if HAS_SEABORN:
            sns.heatmap(attention_matrix, 
                       cmap='Blues',
                       cbar=True,
                       square=True,
                       ax=ax,
                       vmin=0,
                       vmax=attention_matrix.max())
        else:
            # Fallback to matplotlib imshow
            im = ax.imshow(attention_matrix, cmap='Blues', aspect='auto', 
                          vmin=0, vmax=attention_matrix.max())
            plt.colorbar(im, ax=ax)
        
        # Add ECG/Text region indicators
        ax.axhline(y=self.num_ecg_tokens, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=self.num_ecg_tokens, color='red', linestyle='--', alpha=0.5)
        
        # Labels
        ax.set_xlabel('Keys (ECG → Text)')
        ax.set_ylabel('Queries (ECG → Text)')
        title = f'Attention Pattern - {layer_name}'
        if step is not None:
            title += f' (Step {step})'
        ax.set_title(title)
        
        # Add region labels
        ax.text(self.num_ecg_tokens/2, -5, 'ECG Tokens', ha='center', fontsize=10)
        ax.text(attention_matrix.shape[1] - (attention_matrix.shape[1] - self.num_ecg_tokens)/2, 
                -5, 'Text Tokens', ha='center', fontsize=10)
        
        plt.tight_layout()
        
        return fig
    
    def visualize_cross_modal_attention(self,
                                       attention_weights: torch.Tensor,
                                       layer_name: str = "",
                                       step: Optional[int] = None) -> plt.Figure:
        """
        Visualize specifically the cross-modal attention between ECG and text.
        
        Args:
            attention_weights: Full attention weights
            layer_name: Name of the layer
            step: Training step
            
        Returns:
            Matplotlib figure
        """
        # Average over batch and heads if needed
        if attention_weights.dim() == 4:
            attention_weights = attention_weights.mean(dim=[0, 1])
        elif attention_weights.dim() == 3:
            attention_weights = attention_weights.mean(dim=0)
        
        attention_matrix = attention_weights.numpy()
        
        # Extract cross-modal attention regions
        ecg_to_text = attention_matrix[:self.num_ecg_tokens, self.num_ecg_tokens:]
        text_to_ecg = attention_matrix[self.num_ecg_tokens:, :self.num_ecg_tokens]
        
        # Create subplot figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # ECG → Text attention
        if HAS_SEABORN:
            sns.heatmap(ecg_to_text,
                       cmap='Blues',
                       cbar=True,
                       ax=ax1,
                       vmin=0,
                       vmax=ecg_to_text.max())
        else:
            im1 = ax1.imshow(ecg_to_text, cmap='Blues', aspect='auto',
                            vmin=0, vmax=ecg_to_text.max())
            plt.colorbar(im1, ax=ax1)
        ax1.set_xlabel('Text Token Position')
        ax1.set_ylabel('ECG Token Position')
        ax1.set_title(f'ECG → Text Attention\n{layer_name}')
        
        # Text → ECG attention
        if HAS_SEABORN:
            sns.heatmap(text_to_ecg,
                       cmap='Reds',
                       cbar=True,
                       ax=ax2,
                       vmin=0,
                       vmax=text_to_ecg.max())
        else:
            im2 = ax2.imshow(text_to_ecg, cmap='Reds', aspect='auto',
                            vmin=0, vmax=text_to_ecg.max())
            plt.colorbar(im2, ax=ax2)
        ax2.set_xlabel('ECG Token Position')
        ax2.set_ylabel('Text Token Position')
        ax2.set_title(f'Text → ECG Attention\n{layer_name}')
        
        if step is not None:
            fig.suptitle(f'Cross-Modal Attention (Step {step})', fontsize=14)
        
        plt.tight_layout()
        
        return fig
    
    def log_attention_patterns(self, step: int) -> None:
        """
        Log all captured attention patterns.
        
        Args:
            step: Current training step
        """
        for name, hook in self.attention_hooks.items():
            if not hook.attention_weights:
                continue
            
            # Get the last attention weights
            attention_weights = hook.attention_weights[-1]
            
            # Create visualizations
            fig_full = self.visualize_attention(attention_weights, layer_name=name, step=step)
            fig_cross = self.visualize_cross_modal_attention(attention_weights, layer_name=name, step=step)
            
            # Save locally
            fig_full.savefig(self.save_dir / f'attention_full_{name}_{step}.png', dpi=100)
            fig_cross.savefig(self.save_dir / f'attention_cross_{name}_{step}.png', dpi=100)
            
            # Log to wandb if enabled
            if self.use_wandb and HAS_WANDB and wandb.run is not None:
                wandb.log({
                    f'attention/{name}_full': wandb.Image(fig_full),
                    f'attention/{name}_cross_modal': wandb.Image(fig_cross),
                }, step=step)
            
            # Close figures to free memory
            plt.close(fig_full)
            plt.close(fig_cross)
            
            # Clear the hook for next iteration
            hook.clear()
    
    def get_attention_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics about attention patterns.
        
        Returns:
            Dictionary of statistics per layer
        """
        stats = {}
        
        for name, hook in self.attention_hooks.items():
            if not hook.attention_weights:
                continue
            
            attention = hook.attention_weights[-1]
            
            # Average over batch and heads
            if attention.dim() == 4:
                attention = attention.mean(dim=[0, 1])
            elif attention.dim() == 3:
                attention = attention.mean(dim=0)
            
            # Compute cross-modal attention strength
            ecg_to_text_attn = attention[:self.num_ecg_tokens, self.num_ecg_tokens:].mean().item()
            text_to_ecg_attn = attention[self.num_ecg_tokens:, :self.num_ecg_tokens].mean().item()
            
            # Compute entropy (attention diversity)
            entropy = -(attention * (attention + 1e-10).log()).sum(dim=-1).mean().item()
            
            stats[name] = {
                'ecg_to_text_strength': ecg_to_text_attn,
                'text_to_ecg_strength': text_to_ecg_attn,
                'attention_entropy': entropy,
                'max_attention': attention.max().item(),
                'min_attention': attention.min().item(),
            }
        
        return stats


# Example usage
if __name__ == "__main__":
    # Test the visualizer
    visualizer = ECGAttentionVisualizer(num_ecg_tokens=128)
    
    # Create dummy attention weights
    batch_size = 2
    num_heads = 8
    seq_len = 256  # 128 ECG + 128 text tokens
    
    dummy_attention = torch.randn(batch_size, num_heads, seq_len, seq_len).softmax(dim=-1)
    
    # Visualize
    fig = visualizer.visualize_attention(dummy_attention, layer_name="test_layer", step=100)
    plt.show()
    
    fig_cross = visualizer.visualize_cross_modal_attention(dummy_attention, layer_name="test_layer", step=100)
    plt.show()
    
    print("Attention visualization test complete!")