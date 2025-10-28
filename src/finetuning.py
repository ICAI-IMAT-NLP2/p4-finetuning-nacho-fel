import torch
import torch.nn as nn
import math

try:
    from utils import download_and_load_model
except:
    from src.utils import download_and_load_model

class LoRA(nn.Module):
    def __init__(self, original_layer, r=4, alpha=32):
        """
        Low-Rank Adaptation (LoRA) module.
        
        Args:
            original_layer (nn.Module): The original layer to which LoRA is applied.
            r (int): Rank of the low-rank approximation.
            alpha (int): Scaling factor for the LoRA module.
        """
        super().__init__()
        # TODO: Initialize LoRA parameters
        self.r = r
        self.alpha = alpha
        self.original_layer = original_layer

        # TODO: Low-rank matrices A and B for LoRA
        self.B = nn.Parameter(torch.zeros((r,  original_layer.weight.shape[1])))
        self.A = nn.Parameter(torch.zeros((original_layer.weight.shape[0], r)))

        # TODO: Initialize LoRA weights (B is zero-initialized, A is random)
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)
        
        # TODO: Scaling factor alpha 
        self.scaling = self.alpha / self.r

        # TODO: Freeze the original layer parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False
                
    def forward(self, x):
        # TODO: Perform forward pass with low-rank update
        x_base = self.original_layer(x)

        lora_update = (x @ self.A @ self.B)
        
        return x_base + self.scaling*lora_update

def inject_lora_into_model(model, r=4, alpha=32, device='cpu'):
    """
    Inject LoRA layers into the linear layers of the attention modules of the model.
    
    Args:
        model (PreTrainedModel): The pre-trained model.
        r (int): Rank of the low-rank approximation.
        alpha (int): Scaling factor for LoRA.
        device (torch.device): The device to run the model on ('cuda' or 'cpu').
    
    Returns:
        model (PreTrainedModel): The model with LoRA injected into attention layers.
    """

    # TODO: Iterate through all child modules of the model
    for child_name, child_module in model.named_children():
        # TODO: Check if the child module is a linear layer of the attention module
        if child_name.lower() in ["q", "k", "v", "o"]:
            # TODO: Create LoRA layer for linear module
            lora_layer = LoRA(child_module, r,alpha)
            setattr(model, child_name, lora_layer)

        else:
            # TODO: Recursively inject LoRA into child module
            inject_lora_into_model(child_module, r, alpha, device)
    return model.to(device)


class SoftPromptEmbedding(nn.Module):
    def __init__(self, prompt_length: int, model_hidden_size: int):
        """
        Crea 'soft prompts' entrenables para anteponer a los embeddings de entrada.

        Args:
            prompt_length (int): Nº de tokens virtuales del soft prompt.
            model_hidden_size (int): Hidden size del modelo pre-entrenado.
        """
        super().__init__()
        # TODO: Initialize soft prompt embeddings
        self.soft_prompt = nn.Parameter(torch.zeros(prompt_length, model_hidden_size))
        nn.init.normal_(self.soft_prompt, mean=0.0, std=0.02)

    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Prepend soft prompts a los embeddings de entrada.

        Args:
            input_embeddings (torch.Tensor): Tensor [batch, seq_len, hidden_size].

        Returns:
            torch.Tensor: Tensor [batch, prompt_length + seq_len, hidden_size].
        """
        batch_size = input_embeddings.size(0)

        # TODO: Expand soft prompt to match batch size
        soft_prompt_expanded = (
            self.soft_prompt.unsqueeze(0)  # [1, P, H]
            .expand(batch_size, -1, -1)    # [B, P, H]
        )

        # TODO: Concatenate soft prompt and input
        return torch.cat([soft_prompt_expanded, input_embeddings], dim=1)