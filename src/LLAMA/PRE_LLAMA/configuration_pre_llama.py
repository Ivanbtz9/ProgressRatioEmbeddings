# coding=utf-8
"""PreLLaMA model configuration"""
import transformers
from transformers import LlamaConfig


class PreLlamaConfig(LlamaConfig):
    model_type = "pre-llama"

    def __init__(
        self,
        pre_status: bool = True,       
        gaussian_noise: bool = True,   
        M_shannon: float = None,       
        lambda_noise: float = None,    
        **kwargs
    ):
        super().__init__(**kwargs)

        self.pre_status = pre_status
        self.gaussian_noise = gaussian_noise

        # Defaults derived from hidden_size (set by super().__init__)
        self.M_shannon = M_shannon if M_shannon is not None else self.hidden_size / 2
        self.lambda_noise = lambda_noise if lambda_noise is not None else 2.0 / self.hidden_size
        
        self.transformers_version = transformers.__version__
        self.pad_token_id = 128002

__all__ = ["PreLlamaConfig"]