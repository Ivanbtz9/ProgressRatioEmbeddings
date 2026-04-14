# coding=utf-8

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable
from typing import Optional, Union
import sys, os

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging

from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding, LlamaPreTrainedModel, LlamaDecoderLayer


from .configuration_pre_llama import PreLlamaConfig


logger = logging.get_logger(__name__)



@torch.no_grad()     
def pre_sinusoidal(config:PreLlamaConfig, cache_position:torch.Tensor, max_input_len:int, target_len:torch.LongTensor) -> torch.Tensor:

    target_len = torch.clamp(target_len, min=1, max=config.max_position_embeddings).unsqueeze(-1)
    
    ratio = torch.clamp(
        (cache_position - max_input_len).clamp(min=0).unsqueeze(0) / target_len,
        0.0, 1.0
    )

    if config.gaussian_noise:
        ratio_mask = (ratio > 0.0)
        noise = config.lambda_noise * torch.randn_like(ratio,device=target_len.device)
        ratio = ratio_mask*torch.clamp(ratio + noise,0,1) 

    omegas = config.M_shannon * ratio.unsqueeze(-1) #(Batch_size, max_len, 1)
    
    indices = torch.arange(config.hidden_size, device=ratio.device)//2 #(0,0,1,1,..., d_model/2-1, d_model/2-1)   
    x_j = (2 * indices /config.hidden_size).unsqueeze(0).unsqueeze(0) #(1, 1, d_model)

    pre = omegas * x_j  # (Batch_size, max_len, d_model)

    pre[..., 0::2] = torch.cos(pre[..., 0::2])
    pre[..., 1::2] = torch.sin(pre[..., 1::2])

    return pre



@auto_docstring
class PreLlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: PreLlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.pre_gate = nn.Parameter(torch.zeros(1)) 

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        max_input_len: Optional[int] = None,
        target_len: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    )-> BaseModelOutputWithPast:
        """
        Forward pass of the PreLlama base model.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, seq_len)`, *optional*):
                Indices of input sequence tokens in the vocabulary.
            attention_mask (`torch.Tensor` of shape `(batch_size, seq_len)`, *optional*):
                Mask to avoid attending to padding tokens. 1 = keep, 0 = mask.
            position_ids (`torch.LongTensor` of shape `(batch_size, seq_len)`, *optional*):
                Indices of positions for each input token. Defaults to `cache_position`.
            past_key_values (`Cache`, *optional*):
                Cached key/value states for autoregressive generation.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
                Token embeddings. Mutually exclusive with `input_ids`.
            cache_position (`torch.LongTensor` of shape `(seq_len,)`, *optional*):
                Absolute position indices of the current tokens in the full sequence.
                Used to compute the PRE progress ratio.
            use_cache (`bool`, *optional*):
                Whether to return and update `past_key_values`.
            max_input_len (`int`, *optional*):
                Length of the (padded) input prompt. Used as the boundary between
                prompt and generated tokens when computing the PRE progress ratio.
                If `None` or if `pre_status=False`, PRE is not applied.
            target_len (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Desired output length for each sample in the batch. Used to
                normalise the PRE progress ratio: `r_t = (t - max_input_len + 1) / target_len`.
                If `None` or if `pre_status=False`, PRE is not applied.

        Returns:
            `BaseModelOutputWithPast`
        """
            
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        
        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None: #Never None at the generation time 
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            # print("past_seen_tokens : ", past_seen_tokens)
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
            # print("cache_position : ", cache_position)
        # print("cache_position : ", cache_position)
        

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
            # print("position_ids", position_ids)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        if self.config.pre_status and max_input_len is not None and target_len is not None:
            pre = pre_sinusoidal(self.config, cache_position, max_input_len, target_len)           
            hidden_states = inputs_embeds + torch.tanh(self.pre_gate) * pre.to(dtype=inputs_embeds.dtype)
            # print(pre)
            # print(pre.shape)
            # sys.exit()
        else:
            hidden_states = inputs_embeds
 
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        # print("position_embeddings", position_embeddings)
        # sys.exit()

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class PreLlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config:PreLlamaConfig):
        super().__init__(config)
        self.model = PreLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        max_input_len: Optional[int] = None,
        target_len:Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Forward pass of the PreLlama causal language model.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, seq_len)`, *optional*):
                Indices of input sequence tokens in the vocabulary.
            attention_mask (`torch.Tensor` of shape `(batch_size, seq_len)`, *optional*):
                Mask to avoid attending to padding tokens. 1 = keep, 0 = mask.
            position_ids (`torch.LongTensor` of shape `(batch_size, seq_len)`, *optional*):
                Indices of positions for each input token.
            past_key_values (`Cache`, *optional*):
                Cached key/value states for autoregressive generation.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
                Token embeddings. Mutually exclusive with `input_ids`.
            labels (`torch.LongTensor` of shape `(batch_size, seq_len)`, *optional*):
                Labels for language modelling loss. Positions with `-100` are ignored.
            use_cache (`bool`, *optional*):
                Whether to return and update `past_key_values`.
            cache_position (`torch.LongTensor` of shape `(seq_len,)`, *optional*):
                Absolute position indices of the current tokens in the full sequence.
            logits_to_keep (`int` or `torch.Tensor`, *optional*, defaults to 0):
                Number of last-token logits to compute. 0 means all tokens.
            max_input_len (`int`, *optional*):
                Length of the (padded) input prompt. Passed through to `PreLlamaModel`
                to compute the PRE progress ratio boundary.
            target_len (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Desired output length per sample. Passed through to `PreLlamaModel`
                to normalise the PRE progress ratio.

        Returns:
            `CausalLMOutputWithPast`

        Example:

        ```python
        >>> from transformers import AutoTokenizer, PreLlamaForCausalLM

        >>> model = PreLlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```

        """
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            max_input_len=max_input_len,
            target_len=target_len,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "PreLlamaForCausalLM",
    "PreLlamaModel",

]
