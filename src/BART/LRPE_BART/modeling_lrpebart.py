"""PyTorch LRPE-BART model."""
import copy
import math
import logging
import sys, os
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import (
    GenerationMixin,
    BartModel,  
    BartPreTrainedModel
)
from transformers.utils import logging
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.bart.modeling_bart import shift_tokens_right

from .configuration_lrpebart import LRPEBartConfig


logger = logging.get_logger(__name__)


class LRPEmbedding(nn.Module):

    def __init__(self, 
                 embedding_dim: int, 
                 gaussian_noise: bool= True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.gaussian_noise = gaussian_noise
        self.lambda_noise = (2 / self.embedding_dim)

    @torch.no_grad()     
    def _sinusoidal_weight(self,rates:torch.Tensor, target_len:torch.Tensor) -> torch.Tensor:
        """Return padded tensor of sinusoidal length ratio positional embedding 
        of shape (B, max_len, embedding_dim)."""

        indices = torch.arange(self.embedding_dim, device=rates.device)//2 

        x_j = 1-(2*indices/self.embedding_dim).unsqueeze(0).unsqueeze(0)#(1, 1, d_model) 

        len_ = target_len.unsqueeze(-1).unsqueeze(-1)

        lrpe = rates * len_**x_j  # (Batch_size, max_len, d_model)

        lrpe[:,:,0::2] = torch.cos(lrpe[:,:,0::2])
        lrpe[:,:,1::2] = torch.sin(lrpe[:,:,1::2])

        return lrpe

    def forward(self,
                target_len:torch.LongTensor,
                max_len: Optional[int] = None)-> torch.Tensor:
        
        if max_len is None:
            max_len = target_len.max().item()

        rates = torch.stack([torch.cat([torch.linspace(0,1,l),torch.ones((max_len - l))]) for l in target_len],dim=0).to(target_len.device).unsqueeze(-1) #(Batch_size, max_len, 1)

        if self.gaussian_noise:
            noise = self.lambda_noise * torch.randn_like(rates,device=target_len.device)
            rates = torch.clamp(rates + noise,0,1) 

        return self._sinusoidal_weight(rates,target_len)
            

class LRPEBartForConditionalGeneration(BartPreTrainedModel, GenerationMixin):
    base_model_prefix = "model"
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]
    _keys_to_ignore_on_load_missing = ["final_logits_bias"]
    config_class = LRPEBartConfig

    def __init__(self, 
                 config:LRPEBartConfig,
                 lrpe_emb_status:bool=True,
                 gaussian_noise:bool=False,
                 compute_loss_status:bool=True):
        
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))) #not updated by backpropagation
        
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        
        self.lrpe_emb_status = lrpe_emb_status
        self.lrpe_model = LRPEmbedding(embedding_dim=config.d_model,gaussian_noise=gaussian_noise)
        self.compute_loss_status = compute_loss_status
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens:int, pad_to_multiple_of:Optional[int]=None) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings 

    def _get_lrpe(
        self,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        target_len: Optional[torch.Tensor] = None,) -> Tuple[Optional[torch.Tensor], Optional[torch.LongTensor]]:
        
        if decoder_input_ids is not None:

            # Compute position embeddings
            position_embeddings = self.model.decoder.embed_positions(decoder_input_ids)  # shape-based 

            # Compute token embeddings 
            token_embeddings = self.model.decoder.embed_tokens(decoder_input_ids) #index-based

            

            if self.lrpe_emb_status and target_len is not None:
                # print(self.lrpe_emb_status,target_len)
                max_len = None
                batch_size, seq_len, dim_emb = token_embeddings.shape
                if seq_len > target_len.max().item(): #allow the possible longer generation as expected 
                    max_len = seq_len
                len_rate_position_embeddings = self.lrpe_model(target_len,max_len=max_len)
                # print(token_embeddings.shape,position_embeddings.shape,len_rate_position_embeddings.shape)
                decoder_inputs_embeds =  token_embeddings  + position_embeddings + len_rate_position_embeddings[:batch_size,:seq_len,:dim_emb] 
            else:
                decoder_inputs_embeds = token_embeddings + position_embeddings
            
            decoder_input_ids = None

        return decoder_input_ids, decoder_inputs_embeds

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,#it can be None 
        return_dict: Optional[bool] = None,
        target_len:Optional[torch.Tensor]=None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                #A way to create decoder_input_ids with labels 
                decoder_input_ids = self.prepare_decoder_input_ids_from_labels(labels)
        # print("decoder_input_ids",decoder_input_ids)
        # print("target_len",target_len)
        decoder_input_ids, decoder_inputs_embeds = self._get_lrpe(decoder_input_ids, decoder_inputs_embeds, target_len)
        # print("decoder_input_ids",decoder_input_ids)
        # sys.exit()

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )    

        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        if labels is not None and self.compute_loss_status:
            labels = labels.to(lm_logits.device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        decoder_inputs_embeds=None,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        target_len=None,
        **kwargs,
        ):

        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            # Get length of previously cached sequence
            past_length = past_key_values[0][0].shape[2]  # shape: (batch, heads, seq_len, dim)

            # Cut decoder_input_ids to only the newly added token(s)
            if decoder_input_ids.shape[1] > past_length: # decoder_input_ids.shape : (batch, seq_len, 1)
                remove_prefix_length = past_length
            else:
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids, decoder_inputs_embeds = self._get_lrpe(decoder_input_ids, decoder_inputs_embeds, target_len)        
            decoder_inputs_embeds = decoder_inputs_embeds[:, remove_prefix_length:,:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "decoder_inputs_embeds":decoder_inputs_embeds,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past