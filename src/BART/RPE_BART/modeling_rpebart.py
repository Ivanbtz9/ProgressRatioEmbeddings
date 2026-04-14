"""PyTorch RpeBART model."""
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
    BartConfig, 
    BartPreTrainedModel
)
from transformers.utils import logging
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.bart.modeling_bart import shift_tokens_right


_CHECKPOINT_FOR_DOC = "facebook/bart-large"
_CONFIG_FOR_DOC = "BartConfig"

logger = logging.get_logger(__name__)

class RpeBartLearnedReversePositionalEmbedding(nn.Embedding):
    """
    This module learns reverse positional embeddings.
    """
    def __init__(self, 
                 num_embeddings: int, 
                 embedding_dim: int,
                 pad_id: int=1,#the pad id associated to the model pad_token_id
                 learned_weights: bool=True, 
                 freeze: bool=True, 
                 gaussian_noise: bool= True):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        self.pad_id = pad_id
        self.gaussian_noise = gaussian_noise
        weight = None
        if learned_weights:
            weight = self._sinusoidal_weigth(num_embeddings + self.offset, embedding_dim)

        super().__init__(num_embeddings + self.offset, embedding_dim,_weight=weight,_freeze=freeze)
    
    def _sinusoidal_weigth(
            self,
            num_embeddings: int, 
            embedding_dim: int)->torch.tensor:
        """Return the weigths associated to the usual sinusoidal positional embedding weigths """

        pe = torch.zeros(num_embeddings, embedding_dim)
        position = torch.arange(0, num_embeddings, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(-2 * (torch.arange(0, embedding_dim) // 2) / embedding_dim * math.log(10000.0))

        pe[:, 0::2] = torch.sin(position * div_term[0::2])
        pe[:, 1::2] = torch.cos(position * div_term[1::2])

        return pe
    
    def _get_reverse_position_ids(
            self,
            input_ids:torch.LongTensor,
            target_len:Optional[torch.LongTensor]=None
        )->torch.LongTensor:

        """Computes reversed position indices for the decoder inputs."""
        mask = ~torch.isin(input_ids,torch.tensor([self.pad_id], device=input_ids.device))
        bsz, seq_len = input_ids.shape[:2]
        
        if target_len is None:
            reversed_position_input  = torch.ones(mask.shape, device=self.weight.device) * mask
            reversed_position_input = torch.flip(torch.flip(reversed_position_input , dims=(1,)).cumsum(dim=1), dims=(1,))        
        else:
            k = torch.arange(seq_len, device=self.weight.device)  # Create a tensor [0, 1, 2, ..., seq_len-1]
            reversed_position_input = F.relu(target_len.unsqueeze(1) - k)  # Broadcast subtraction over all positions

        if self.gaussian_noise:
            normal_round = torch.randn(reversed_position_input.shape, device=self.weight.device) * mask
        else:
            normal_round = torch.zeros(reversed_position_input.shape, device=self.weight.device)
        
        return torch.abs(torch.round(reversed_position_input  + normal_round)).to(torch.long)
    
    def forward(self, 
                input_ids:torch.LongTensor,
                target_len:Optional[torch.LongTensor]=None):
        """`input_ids' shape is expected to be [bsz x seqlen]."""
        positions = self._get_reverse_position_ids(input_ids,target_len)
        return super().forward(positions)


class RpeBartForConditionalGeneration(BartPreTrainedModel, GenerationMixin):
    base_model_prefix = "model"
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "re_pos_emb", "lm_head.weight"]
    _keys_to_ignore_on_load_missing = ["final_logits_bias"]

    def __init__(self, config:BartConfig, re_emb_status:bool=True):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))) #not updated by backpropagation
        self.re_pos_emb = RpeBartLearnedReversePositionalEmbedding(config.max_position_embeddings,config.d_model)
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.re_emb_status = re_emb_status
        # Initialize weights and apply final processing
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
    
    def get_reverse_positional_embeddings(self):
        return self.re_pos_emb

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings 

    def _get_reverse_decoder_inputs_embeds(
        self,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        target_len: Optional[torch.Tensor] = None,) -> Tuple[Optional[torch.Tensor], Optional[torch.LongTensor]]:
        
        if decoder_input_ids is not None:

            # Compute position embeddings
            position_embeddings = self.model.decoder.embed_positions(decoder_input_ids)  # shape-based
            # Compute token embeddings + positionals
            token_embeddings = self.model.decoder.embed_tokens(decoder_input_ids)

            if self.re_emb_status:
                reverse_position_embeddings = self.re_pos_emb(decoder_input_ids, target_len)
                decoder_inputs_embeds =  token_embeddings  + position_embeddings + reverse_position_embeddings 
            else:
                decoder_inputs_embeds = token_embeddings + position_embeddings
            
            decoder_input_ids = None

        return decoder_input_ids, decoder_inputs_embeds  

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
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
        
        decoder_input_ids, decoder_inputs_embeds = self._get_reverse_decoder_inputs_embeds(decoder_input_ids, decoder_inputs_embeds, target_len)

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
        if labels is not None:
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
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids, decoder_inputs_embeds = self._get_reverse_decoder_inputs_embeds(decoder_input_ids, decoder_inputs_embeds, target_len)        
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

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

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




