import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block

class AttentionModulator(nn.Module):
    def __init__(
        self,
        distraction_level=0.0,
        hyperfocus_tokens=None,
        avoid_tokens=None,
        tokenizer=None,
        hyperfocus_strength=1.8,
        avoid_strength=0.01
    ):
        super().__init__()
        self.distraction_level = distraction_level
        self.hyperfocus_strength = hyperfocus_strength
        self.avoid_strength = avoid_strength
        if tokenizer:
            self.hyperfocus_token_ids = set(
                tid for tok in (hyperfocus_tokens or [])
                for tid in tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tok))
            )
            self.avoid_token_ids = set(
                tid for tok in (avoid_tokens or [])
                for tid in tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tok))
            )
        else:
            self.hyperfocus_token_ids = set()
            self.avoid_token_ids = set()

    def modulate(self, attn_weights, input_ids):
        if self.distraction_level > 0:
            noise = torch.randn_like(attn_weights) * self.distraction_level
            attn_weights = attn_weights + noise
        if input_ids is not None:
            bsz, nhead, qlen, klen = attn_weights.size()
            for b in range(bsz):
                for pos in range(klen):
                    tid = input_ids[b, pos].item()
                    if tid in self.hyperfocus_token_ids:
                        attn_weights[b, :, :, pos] *= (1 + self.hyperfocus_strength * 0.1)
                    if tid in self.avoid_token_ids:
                        attn_weights[b, :, :, pos] *= (1 - self.avoid_strength * 0.1)
        return F.softmax(attn_weights, dim=-1)


class CustomAttention(GPT2Attention):
    def __init__(self, config, modulator=None, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        self.modulator = modulator
        self.input_ids = None

    def _attn(self, query, key, value, attention_mask=None, head_mask=None, **kwargs):
        attn_output, attn_weights = super()._attn(
            query, key, value,
            attention_mask=attention_mask,
            head_mask=head_mask,
            **kwargs
        )
        if self.modulator and self.input_ids is not None:
            attn_weights = self.modulator.modulate(attn_weights, self.input_ids)
            attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights

class CustomBlock(GPT2Block):
    def __init__(self, config, modulator=None, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        self.attn = CustomAttention(config, modulator=modulator, layer_idx=layer_idx)

class GPT2AutismModel(GPT2LMHeadModel):
    def __init__(self, config, modulator=None, sae=None, reforcar_ids=None, punir_ids=None):
        super().__init__(config)
        base = GPT2LMHeadModel.from_pretrained("gpt2-medium", config=config)
        self.transformer = base.transformer
        self.lm_head = base.lm_head
        self.modulator = modulator
        self.sae = sae
        self.reforcar_ids = reforcar_ids
        self.punir_ids = punir_ids
        new_blocks = []
        for i, block in enumerate(self.transformer.h):
            custom = CustomBlock(config, modulator=modulator, layer_idx=i)
            custom.load_state_dict(block.state_dict())
            new_blocks.append(custom)
        self.transformer.h = nn.ModuleList(new_blocks)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        for blk in self.transformer.h:
            blk.attn.input_ids = input_ids
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        if self.sae is not None: 
            emb = outputs.last_hidden_state[:, -1, :]
            x_recon, _, _ = self.sae(
                emb,
                reforcar_ids=self.reforcar_ids,
                punir_ids=self.punir_ids,
                return_hidden=True
            )
            alpha = 0.7
            mixed = alpha * x_recon + (1 - alpha) * emb
            lm_logits = self.lm_head(mixed.unsqueeze(1)).squeeze(1)

        lm_logits = self.lm_head(outputs.last_hidden_state)

        if self.modulator and self.modulator.hyperfocus_token_ids:
            bias = lm_logits.new_zeros(lm_logits.size(-1))
            bias[list(self.modulator.hyperfocus_token_ids)] = self.modulator.hyperfocus_strength * 2.0
            lm_logits = lm_logits + bias

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=getattr(outputs, "cross_attentions", None)
        )