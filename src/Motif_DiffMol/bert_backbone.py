import torch
import torch.nn as nn
import math
from typing import Optional
from functools import partial

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_ATTN_AVAILABLE = True
except ImportError:
    FLEX_ATTN_AVAILABLE = False
    create_block_mask = None

try:
    import einops
    from einops import rearrange
except ImportError:
    einops = None
    rearrange = None

def unit_diff_mask(b, h, q_idx, kv_idx, unit_size=None, n=None):
    x0_flag_q = (q_idx >= n)
    x0_flag_kv = (kv_idx >= n)
    unit_q = torch.where(x0_flag_q == 1, (q_idx - n) // unit_size,q_idx // unit_size)
    unit_kv = torch.where(x0_flag_kv == 1, (kv_idx - n) // unit_size,kv_idx // unit_size)
    unit_diagonal = (unit_q == unit_kv) & (x0_flag_q == x0_flag_kv)
    offset_unit_causal = ((unit_q > unit_kv) & (x0_flag_kv == 1) & (x0_flag_q == 0))
    unit_causal = (unit_q >= unit_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)
    return unit_diagonal | offset_unit_causal | unit_causal


def create_unit_diff_mask_with_padding(unit_size, n, padding_mask=None):

    def combined_mask_fn(b, h, q_idx, kv_idx):

        unit_allowed = unit_diff_mask(b, h, q_idx, kv_idx, unit_size=unit_size, n=n)
        if padding_mask is not None:
            q_valid = padding_mask[b, q_idx].bool()
            kv_valid = padding_mask[b, kv_idx].bool()

            return unit_allowed & q_valid & kv_valid

        return unit_allowed

    return combined_mask_fn

@torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
def fused_flex_attention(q, k, v, mask=None):
    if FLEX_ATTN_AVAILABLE:
        return flex_attention(q, k, v, unit_mask=mask)
    else:
        raise RuntimeError("flex_attention not available")

class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False
        )
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False
        )

    def forward(
            self,
            input_ids: torch.LongTensor,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
    ):
        seq_length = input_ids.size(1)

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = self.token_type_ids[:, :seq_length]

        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        embeddings = word_embeds + position_embeds + token_type_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config, attn_backend='flash_attn', unit_size=4, max_seqlen=512):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.hidden_size = config.hidden_size
        self.attn_backend = attn_backend
        self.unit_size = unit_size
        self.max_seqlen = max_seqlen
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def get_qkv(self, hidden_states: torch.Tensor):
        
        batch_size = hidden_states.size(0)
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        qkv = torch.cat([q, k, v], dim=-1)  # (B, seq_len, hidden_size * 3)

        if rearrange is not None:
            qkv = rearrange(
                qkv,
                'b s (three h d) -> b s three h d',
                three=3,
                h=self.num_attention_heads
            )
        else:
            qkv = qkv.view(batch_size, -1, 3, self.num_attention_heads, self.attention_head_size)

        return qkv

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
    ):
        batch_size = hidden_states.size(0)
        seq_len = hidden_states.size(1)

        
        qkv = self.get_qkv(hidden_states)

        
        if self.attn_backend == 'flash_attn' and attention_mask is None:
            x = self._flash_attention(qkv, batch_size, seq_len)
        elif self.attn_backend == 'flex' and FLEX_ATTN_AVAILABLE:
            x = self._flex_attention(qkv, attention_mask)
        elif self.attn_backend == 'sdpa':
            x = self._sdpa_attention(qkv, attention_mask)
        else:
            # qkv: (B, S, 3, H, D)
            q, k, v = qkv.unbind(dim=2)  # (B, S, H, D)
            q = rearrange(q, 'b s h d -> b h s d')  # (B, H, S, D)
            k = rearrange(k, 'b s h d -> b h s d')  # (B, H, S, D)
            v = rearrange(v, 'b s h d -> b h s d')  # (B, H, S, D)

            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))  # [B, H, S, S]

            
            if attention_mask is not None:
                
                mask = attention_mask[:, None, None, :].to(dtype=torch.bool, device=attn_scores.device)
                attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

            attn_probs = torch.softmax(attn_scores, dim=-1)
            x = torch.matmul(attn_probs, v)  # (B, H, S, D)
            x = rearrange(x, 'b h s d -> b s (h d)')  # (B, S, H*D)

        return x

    def _flash_attention(self, qkv, batch_size, seq_len):
        try:
            import flash_attn

            if rearrange is None:
                raise ImportError("einops required for flash attention")

            qkv = rearrange(qkv, 'b s ... -> (b s) ...')
            cu_seqlens = torch.arange(
                0, (batch_size + 1) * seq_len, step=seq_len,
                dtype=torch.int32, device=qkv.device)
            x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
                qkv, cu_seqlens, seq_len, 0., causal=False)
            x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)
            return x
        except ImportError as e:
            print(f"Warning: Flash attention not available ({e}), falling back to SDPA")
            return self._sdpa_attention(qkv, None)

    def _flex_attention(self, qkv, mask):
        if not FLEX_ATTN_AVAILABLE:
            print("Warning: flex_attention not available, falling back to SDPA")
            return self._sdpa_attention(qkv, mask)

        if rearrange is None:
            print("Warning: einops not available, falling back to SDPA")
            return self._sdpa_attention(qkv, mask)

        # (B, seq_len, 3, num_heads, head_dim) -> (B, num_heads, 3, seq_len, head_dim)
        qkv = rearrange(qkv, 'b s three h d -> b h three s d', h=self.num_attention_heads)

        
        x = fused_flex_attention(
            qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2],
            mask=mask)

        x = rearrange(x, 'b h s d -> b s (h d)')
        return x

    def _sdpa_attention(self, qkv, mask: Optional[torch.Tensor] = None):

        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        
        attn_mask_sdpa = None
        if mask is not None:
            attn_mask_sdpa = mask.unsqueeze(1).unsqueeze(2)

        import torch.nn.functional as F
        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            
            attn_mask=attn_mask_sdpa,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False  
        )

        context = context.transpose(1, 2).contiguous()
        
        new_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(new_shape)

        return context


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config, attn_backend='flash_attn', unit_size=4, max_seqlen=512):
        super().__init__()
        self.self = BertSelfAttention(config, attn_backend, unit_size, max_seqlen)
        self.output = BertSelfOutput(config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            sample_mode: bool = False,
    ):
        self_outputs = self.self(hidden_states, attention_mask, sample_mode)
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config, attn_backend='flash_attn', unit_size=4, max_seqlen=512):
        super().__init__()
        self.attention = BertAttention(config, attn_backend, unit_size, max_seqlen)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            sample_mode: bool = False,
    ):
        attention_output = self.attention(hidden_states, attention_mask, sample_mode)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config, attn_backend='flash_attn', unit_size=4, max_seqlen=512):
        super().__init__()
        self.layer = nn.ModuleList([
            BertLayer(config, attn_backend, unit_size, max_seqlen)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            sample_mode: bool = False,
    ):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask, sample_mode)
        return hidden_states


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class BertBackbone(nn.Module):
    def __init__(self, config, vocab_size: int):
        super().__init__()

        class SimpleConfig:
            def __init__(self, model_config, vocab_size):
                self.vocab_size = vocab_size
                self.hidden_size = model_config.hidden_size
                self.num_hidden_layers = model_config.num_hidden_layers
                self.num_attention_heads = model_config.num_attention_heads
                self.intermediate_size = getattr(model_config, 'intermediate_size', 4 * self.hidden_size)
                self.hidden_dropout_prob = model_config.hidden_dropout_prob
                self.attention_probs_dropout_prob = model_config.attention_probs_dropout_prob
                self.max_position_embeddings = getattr(model_config, 'max_position_embeddings', 512)
                self.type_vocab_size = getattr(model_config, 'type_vocab_size', 2)
                self.layer_norm_eps = 1e-12
                self.pad_token_id = 0

        bert_config = SimpleConfig(config.model, vocab_size)

        self.config = config
        self.n = config.model.length
        self.unit_size = config.unit_size
        self.max_seqlen = getattr(config.model, 'max_seqlen', 512)
        self.attn_backend = getattr(config.model, 'attn_backend', 'flash_attn')

        
        self.embeddings = BertEmbeddings(bert_config)
        self.encoder = BertEncoder(
            bert_config,
            self.attn_backend,
            self.unit_size,
            self.max_seqlen
        )
        self.cls = BertOnlyMLMHead(bert_config)
        if config.model.cross_attn:
            self.gen_mask(config.model.length, self.unit_size, self.attn_backend)

    def gen_mask(self, seqlen, unit_size, attn_backend='sdpa'):
        if attn_backend == 'flex' and FLEX_ATTN_AVAILABLE:
            self.unit_diff_mask_fn = partial(unit_diff_mask, unit_size=unit_size, n=seqlen)
            
        elif attn_backend == 'sdpa':
            self.unit_diff_mask_base = unit_diff_mask(
                b=None, h=None, q_idx=torch.arange(seqlen * 2)[:, None],
                kv_idx=torch.arange(seqlen * 2)[None, :], unit_size=unit_size, n=seqlen)
        else:
            raise ValueError('Unknown attention backend')

    def forward(
            self,
            indices: torch.LongTensor,
            attention_mask: Optional[torch.Tensor] = None,  
            sample_mode: bool = False,
    ):

        batch_size = indices.size(0)
        seq_len = indices.size(1)
        
        cross_attn = hasattr(self, 'unit_diff_mask_base') or hasattr(self, 'unit_diff_mask_fn')

        if cross_attn:
            if sample_mode:
                unit_mask = None
                hidden_states = self.embeddings(indices)
                combined_mask = attention_mask
            else:
                if hasattr(self, 'unit_diff_mask_base'):
                    unit_mask = self.unit_diff_mask_base
                else:
                    # Flex backend
                    unit_mask = create_block_mask(
                        self.unit_diff_mask_fn,
                        B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len
                    )
                hidden_states = self.embeddings(indices)

            if attention_mask is not None:
                if self.attn_backend == 'flex' and unit_mask is not None:
                    if not sample_mode:
                        mask_fn = create_unit_diff_mask_with_padding(
                            self.unit_size, self.n, attention_mask
                        )
                        combined_mask = create_block_mask(
                            mask_fn, B=batch_size, H=None, Q_LEN=seq_len, KV_LEN=seq_len
                        )
                    else:
                        mask_fn = create_unit_diff_mask_with_padding(
                            self.unit_size, self.n, attention_mask
                        )
                        combined_mask = create_block_mask(
                            mask_fn, B=batch_size, H=None, Q_LEN=seq_len, KV_LEN=seq_len
                        )
                elif self.attn_backend == 'sdpa':
                    extended_attention_mask = attention_mask
                    if unit_mask is not None:
                        unit_mask_additive = torch.where(
                            unit_mask,
                            torch.tensor(0.0, dtype=hidden_states.dtype, device=unit_mask.device),
                            torch.tensor(torch.finfo(hidden_states.dtype).min, dtype=hidden_states.dtype,
                                         device=unit_mask.device)
                        )
                        combined_mask = unit_mask_additive.unsqueeze(0) + extended_attention_mask
                    else:
                        combined_mask = extended_attention_mask
                else:
                    
                    combined_mask = unit_mask
            else:
                combined_mask = unit_mask

        else:
            hidden_states = self.embeddings(indices)
            combined_mask = attention_mask

        # Encoder
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            if sample_mode:
                sequence_output = self.encoder(
                    hidden_states,
                    attention_mask=attention_mask,
                    sample_mode=sample_mode,
                )
            else:
                sequence_output = self.encoder(
                    hidden_states,
                    attention_mask=combined_mask,
                    sample_mode=sample_mode,
                )

            logits = self.cls(sequence_output)

        if cross_attn and not sample_mode:
            logits = logits[:, :self.n]

        return logits
