
import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig, TrainingArguments
from transformers.modeling_outputs import CausalLMOutput


class A2ModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the Transformer language model."""
    def __init__(self, vocab_size=None, hidden_size=None, intermediate_size=None, num_attention_heads=None, 
                 num_hidden_layers=None,
                 rope_theta=None, hidden_act='silu', max_position_embeddings=None, rms_norm_eps=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.num_attention_heads = num_attention_heads
        self.rope_theta = rope_theta
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers


class A2MLP(nn.Module):
    """The MLP layer of the Transformer. Uses the SwiGLU architecture."""
    def __init__(self, config):
        super().__init__()
        assert(config.hidden_act == 'silu')
        # TODO: initalize components here
        self.linear_right = nn.Linear(in_features=config.hidden_size, out_features=config.intermediate_size)
        self.linear_left = nn.Linear(in_features=config.hidden_size, out_features=config.intermediate_size)
        self.linear_top = nn.Linear(config.intermediate_size, config.hidden_size)
        self.silu = nn.SiLU()

    def forward(self, hidden_states):
        hidden_right = self.linear_right(hidden_states)
        hidden_left = self.linear_left(hidden_states)
        hidden = hidden_right * self.silu(hidden_left)
        output = self.linear_top(hidden)
        return output


# Sanity check:
config = A2ModelConfig(
    vocab_size=10000,
    hidden_size=6,
    intermediate_size=11,
    num_attention_heads=1,
    num_hidden_layers=1,
    rope_theta=100000.0,
    hidden_act='silu',
    max_position_embeddings=512,
    rms_norm_eps=1e-6,
)
layer = A2MLP(config)
random_tensor = torch.tensor( # the first dimension has size 4, the second has size 3, the last has size 6
    [
        [
            [1,1,1,1,1,1], [1,1,1,1,1,1], [1,1,1,1,1,1]
        ],
        [
            [1,1,1,1,1,1], [1,1,1,1,1,1], [1,1,1,1,1,1]
        ],
        [
            [1,1,1,1,1,1], [1,1,1,1,1,1], [1,1,1,1,1,1]
        ],
        [
            [1,1,1,1,1,1], [1,1,1,1,1,1], [1,1,1,1,1,1]
        ],
    ], dtype=torch.float32)
# print(random_tensor.shape)
output = layer(random_tensor)
# print(output.shape) # The output should have the same shape as the input, i.e. (4, 3, 6).


# This is optional, since you can use PyTorch's RMSNorm.
class A2RMSNorm(nn.Module):
    """RMS layer normalization."""
    def __init__(self, config):
        super().__init__()
        # TODO: Use config.rms_norm_eps
        self.eps = config.rms_norm_eps
        # TODO: initalize weights here
        self.weight = nn.Parameter(torch.ones(config.hidden_size)) # The weight is initialized to all ones, so it doesn't change the output at the beginning of training.
        # NOTE: I didn't know this was the way to initialize learnable parameters


    def forward(self, hidden_states):
        """
        1. Compute the variance of the hidden states along the last dimension.
        2. Normalize the hidden states by dividing by the square root of the variance plus epsilon.
        3. Scale the normalized hidden states by the learnable weight parameter.
        """
        # Implementation by CoPilot:
        mean_a_squared = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(mean_a_squared + self.eps) # rqst(x) = 1/sqrt(x) but more numerically stable
        return hidden_states * self.weight 
    

# Sanity check --------------
config = A2ModelConfig(
    vocab_size=10000,
    hidden_size=6,
    intermediate_size=11,
    num_attention_heads=1,
    num_hidden_layers=1,
    rope_theta=100000.0,
    hidden_act='silu',
    max_position_embeddings=512,
    rms_norm_eps=1e-6,
)
layer = A2RMSNorm(config)
# use the same random tensor as before
output = layer(random_tensor)
# print(output.shape) # The output should have the same shape as the input, i.e. (4, 3, 6).
# ---------------------------

def scaled_dot_product_attention(q, k, v):
    # The mask is a boolean tensor used to tell the model which position to pay attention to.
    # Here, we will use it to implement causal masking, which prevents the model from attending
    # to future position in the sequence.
    seq_length = q.shape[-2]
    attn_mask = torch.ones(seq_length, seq_length, device=q.device).tril().bool()
    hidden_dim = q.shape[-1]
    scale = hidden_dim ** 0.5
    attn_weights = q @ k.transpose(-2, -1) / scale
    attn_weights = torch.softmax(attn_weights + attn_mask, dim=-1)
    output = attn_weights @ v
    return output


class A2Attention(nn.Module):
    """The multi-head attention layer of the Transformer. Uses standard scaled dot-product attention with causal masking."""
    
    def __init__(self, config):
        super().__init__()
        # TODO: set up W_q, W_k, W_v, W_o here
        self.W_q = nn.Linear(config.hidden_size, config.hidden_size) # these are square matrices since they compute y = W * x, y and x same shape (hidden_size,)
        self.W_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.W_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.W_o = nn.Linear(config.hidden_size, config.hidden_size)
        # TODO: set up normalizers here
        self.normaliser = A2RMSNorm(config)

    def forward(self, hidden_states, rope_rotations):
        # Pass hidden states through linear layers to get queries, keys, and values.
        q = self.W_q(hidden_states)
        k = self.W_k(hidden_states)
        v = self.W_v(hidden_states)

        # Divide the hidden states into multiple heads, so normalisation and attention are computed in each head separately.
        batch_size = hidden_states.shape[0]
        seq_length = hidden_states.shape[1]
        dim_per_head = hidden_states.shape[2] // config.num_attention_heads # where hidden_states.shape[2] is the hidden size
        
        # Long explanation of the reshape and transpose operations below:
        """
        The hidden states have shape (batch_size, seq_length, hidden_size). We want to reshape them to (batch_size, num_attention_heads, seq_length, dim_per_head) so that we can compute attention for each head separately.        
        Now, reshape q, k, v to have shape(batch_size, num_attention_heads, seq_length, dim_per_head)
        Their original shape is (batch_size, seq_length, hidden_size), where hidden_size = num_attention_heads * dim_per_head.
        So we create another dimension for the attention heads.

        Simple example with batch_size=1, seq_length=2, hidden_size=4, num_attention_heads=2, dim_per_head=2:

        # Shape before reshape: (1, 2, 4)
        [
          [
              [1, 2, 3, 4], 
              [5, 6, 7, 8]
          ]
        ] 

        # Reshape this to (1, 2, 2, 2)
        [
          [
              [[1, 2], [3, 4]],
              [[5, 6], [7, 8]]
          ]
        ]
        # Then transpose to (batch_size=1, seq_length=2,          num_attention_heads=2, dim_per_head=2) -> 
        #                -> (batch_size=1, num_attention_heads=2, seq_length=2,          dim_per_head=2)
        [
          [
              [[1, 2], [5, 6]], # head 1 gets this
              [[3, 4], [7, 8]]  # head 2 gets this
          ]
        ]

        """
        
        # Normalize queries and keys before splitting into heads 
        q = self.normaliser(q)
        k = self.normaliser(k)

        q = q.view(batch_size, seq_length, config.num_attention_heads, dim_per_head).transpose(1, 2) # first reshape to (batch_size, seq_length, num_attention_heads, dim_per_head) then transpose to (batch_size, num_attention_heads, seq_length, dim_per_head)
        k = k.view(batch_size, seq_length, config.num_attention_heads, dim_per_head).transpose(1, 2)
        v = v.view(batch_size, seq_length, config.num_attention_heads, dim_per_head).transpose(1, 2)

        if rope_rotations is not None:
            q, k = apply_rotary_pos_emb(q, k, rope_rotations)

        # Now, apply the Scaled Dot-Product Attention function:

        scaled_dot_product_attention_output = scaled_dot_product_attention(q, k, v) 
        # (batch_size, num_attention_heads, seq_length, dim_per_head)

        # Now we need to reshape this back to (batch_size, seq_length, hidden_size) to pass through the final linear layer W_o.
        
        scaled_dot_product_attention_output = scaled_dot_product_attention_output.transpose(1, 2).contiguous() 
        # (batch_size, seq_length, num_attention_heads, dim_per_head)

        scaled_dot_product_attention_output = scaled_dot_product_attention_output.view(batch_size, seq_length, config.hidden_size)
        # (batch_size, seq_length, hidden_size)

        # For this attention computation, we could have simply used PyTorch built-in sdpa function:
        # attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

        scaled_dot_product_attention_output = self.W_o(scaled_dot_product_attention_output)

        return scaled_dot_product_attention_output


class A2DecoderLayer(nn.Module):
    """A complete Transformer decoder layer."""
    def __init__(self, config):
        super().__init__()
        # TODO: set up attention, MLP, and normalizers here.
        self.normaliser = A2RMSNorm(config)
        self.attention = A2Attention(config)
        self.mlp = A2MLP(config)

    def forward(self, hidden_states, rope_rotations):
        h_new = self.normaliser(self.attention(hidden_states, rope_rotations)) + hidden_states # sum with hidden_states for residual
        h_final = self.normaliser(self.mlp(h_new)) + h_new # sum with h_new for residual
        return h_final


class A2Transformer(PreTrainedModel):
    """A language model based on the Transformer architecture."""
    
    config_class = A2ModelConfig

    def __init__(self, config):
        super().__init__(config)

        self.rotary_emb = A2RotaryEmbedding(config)
        # TODO: Set up the other components here.
        self.embedding = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size)
        self.normaliser = A2RMSNorm(config)
        self.unembedding = nn.Linear(config.hidden_size, config.vocab_size, bias=False) # bias=False since we don't want bias terms in the unembedding layer
        
        # TODO: put all transformer decoder layers in a ModuleList instead of a plain Python list. For some PyTorch reason. 
        # The ModuleList makes sure your parameters are registered so that they are included  when you compute the gradients.
        self.decoder_layers = nn.ModuleList([A2DecoderLayer(config) for _ in range(config.num_hidden_layers)])

        # Copilot: This method is defined in the PreTrainedModel class and is used to initialize the weights of the model. 
        # It should be called after you have set up all components of the model, so that it can initialize the weights of those components.
        self.post_init() 

    def forward(self, input_ids, labels=None):
        rope_rotations = self.rotary_emb(input_ids) # pass this to all the transformer decoder layers

        loss = None

        # TODO: Call embedding, transformer decoder layers, last normalizer, and unembedding.
        hidden = self.embedding(input_ids) # (batch_size, seq_length, hidden_size)
        for decoder_layer in self.decoder_layers:
            hidden = decoder_layer(hidden, rope_rotations)

        hidden = self.normaliser(hidden)
        output_logits = self.unembedding(hidden) # (batch_size, seq_length, voc_size) i.e. every token gives predictions to all words in vocab

        # TODO: Compute the loss as in Assignment 1 if labels is not None.
        if labels is not None:
            shift_logits = output_logits[:, :-1, :].contiguous() # we don't care about the prediction after the last token
            shift_labels = labels[:, 1:].contiguous() # we don't care about the loss for the first token

            # Reshape logits and labels to compute the loss
            shift_labels = shift_labels.view(-1) # (batch_size * (seq_length - 1),)
            shift_logits = shift_logits.view(-1, self.config.vocab_size) # (batch_size * (seq_length - 1), vocab_size)
            loss = self.loss_function(shift_logits, shift_labels, self.config.vocab_size)

        return CausalLMOutput(logits=output_logits, loss=loss)



#### RoPE implementation (copied and simplified from HuggingFace). ####

def apply_rotary_pos_emb(q, k, rope_rotations, unsqueeze_dim=1):
    """Applies precomputed RoPE rotations to the query and key representations."""
    assert(q.shape == k.shape)
    assert(len(q.shape) == 4)
    cos, sin = rope_rotations
    assert(q.shape[2] == cos.shape[1])
    assert(q.shape[3] == cos.shape[2])    
    q_type, k_type = q.dtype, k.dtype
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q_type), k_embed.to(k_type)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class A2RotaryEmbedding(nn.Module):
    """RoPE position representation for use in Transformer attention."""

    def __init__(self, config, device=None):
        super().__init__()
        rope_theta = config.rope_theta
        head_dim = config.hidden_size // config.num_attention_heads
        partial_rotary_factor = 1.0
        dim = int(head_dim * partial_rotary_factor)
        self.inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))

    @torch.no_grad()
    def forward(self, x):
        position_ids = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            return cos, sin

# Sanity check: take the already created tensor and apply the MHA to it and make sure it doesn't crash --------------
config = A2ModelConfig(
    vocab_size=10000,
    hidden_size=6,
    intermediate_size=11,
    num_attention_heads=3,
    num_hidden_layers=10,
    rope_theta=100000.0,
    hidden_act='silu',
    max_position_embeddings=512,
    rms_norm_eps=1e-6,
)
# layer = A2Attention(config)
# layer = A2DecoderLayer(config)
# layer = A2Transformer(config)
# random_tensor = torch.randint(0, config.vocab_size, (4, 3))  # (batch_size=4, seq_length=3) integer token IDs
# output, _ = layer(random_tensor)
# print(output.shape) # the output should be (4, 3, vocab_size), i.e. logits over vocab for each token.
# ---------------------------

# ====================================================================================
# =================================== Train the model ================================
# ====================================================================================
import sys
sys.path.insert(0, '../../ass_1_tokenisation_and_embeddings/a1_1')
from A1_skeleton import load_tokenizer, get_dataset, A1Trainer, A1Tokenizer


if __name__ == '__main__':
    # Train:
    tokenizer = load_tokenizer()
    tokenizer.model_max_length = 256
    config = A2ModelConfig(
        vocab_size=len(tokenizer),
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=8,
        num_hidden_layers=3,
        rope_theta=100000.0,
        hidden_act='silu',
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
    )
    model = A2Transformer(config)

    training_args = TrainingArguments(
        optim='adamw_torch',
        use_cpu=False,
        eval_strategy='epoch',
        output_dir='./results',
        num_train_epochs=3,
        per_device_eval_batch_size=64,
        per_device_train_batch_size=64,
        logging_dir='./logs',
        learning_rate=0.001,
    )

    dataset = get_dataset(use_subset=True)
    trainer = A1Trainer(model, 
                        training_args, 
                        dataset['train'], 
                        dataset['val'], 
                        tokenizer
                        )
    trainer.train()
    