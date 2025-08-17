"""
CLM (Causal Language Modeling) Models
====================================

BERT-based models adapted for causal language modeling with different attention mechanisms.
"""

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from typing import Optional, Dict, Any

from ..attention import get_attention_class


class CLMBertModel(nn.Module):
    """
    BERT model adapted for Causal Language Modeling
    """
    
    def __init__(self, config: BertConfig, attention_type: str = "standard"):
        super().__init__()
        self.config = config
        self.attention_type = attention_type
        
        # Use base BERT model (no MLM head)
        self.bert = BertModel(config)
        
        # Add causal LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Replace attention layers with specified type
        self._replace_attention_layers(attention_type)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _replace_attention_layers(self, attention_type: str):
        """Replace all attention layers with specified type"""
        attention_class = get_attention_class(attention_type)
        
        for layer_idx in range(self.config.num_hidden_layers):
            layer = self.bert.encoder.layer[layer_idx]
            
            # Create new attention layer
            new_attention = attention_class(
                hidden_size=self.config.hidden_size,
                num_heads=self.config.num_attention_heads,
                max_position_embeddings=self.config.max_position_embeddings,
                dropout=self.config.attention_probs_dropout_prob
            )
            
            # Replace the self-attention
            layer.attention.self = new_attention
    
    def _init_weights(self, module):
        """Initialize weights following GPT-2 style initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal attention mask for autoregressive generation
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        mask = mask.view(1, 1, seq_len, seq_len)
        return mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for causal language modeling
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create causal attention mask
        causal_mask = self.get_causal_mask(seq_len, device)
        
        # Combine with padding mask if provided
        if attention_mask is not None:
            # Expand attention mask to match causal mask shape
            extended_attention_mask = attention_mask.view(batch_size, 1, 1, seq_len)
            extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            
            # Combine causal and padding masks
            causal_mask = causal_mask + extended_attention_mask
        
        # BERT forward pass with causal mask
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=causal_mask,
            return_dict=True,
            **kwargs
        )
        
        hidden_states = bert_outputs.last_hidden_state
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for loss computation
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        if return_dict:
            return {
                'loss': loss,
                'logits': logits,
                'hidden_states': hidden_states,
                'attentions': bert_outputs.attentions if hasattr(bert_outputs, 'attentions') else None
            }
        else:
            outputs = (logits,)
            if loss is not None:
                outputs = (loss,) + outputs
            return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True
    ) -> torch.Tensor:
        """
        Generate text autoregressively
        """
        self.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                outputs = self(generated)
                next_token_logits = outputs['logits'][:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                if do_sample:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if EOS token is generated
                if next_token.item() == self.config.eos_token_id:
                    break
        
        return generated


class StandardCLMBert(CLMBertModel):
    """BERT with standard attention for CLM"""
    def __init__(self, config: BertConfig):
        super().__init__(config, attention_type="standard")


class RoPECLMBert(CLMBertModel):
    """BERT with RoPE attention for CLM"""
    def __init__(self, config: BertConfig):
        super().__init__(config, attention_type="rope")


class ExpoSBCLMBert(CLMBertModel):
    """BERT with ExpoSB attention for CLM"""
    def __init__(self, config: BertConfig):
        super().__init__(config, attention_type="exposb")


class AbsoluteCLMBert(CLMBertModel):
    """BERT with absolute positional encoding for CLM"""
    def __init__(self, config: BertConfig):
        super().__init__(config, attention_type="absolute")


# Factory function
def create_clm_model(config: BertConfig, attention_type: str = "standard") -> CLMBertModel:
    """
    Create a CLM model with specified attention type
    
    Args:
        config: BERT configuration
        attention_type: Type of attention ("standard", "rope", "exposb", "absolute")
    
    Returns:
        CLM model with specified attention mechanism
    """
    clm_classes = {
        "standard": StandardCLMBert,
        "rope": RoPECLMBert,
        "exposb": ExpoSBCLMBert,
        "absolute": AbsoluteCLMBert
    }
    
    if attention_type not in clm_classes:
        raise ValueError(f"Unknown attention type: {attention_type}. "
                        f"Available: {list(clm_classes.keys())}")
    
    model_class = clm_classes[attention_type]
    return model_class(config)


# Testing function
if __name__ == "__main__":
    from transformers import BertTokenizer
    
    # Test CLM models
    config = BertConfig(
        vocab_size=30522,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
        max_position_embeddings=512
    )
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Test each attention type
    for attention_type in ["standard", "rope", "exposb", "absolute"]:
        print(f"\n=== Testing {attention_type.upper()} CLM Model ===")
        
        model = create_clm_model(config, attention_type)
        model.eval()
        
        # Test input
        text = "The quick brown fox"
        inputs = tokenizer(text, return_tensors="pt")
        
        # Test forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            print(f"Input shape: {inputs['input_ids'].shape}")
            print(f"Output logits shape: {outputs['logits'].shape}")
            
            # Test generation
            generated = model.generate(
                inputs['input_ids'], 
                max_length=inputs['input_ids'].shape[1] + 10,
                do_sample=False
            )
            generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            print(f"Generated: {generated_text}")
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")