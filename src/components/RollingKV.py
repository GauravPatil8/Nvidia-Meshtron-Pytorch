import torch

class RollingKVCache:
    def __init__(self, n_layers, n_heads, head_dim, max_len):
        """
        Args:
            num_layers: number of transformer layers
            num_heads: number of attention heads
            head_dim: dimension per head
            max_len: rolling cache length (72k)
        """
        self.num_layers = n_layers
        self.num_heads = n_heads
        self.head_dim = head_dim
        self.max_len = max_len
        
        # initialize cache
        self.clear()

    def clear(self):
        self.cache = {
            "k": [torch.zeros(0, self.num_heads, self.head_dim, device=self.device) for _ in range(self.num_layers)],
            "v": [torch.zeros(0, self.num_heads, self.head_dim, device=self.device) for _ in range(self.num_layers)]
        }

    def update(self, layer_idx, k_new, v_new):
        """
        Update cache with new keys and values.

        Args:
            layer_idx: index of transformer layer
            k_new: [seq_len, num_heads, head_dim]
            v_new: [seq_len, num_heads, head_dim]
        """
        k_cache = torch.cat([self.cache["k"][layer_idx], k_new], dim=0)
        v_cache = torch.cat([self.cache["v"][layer_idx], v_new], dim=0)

        if k_cache.size(0) > self.max_len:
            k_cache = k_cache[-self.max_len:]
            v_cache = v_cache[-self.max_len:]

        self.cache["k"][layer_idx] = k_cache
        self.cache["v"][layer_idx] = v_cache

    def get(self, layer_idx):
        """Return cached keys and values for a layer"""
        return self.cache["k"][layer_idx], self.cache["v"][layer_idx]