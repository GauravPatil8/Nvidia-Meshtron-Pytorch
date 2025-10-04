# def greedy_decode(self, point_cloud, face_count, quad_ratio):
#         """
#         Greedy decoding with rolling KV cache support.
        
#         Args:
#             point_cloud: Input point cloud data
#             face_count: Face count information  
#             quad_ratio: Quad ratio information
#             kv_cache: Optional RollingKVCache instance for inference
#         """
#         decoder_input = torch.empty(1,9).fill_(self.tokentizer.SOS.item()).to(dtype=torch.int64, device=self.device)

#         while True:
#             if decoder_input.size(1) == self.model_params.seq_len:
#                 break

#             # For KV cache, we only need to process the last token
#             if self.model_params.use_kv_cache:
#                 # Only process the new token (last token in decoder_input)
#                 current_token = decoder_input[:, -1:]
#                 decoder_mask = causal_mask(1).to(dtype=torch.long, device=self.device)
                
#                 out = self.model(current_token, point_cloud, face_count, quad_ratio, 
#                             decoder_mask)
#             else:
#                 # Without cache, process entire sequence
#                 decoder_mask = causal_mask(decoder_input.size(1)).to(dtype=torch.int32, device=self.device)
#                 out = self.model(decoder_input, point_cloud, face_count, quad_ratio, decoder_mask)

#             prob = self.model.project(out[:, -1])
#             _, next_token = torch.argmax(prob, dim=1)

#             decoder_input = torch.cat(
#                 [decoder_input, torch.empty(1,1).fill_(next_token.item()).to(device=self.device, dtype=torch.int64)],
#                 dim=-1,
#             )
#             if next_token == self.tokentizer.EOS.item():
#                 break

#         return decoder_input.squeeze(0)
