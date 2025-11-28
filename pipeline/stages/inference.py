import torch
import torch.nn as nn
import torch.nn.functional as F
from meshtron.VertexTokenizer import VertexTokenizer
from pipeline.config_entities import ModelParams
from pipeline.utils.model import get_model

def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs
    
class Inference(nn.Module):
    def __init__(self, model_params: ModelParams, weights_path: str):
        super().__init__()
        self.tokenizer = VertexTokenizer(128)
        model_params.pad_token = self.tokenizer.PAD.item()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = get_model(model_params).to(self.device)

        try:
            state = torch.load(weights_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Weight file not found: {weights_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {e}")
        
        self.model.load_state_dict(state["model_state_dict"])


    def run(self, point_cloud: torch.Tensor, face_count: torch.Tensor, quad_ratio: torch.Tensor):
        """ Follows greedy decode approach
            point_cloud shape : [1, N, 6]
            face_count shape: [1]
            quad_ratio shape: [1]
        """
        self.model.eval()
        point_cloud = point_cloud.to(device=self.device)
        face_count = face_count.to(device=self.device)
        quad_ratio = quad_ratio.to(device=self.device)
        decoder_input = torch.empty(1,9).fill_(self.tokenizer.SOS.item()).to(dtype=torch.int64, device=self.device)

        while True:
            if decoder_input.size(1) == 10377:
                break

            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    out = self.model(decoder_input, point_cloud, face_count, quad_ratio, None)

            logits = self.model.project(out[:, decoder_input.shape[-1] - 9])

            filtered_logits = top_k(logits, thres=0.9)
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            print(next_token)
            if next_token == self.tokenizer.EOS.item():
                break

            next_token = torch.tensor([[next_token]], device=self.device, dtype=torch.int64)
            decoder_input = torch.cat([decoder_input, next_token], dim=1)

        coords = self.tokenizer.decode(decoder_input[:, 9:])
        
        return coords
