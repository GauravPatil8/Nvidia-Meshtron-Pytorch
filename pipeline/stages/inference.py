import torch
import torch.nn as nn
from meshtron.VertexTokenizer import VertexTokenizer
from pipeline.config_entities import ModelParams
from pipeline.utils.model import get_model
from pipeline.utils.data import get_max_seq_len

class Inference:
    def __init__(self, model_params: ModelParams, weights_path: str):
        self.tokenizer = VertexTokenizer(128, 1.0)
        model_params.pad_token = self.tokenizer.PAD
        self.model_params = model_params
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
            point_cloud shape : N, 6
            face_count shape: [1]
            quad_ratio shape: [1]
        """
        point_cloud = point_cloud.to(self.device)
        face_count = face_count.to(self.device)
        quad_ratio = quad_ratio.to(self.device)
        decoder_input = torch.empty(1,9).fill_(self.tokenizer.SOS.item()).to(dtype=torch.int64, device=self.device)
        max_seq_len = 10377


        while True:
            if decoder_input.size(1) == max_seq_len:
                break
            with torch.amp.autocast(device_type='cuda'):
                out = self.model(decoder_input, point_cloud, face_count, quad_ratio, mask=None)

            o_proj = self.model.project(out[:, -1])
            next_token = torch.argmax(o_proj, dim = 1)


            if next_token == self.tokenizer.EOS.item():
                break

            decoder_input = torch.cat(
                [decoder_input, torch.empty(1,1).fill_(next_token.item()).to(device=self.device, dtype=torch.int64)],
                dim=-1,
            )

        coords = self.tokenizer.decode(decoder_input)

        return coords
            