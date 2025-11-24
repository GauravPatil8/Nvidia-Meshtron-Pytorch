import torch
import torch.nn as nn
from meshtron.VertexTokenizer import VertexTokenizer
from pipeline.config_entities import ModelParams
from pipeline.utils.model import get_model

class Inference:
    def __init__(self, model_params: ModelParams, weights_path: str):

        self.tokenizer = VertexTokenizer(131, 1.0)
        model_params.pad_token = self.tokenizer.PAD
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = get_model(model_params).to(self.device)

        try:
            state = torch.load(weights_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Weight file not found: {weights_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {e}")
        
        self.model.load_state_dict(state["model_state_dict"])


    def forward(self, point_cloud: torch.Tensor, face_count: torch.Tensor, quad_ratio: torch.Tensor):
        """ Follows greedy decode approach
            point_cloud shape : N, 6
            face_count shape: [1]
            quad_ratio shape: [1]
        """
        
        decoder_input = torch.empty(1,9).fill_(self.tokentizer.SOS.item()).to(dtype=torch.int64, device=self.device)

        while True:
            if decoder_input.size(1) == self.model_params.seq_len:
                break

            out = self.model(decoder_input, point_cloud, face_count, quad_ratio)

            o_proj = self.model.project(out[:, -1])
            _, next_token = torch.argmax(o_proj, dim = 1)


            if next_token == self.tokentizer.EOS.item():
                break

            decoder_input = torch.cat(
                [decoder_input, torch.empty(1,1).fill_(next_token.item()).to(device=self.device, dtype=torch.int64)],
                dim=-1,
            )

        coords = self.tokenizer.decode(decoder_input)

        return coords
            