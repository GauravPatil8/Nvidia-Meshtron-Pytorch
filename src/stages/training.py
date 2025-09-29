import os
import torch
import logging
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from src.utils.common import logger_init
from src.components.model import get_model
from src.components.PrimitiveDataset import get_dataloaders, causal_mask
from src.config_entities import TrainingConfig, ModelParams, DatasetConfig, DataLoaderConfig
from src.config import get_latest_weights_path, get_weights_path

class Trainer(nn.Module):
    def __init__(self, 
                 training_config: TrainingConfig,  
                 model_params: ModelParams, 
                 dataset_config: DatasetConfig, 
                 loader_config: DataLoaderConfig
                 ):
        super().__init__()
        self.training_config = training_config

        self.model_params = model_params

        self.train_dataloader, self.test_dataloader, self.tokentizer = get_dataloaders(dataset_config, loader_config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_params.tokenizer = self.tokentizer

        self.model = get_model(model_params).to(device=self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), training_config.learning_rate, eps=1e-9, weight_decay=1e-2)

        self.total_steps = self.training_config.num_epochs * len(self.train_dataloader)

        self.warmup_steps = 5000

        self.scheduler = LambdaLR(self.optimizer, self._lr_lambda)

        self.loss_func = nn.CrossEntropyLoss(ignore_index=self.tokentizer.PAD, label_smoothing=training_config.label_smoothing).to(self.device)

    def _lr_lambda(self, current_step: int):
        if current_step < self.warmup_steps:
            # linear warm-up
            return float(current_step) / float(max(1, self.warmup_steps))
        # cosine decay after warm-up
        progress = float(current_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))) * (1 - 1e-5 / self.training_config.learning_rate) + (1e-5 / self.training_config.learning_rate)
    
    def greedy_decode(self, point_cloud, face_count, quad_ratio):
        """
        Greedy decoding with rolling KV cache support.
        
        Args:
            point_cloud: Input point cloud data
            face_count: Face count information  
            quad_ratio: Quad ratio information
            kv_cache: Optional RollingKVCache instance for inference
        """
        decoder_input = torch.empty(1,9).fill_(self.tokentizer.SOS.item()).to(dtype=torch.int64, device=self.device)
        print("-"*100)
        print(decoder_input)
        print(decoder_input.shape)
        print("-"*100)

        while True:
            if decoder_input.size(1) == self.model_params.seq_len:
                break

            # For KV cache, we only need to process the last token
            if self.model_params.use_kv_cache:
                # Only process the new token (last token in decoder_input)
                current_token = decoder_input[:, -1:]
                decoder_mask = causal_mask(1).to(dtype=torch.long, device=self.device)
                
                out = self.model(current_token, point_cloud, face_count, quad_ratio, 
                            decoder_mask)
            else:
                # Without cache, process entire sequence
                print("-"*100)
                print("not using kv cache")
                print("-"*100)
                decoder_mask = causal_mask(decoder_input.size(1)).to(dtype=torch.int32, device=self.device)
                out = self.model(decoder_input, point_cloud, face_count, quad_ratio, decoder_mask)

            prob = self.model.project(out[:, -1])
            _, next_token = torch.argmax(prob, dim=1)

            decoder_input = torch.cat(
                [decoder_input, torch.empty(1,1).fill_(next_token.item()).to(device=self.device, dtype=torch.int64)],
                dim=-1,
            )
            print("-"*100)
            print(decoder_input)
            print("-"*100)
            if next_token == self.tokentizer.EOS.item():
                break

        return decoder_input.squeeze(0)


    def validate(self):
        """
        Validation with optional KV cache for faster inference.
        
        Args:
            use_kv_cache: Whether to use rolling KV cache during validation
        """
        self.model.eval()
        expected = []
        predicted = []

        total_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in tqdm(self.test_dataloader):
                point_cloud = batch["point_cloud"].to(self.device)
                quad_ratio = batch["quad_ratio"].to(self.device)
                face_count = batch["face_count"].to(self.device)
                target = batch["target"].to(self.device)

                out = self.greedy_decode(point_cloud, face_count, quad_ratio)

                expected.append(target)
                predicted.append(out)

                # Calculate accuracy - need to handle different sequence lengths
                min_len = min(out.size(0), target.size(0))
                total_correct += torch.sum(out[:min_len] == target[:min_len])
                total_tokens += min_len

        accuracy = total_correct / total_tokens
        return accuracy


    def run(self):

        initial_epoch = 0
        global_step = 0
        logger = logger_init()

        model_filename = get_latest_weights_path(self.training_config) if self.training_config.preload == "latest" else get_weights_path(self.training_config, epoch=self.training_config.preload)

        if model_filename:
            logger.info(f"Preloading model: {model_filename}")
            state = torch.load(model_filename)
            self.model.load_state_dict(state["model_state_dict"])
            initial_epoch  = state["epoch"] + 1
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            global_step = state['global_step']
            del state
        else:
            logger.warning("No model to load, starting from sratch")

        
        for epoch in range(initial_epoch, self.training_config.num_epochs):
            torch.cuda.empty_cache()
            self.model.train()
            batch_iter = tqdm(self.train_dataloader, desc=f"Processing epoch: {epoch:02d}")

            for batch in batch_iter:
                decoder_input = batch["decoder_input"].to(self.device)
                decoder_mask = batch["decoder_mask"].to(self.device)
                point_cloud = batch["point_cloud"].to(self.device)
                quad_ratio = batch["quad_ratio"].to(self.device)
                face_count = batch["face_count"].to(self.device)
                target = batch["target"].to(self.device)


                output = self.model(decoder_input, point_cloud, face_count, quad_ratio, decoder_mask)
                proj_out = self.model.project(output)

                loss = self.loss_func(proj_out.view(-1, self.tokentizer.vocab_size), target.view(-1))
                batch_iter.set_postfix({"loss": f"{loss.item():6.3f}"})

                loss.backward()

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                global_step += 1

                #stats
                pred = torch.argmax(output, dim=-1)
                correct = torch.sum(pred == target).to(dtype=torch.float32)
                total = torch.numel(target)
                train_acc = correct / total

                #run validations
                test_acc = self.validate()

                logger.info(f"Training iteration epoch: {epoch:02d}, Training accuracy: {train_acc}, Validation accuracy: {test_acc}, loss: {loss}")
                
                model_file_path = get_weights_path(self.training_config, f"{epoch:02d}")
                old_model_file_path = get_weights_path(self.training_config, f"{epoch - 1:02d}")

                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'global_step': global_step
                    },
                    model_file_path
                )

                #saving some memory
                if os.path.exists(old_model_file_path):
                    os.remove(old_model_file_path)
