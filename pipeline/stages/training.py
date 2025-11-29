import os
import torch
import math
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pipeline.utils.common import logger_init, get_root_folder
from pipeline.utils.model import get_model, get_latest_weights_path, get_weights_path
from pipeline.primitive_dataset import get_dataloaders
from pipeline.config_entities import TrainingConfig, ModelParams, DatasetConfig, DataLoaderConfig


class Trainer(nn.Module):
    def __init__(self, 
                 dataset_len:int,
                 training_config: TrainingConfig,  
                 model_params: ModelParams, 
                 dataset_config: DatasetConfig, 
                 loader_config: DataLoaderConfig
                 ):
        super().__init__()
        self.training_config = training_config

        self.model_params = model_params

        self.train_dataloader, self.test_dataloader, self.tokenizer = get_dataloaders(dataset_config, loader_config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_params.pad_token = self.tokenizer.PAD.item()

        self.model = get_model(model_params).to(device=self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), training_config.learning_rate, eps=1e-9, weight_decay=1e-2)

        self.total_steps = self.training_config.num_epochs * len(self.train_dataloader)

        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=1000,             
            T_mult=2,              
            eta_min=1e-6 
        )

        self.loss_func = nn.CrossEntropyLoss(ignore_index=self.tokenizer.PAD.item(), label_smoothing=training_config.label_smoothing).to(self.device)

        self.scaler = torch.amp.GradScaler()

    def __str__(self):
        return f"Training Stage f{Trainer}"

    def validate(self):

        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(self.test_dataloader):
                decoder_input = batch["decoder_input"].to(self.device)
                decoder_mask = None
                point_cloud = batch["point_cloud"].to(self.device)
                quad_ratio = batch["quad_ratio"].to(self.device)
                face_count = batch["face_count"].to(self.device)
                target = batch["target"].to(self.device)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    out = self.model(decoder_input, point_cloud, face_count, quad_ratio, decoder_mask)
                    probs = self.model.project(out)

                    loss = self.loss_func(probs.view(-1, self.tokenizer.vocab_size), target.view(-1))

                total_loss += loss.item() * target.size(0)
                total_samples += target.size(0)

        avg_loss = total_loss / total_samples
        return avg_loss

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
            self.scheduler.load_state_dict(state['scheduler_state_dict'])
            self.scaler.load_state_dict(state['scaler_state_dict'])
            global_step = state['global_step']
            del state
        else:
            logger.warning("No model to load, starting from sratch")

        for epoch in range(initial_epoch, self.training_config.num_epochs):
            self.model.train()
            batch_iter = tqdm(self.train_dataloader, desc=f"Processing epoch: {epoch:02d}")

            for batch in batch_iter:
                decoder_input = batch["decoder_input"].to(self.device, non_blocking = True)
                decoder_mask = None
                point_cloud = batch["point_cloud"].to(self.device, non_blocking = True)
                quad_ratio = batch["quad_ratio"].to(self.device, non_blocking = True)
                face_count = batch["face_count"].to(self.device, non_blocking = True)
                target = batch["target"].to(self.device, non_blocking = True)

                #forward
                with torch.amp.autocast('cuda'):
                    output = self.model(decoder_input, point_cloud, face_count, quad_ratio, decoder_mask)
                    proj_out = self.model.project(output)
                

                    loss = self.loss_func(proj_out.view(-1, self.tokenizer.vocab_size), target.view(-1))
                    
                    with open(os.path.join(get_root_folder(),'pipeline','logs','loss.txt'), 'a') as f:
                        f.write(f"{loss.item():0.6f}\n")

                batch_iter.set_postfix({"loss": f"{loss.item():6.3f}"})
                logger.info(f"Epoch: {epoch}, Iteration: {global_step:02d}, loss: {loss}")

                #backward
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                global_step += 1


                if global_step % self.training_config.val_after_every == 0:
                    testing_loss = self.validate()

                    logger.info(f"Training iteration: {global_step:02d}, training_loss: {loss}, testing_loss: {testing_loss}")
                    
            logger.info(f"Loss After epoch {epoch+1:02d}: {loss}")
            
            model_file_path = get_weights_path(self.training_config, f"{epoch:02d}")
            old_model_file_path = get_weights_path(self.training_config, f"{epoch - 1:02d}")

            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict(),
                    'global_step': global_step
                },
                model_file_path
            )

            #saving some memory
            if os.path.exists(old_model_file_path):
                os.remove(old_model_file_path)
