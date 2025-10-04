import os
import torch
import math
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

        self.train_dataloader, self.test_dataloader, self.tokenizer = get_dataloaders(dataset_config, loader_config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_params.tokenizer = self.tokenizer

        self.model = get_model(model_params).to(device=self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), training_config.learning_rate, eps=1e-9, weight_decay=1e-2)

        self.total_steps = self.training_config.num_epochs * len(self.train_dataloader)

        self.warmup_steps = 5000

        # self.scheduler = LambdaLR(self.optimizer, self._lr_lambda)

        self.loss_func = nn.CrossEntropyLoss(ignore_index=self.tokenizer.PAD.item(), label_smoothing=training_config.label_smoothing).to(self.device)


    def __str__(self):
        return f"Training Stage f{Trainer}"
    
    def _lr_lambda(self, current_step: int):
        if current_step < self.warmup_steps:
            # linear warm-up
            return float(current_step) / float(max(1, self.warmup_steps))
        # cosine decay after warm-up
        progress = float(current_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        return 0.5 * (1.0 + math.cos(progress * math.pi)) * (1 - 1e-5 / self.training_config.learning_rate) + (1e-5 / self.training_config.learning_rate)

    def validate(self):
        """
        Validation with optional KV cache for faster inference.
        
        Args:
            use_kv_cache: Whether to use rolling KV cache during validation
        """
        self.model.eval()
        expected = []
        predicted = []

        with torch.no_grad():
            for batch in tqdm(self.test_dataloader):
                decoder_input = batch["decoder_input"].to(self.device)
                decoder_mask = None
                point_cloud = batch["point_cloud"].to(self.device)
                quad_ratio = batch["quad_ratio"].to(self.device)
                face_count = batch["face_count"].to(self.device)
                target = batch["target"].to(self.device)

                out = self.model(decoder_input, point_cloud, face_count, quad_ratio, decoder_mask)
                probs = self.model.project(out)

                tokens = torch.argmax(probs, dim=-1)
                expected.append(target)
                predicted.append(tokens)

        # Calculate accuracy - need to handle different sequence lengths
        acc=[]
        total_preds = len(predicted)
        for i in range(total_preds):
            correct = sum(predicted[i]==expected[i])
            acc.append(correct/ len(predicted[i]))
        test_acc = sum(acc) / total_preds #avg accuracy

        return test_acc


    def run(self):

        initial_epoch = 0
        global_step = 0
        logger = logger_init()
        expected = []
        predicted = []
        model_filename = get_latest_weights_path(self.training_config) if self.training_config.preload == "latest" else get_weights_path(self.training_config, epoch=self.training_config.preload)
        loss= None
        scaler = torch.amp.GradScaler(device=self.device)

        if model_filename:
            logger.info(f"Preloading model: {model_filename}")
            state = torch.load(model_filename)
            self.model.load_state_dict(state["model_state_dict"])
            initial_epoch  = state["epoch"] + 1
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
            scaler.load_state_dict(state['scaler_state_dict'])
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
                decoder_mask = None
                point_cloud = batch["point_cloud"].to(self.device)
                quad_ratio = batch["quad_ratio"].to(self.device)
                face_count = batch["face_count"].to(self.device)
                target = batch["target"].to(self.device)

                self.optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type= "cuda"):
                    output = self.model(decoder_input, point_cloud, face_count, quad_ratio, decoder_mask)
                    proj_out = self.model.project(output)
                    pred = torch.argmax(proj_out, dim=-1)
                    predicted.append(pred)
                    expected.append(target)

                    loss = self.loss_func(proj_out.view(-1, self.tokenizer.vocab_size), target.view(-1))
                    batch_iter.set_postfix({"loss": f"{loss.item():6.3f}"})

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                # self.scheduler.step()
                global_step += 1
                del decoder_input, decoder_mask, point_cloud, quad_ratio, face_count, target, output, proj_out, pred

            #stats
            acc=[]
            total_preds = len(predicted)
            for i in range(total_preds):
                correct = sum(predicted[i]==expected[i])
                acc.append(correct/ len(predicted[i]))
            total_avg = sum(acc) #avg accuracy
            train_acc = total_avg / total_preds 
            

            

            if epoch != 00 and self.training_config.val_after_every % epoch == 0:
                test_acc = self.validate()

                logger.info(f"Training iteration epoch: {epoch:02d}, Training accuracy: {train_acc}, Validation accuracy: {test_acc}, loss: {loss}")
                
            logger.info(f"Training iteration epoch: {epoch:02d}, Training accuracy: {train_acc}, loss: {loss}")
            
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
