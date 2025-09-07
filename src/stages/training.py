import torch
import logging
import torch.nn as nn
from tqdm import tqdm
from src.utils.common import logger_init
from src.components.model import build_model
from src.components.PrimitiveDataset import get_dataloaders
from src.config_entities import TrainingConfig, ModelParams, DatasetConfig, DataLoaderConfig
from src.config import get_latest_weights_path, get_weights_path

class Trainer(nn.Module):
    def __init__(self, 
                 training_config: TrainingConfig,  
                 model_params: ModelParams, 
                 dataset_config: DatasetConfig, 
                 loader_config: DataLoaderConfig
                 ):

        self.training_config = training_config

        self.train_dataloader, self.test_dataloader, self.tokentizer = get_dataloaders(dataset_config, loader_config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = build_model(model_params).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), training_config.learning_rate, eps=1e-9)

        self.loss_func = nn.CrossEntropyLoss(ignore_index=self.tokentizer.PAD, label_smoothing=training_config.label_smoothing).to(self.device)

    def forward(self):

        initial_epoch = 0
        global_step = 0
        logger = logging.getLogger()

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

                loss = self.loss_func(output.view(-1, self.tokentizer.vocab_size), target.view(-1))
                batch_iter.set_postfix({"loss": f"{loss.item():6.3f}"})

                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                global_step += 1

                #need to run validation here

                model_filename = get_weights_path(self.training_config, f"{epoch:02d}")
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'global_step': global_step
                    },
                    model_filename
                )