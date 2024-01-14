import lightning as L
import lightning.pytorch as pl
from glob import glob
import re
import numpy as np

from torch.utils import data
import wfdb
from typing import List
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch
from torch import nn
from importlib import import_module
from collections import defaultdict
from sklearn.preprocessing import minmax_scale
import utils

# def select_random_windows(arr, window_size):
#     offsets = np.random.randint(0, arr.shape[1]-window_size+1, size=arr.shape[0])
#     return arr[np.arange(arr.shape[0])[:,None], offsets[:,None] + np.arange(window_size)]

def select_random_windows(arr, window_size):
    offset = np.random.randint(0, arr.shape[1]-window_size+1, 1)[0]
    return arr[:, offset: offset + window_size]

class MioDataset(data.Dataset):
    def __init__(self, patients_paths: List[str], type_: str, hparams:dict, transform=None):
        self.hparams = hparams
        self.type = type_
        self.patients_paths = patients_paths
        self.transform = transform
        self.data = []

        paths = []
        for patient in list(patients_paths)[:]:
            paths += [x.rsplit('.dat',1)[0] for x in glob(patient+'/*.dat')]
        
        label_to_id = {label: id for id, label in enumerate(hparams.data.labels)}
        for path in paths:
            data_record = wfdb.rdsamp(path)
            
            mask = [True if x in hparams.data.channels else False for x in data_record[1]['sig_name']]

            label = re.findall(r'Reason for admission:(.*)\n', '\n'.join(data_record[1]['comments']))[0].strip()
            if label not in hparams.data.labels:
                continue
            
            x = data_record[0][:, mask].transpose((1,0))
            self.data.append({
                'X' : minmax_scale(x, axis=1),
                'Y' : torch.LongTensor(torch.tensor(label_to_id[label])),
                'label' : label,
            })
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_record = self.data[idx].copy()
        data_record['X'] = torch.FloatTensor(select_random_windows(data_record['X'], self.hparams.data.window_size))

        if self.transform:
            data_record['X'] = self.transform(data_record['X'])

        return data_record

class Model(L.LightningModule):
    def __init__(self, hparams:dict):
        super().__init__()
        self.hparams.update(hparams)
        self.model = getattr(import_module("models"),hparams.model.name)(hparams)

        self.score_functions = []
        for func_data in hparams.logging.score_functions:
            module_path, class_name = func_data['path'].rsplit('.', 1)
            self.score_functions.append({
                "func": getattr(import_module(module_path),class_name)\
                    (num_classes=len(hparams.data.channels), **func_data.get("params", {})),
                "name": func_data.get("name", class_name),
            })

        self.loss_fn = nn.NLLLoss()
        self.logg_data = defaultdict(list)
        self.save_hyperparameters()

    def loggining(self, type):
        self.logg_data[f'{type}_outputs'] = torch.concat(self.logg_data[f'{type}_outputs']).cpu()
        self.logg_data[f'{type}_outputs'] = torch.nn.functional.softmax(self.logg_data[f'{type}_outputs'], dim=1)
        self.logg_data[f'{type}_y'] = torch.concat(self.logg_data[f'{type}_y']).cpu()

        for func_data in self.score_functions:
            value = func_data["func"](self.logg_data[f'{type}_outputs'], self.logg_data[f'{type}_y'])
            self.log(f"{func_data['name']}/{type}", value)
        
        if self.current_epoch % self.hparams.logging.save_cm_each_steps == 1:
            utils.save_confusion_matrix(
                self.logger.experiment,
                self.logg_data[f'{type}_y'],
                self.logg_data[f'{type}_outputs'].argmax(1),
                self.hparams.data.labels,
                self.current_epoch,
                title=f"Confusion matrix/{type}"
                )

        self.logg_data[f'{type}_outputs'] = []
        self.logg_data[f'{type}_y'] = []
        

    def training_step(self, batch, batch_idx):
        # print(batch_idx)
        x, y, labels = batch.values()
        logits = self(x)
        
        self.logg_data['train_outputs'].append(logits.detach().clone())
        self.logg_data['train_y'].append(y.detach().clone())
        lsfm_logits = logits.log_softmax(1)
        
        loss = self.loss_fn(lsfm_logits, y)
        self.log("Loss/Train", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.loggining('train')


    def validation_step(self, batch, batch_idx):
        # print(batch_idx)
        x, y, labels = batch.values()
        logits = self(x)

        self.logg_data['val_outputs'].append(logits.detach().clone())
        self.logg_data['val_y'].append(y.detach().clone())
        lsfm_logits = logits.log_softmax(1)
        
        loss = self.loss_fn(lsfm_logits, y)
        self.log("Loss/Val", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self.loggining('val')

    def configure_optimizers(self):
        module_path, class_name = self.hparams.training.optimizer.name.rsplit('.', 1)
        self.optimizer = getattr(import_module(module_path),class_name)\
            (self.parameters(), **self.hparams.training.optimizer.params)
        module_path, class_name = self.hparams.training.scheduler.name.rsplit('.', 1)
        self.scheduler = getattr(import_module(module_path),class_name)\
            (self.optimizer, **self.hparams.training.scheduler.params)
        return {'optimizer': self.optimizer, 'lr_scheduler':self.scheduler}

    def forward(self, X):
        X = self.model(X)
        return X


       
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="cfg", config_name="cfg")
def main(cfg : DictConfig):
    hparams = cfg.experiment

    pl.seed_everything(hparams.global_seed)
    # print(OmegaConf.to_yaml((hparams)))
    patients = glob(hparams.data.patients_re)
    train_patients, val_patients = data.random_split(patients, [hparams.data.train_dataset_size, 
                                                                hparams.data.val_dataset_size])
    dataset_train = MioDataset(train_patients, 'train', hparams)
    dataset_val = MioDataset(val_patients, 'val', hparams)
    
    train_loader = torch.utils.data.DataLoader(dataset_train, **hparams.data.train_loader)
    val_loader = torch.utils.data.DataLoader(dataset_val, **hparams.data.val_loader)
    
    if hparams.training.finetune.flag and hparams.training.finetune.new_scheduler:
        model = Model.load_from_checkpoint(
            checkpoint_path = hparams.training.finetune.path, hparams=hparams)
    else:
        model = Model(hparams)
        

    callbacks = [
        pl.callbacks.ModelCheckpoint(
                monitor='Loss/Train',
                filename=r'model-epoch={epoch:04d}-LossTrain={Loss/Train:.3f}',
                auto_insert_metric_name= False),
        pl.callbacks.ModelCheckpoint(
                monitor='Loss/Val',
                filename=r'model-epoch={epoch:04d}-LossVal={Loss/Val:.3f}',
                auto_insert_metric_name= False),
        pl.callbacks.ModelCheckpoint(
                filename=r'model-epoch={epoch:04d}',
                auto_insert_metric_name= False),
        pl.callbacks.LearningRateMonitor(),
    ]

    logger = pl.loggers.TensorBoardLogger(**hparams.logging.logger)

    trainer = pl.Trainer(logger=logger, callbacks=callbacks, **hparams.training.trainer)
    trainer.fit(model=model, 
                train_dataloaders=train_loader,
                val_dataloaders=val_loader, 
                ckpt_path=hparams.training.finetune.path if not hparams.training.finetune.new_scheduler and hparams.training.finetune.flag else None)
    
if __name__ == "__main__":
    main()