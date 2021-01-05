import torch
import dill
from typing import Union
from pathlib import Path
from torchtext import data

def save_checkpoint(save_path, model, valid_loss):

    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model):
    
    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=model.device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):

    if save_path == None:
        return
    
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path, device):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']

def save_cached_dataset(dataset: data.Example , path: Union[str,Path]):
    
    if not isinstance(path, Path):
        path = Path(path)

    path.mkdir(parents=True, exist_ok=True)
    torch.save(dataset.examples, path/"examples.pkl", pickle_module=dill)
    torch.save(dataset.fields, path/"fields.pkl", pickle_module=dill)
    torch.save(dataset.sort_key, path/"sort_key.pkl")

def load_cached_dataset(path: Union[str,Path]):

    if not isinstance(path, Path):
        path = Path(path)

    examples = torch.load(path/"examples.pkl", pickle_module=dill)
    fields = torch.load(path/"fields.pkl", pickle_module=dill)
    sort_key = torch.load(path/"sort_key.pkl", pickle_module=dill)

    ds= data.Dataset(examples, fields)
    ds.sort_key = sort_key

    return ds

