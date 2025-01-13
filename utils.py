import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
import random 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import time

OPTIMIZERS = {
    'adamw': torch.optim.AdamW,
    'adam': torch.optim.Adam,
}

LOSS_FNS = {
    'bce_wll': torch.nn.functional.binary_cross_entropy_with_logits,
    'bce': torch.nn.functional.binary_cross_entropy,
}

def get_predictions_bce(scores: torch.Tensor):
    return (scores >= 0).type(torch.int32) 

def get_predictions_ce(scores: torch.Tensor):
    return torch.argmax(scores, dim=1)

PRED_FNS = {
    'bce': get_predictions_bce,
    'ce': get_predictions_ce,
}


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.random.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_f1s(train_f1s, val_f1s, dest='./tmp/f1s.png'):
    plt.plot(train_f1s, label='train')
    plt.plot(val_f1s, label='val')
    plt.legend()
    plt.tight_layout()
    plt.savefig(dest)
    plt.close()

def get_schedule(initial_value, final_value, n_epochs, steps_per_epoch, warmup_epochs=0, start_warmup_value=0):
    # from https://github.com/facebookresearch/dino/blob/main/utils.py#L187
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * steps_per_epoch
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, initial_value, warmup_iters)

    iters = np.arange(n_epochs * steps_per_epoch - warmup_iters)
    schedule = final_value + 0.5 * (initial_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    
    return schedule

def get_embeddings(model, tokenizer, device, texts, bs=128, key='pooler_output'):
    model.to(device)
    model.eval()
    embeddings = []

    with torch.no_grad():
        for idx in range(0, len(texts), bs):
            batch = texts[idx:min(idx+128, len(texts))]
            batch = tokenizer(batch, return_tensors='pt', truncation=True, padding=True, max_length=512)
            output = model(**batch.to(device), return_dict=True)
            if key in output:
                embeddings.append(output[key].cpu().numpy())
            else:
                embeddings.append(output['last_hidden_state'][:, 0, :].cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings


def get_all_hidden_cls(model, tokenizer, device, texts, bs=128, key='pooler_output', max_depth=48):
    model.to(device)
    model.eval()
    all_hidden_cls = [[] for _ in range(max_depth)]
    embeddings = []

    with torch.no_grad():
        for idx in range(0, len(texts), bs):
            batch = texts[idx:min(idx+128, len(texts))]
            batch = tokenizer(batch, return_tensors='pt', truncation=True, padding=True, max_length=512)
            output = model(**batch.to(device), output_hidden_states=True, return_dict=True)
            if key in output:
                embeddings.append(output[key].cpu().numpy())
            else:
                embeddings.append(output['last_hidden_state'][:, 0, :].cpu().numpy())
            
            for i, hidden_state in enumerate(output['hidden_states']):
                all_hidden_cls[i].append(hidden_state[:, 0, :].cpu().numpy())

    embeddings = np.concatenate(embeddings)
    all_hidden_cls = all_hidden_cls[:len(output['hidden_states'])] ## cut off empty lists at the end
    
    for i in range(len(all_hidden_cls)):
        all_hidden_cls[i] = np.concatenate(all_hidden_cls[i], axis=0) # collapse to 1 tensor
       
    return all_hidden_cls, embeddings # embeddings could be post a pooler output, if the encoder has a pooler layer, else it is the last hidden state of the cls token


def load_embeddings(model: str, lang:str, layer=None) -> np.ndarray:
    path = f'./embeddings/{model.lower().split("/")[-1]}/{lang}/train'
    path += f'_{layer}.npy' if layer is not None else '.npy'
    embeddings = np.load(path)
    return embeddings

def load_split_indices(model: str, lang: str):
    train_indices = np.load(f'./embeddings/{model.lower().split("/")[-1]}/{lang}/train_indices.npy')
    val_indices = np.load(f'./embeddings/{model.lower().split("/")[-1]}/{lang}/val_indices.npy')

    return train_indices, val_indices



def train_lp_one_epoch(net, device, train_dl, criterion, optimizer, scheduler=None, use_tqdm=False, pred_func=get_predictions_bce):
    net.to(device)
    net.train()

    epoch_losses = []
    labels_ = []
    predictions_ = []

    net.zero_grad()
    optimizer.zero_grad()
    dl_iter = tqdm(train_dl) if use_tqdm else train_dl
    for batch_data, batch_labels in dl_iter:
        batch_data, batch_labels = batch_data.to(device), batch_labels.type(torch.float32).to(device)
        out = torch.flatten(net(batch_data))

        batch_loss = criterion(out, batch_labels)
        
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        epoch_losses.append(batch_loss.item())
        batch_predictions = pred_func(out.detach())
        labels_.append(batch_labels.cpu().numpy())
        predictions_.append(batch_predictions.cpu().numpy())
        
        if scheduler is not None:
            scheduler.step()
    
    labels_ = np.concatenate(labels_)
    predictions_ = np.concatenate(predictions_)

    # losses, (labels, prediction)
    return epoch_losses, (labels_, predictions_)


def validate_lp(net, device, val_dl, criterion, use_tqdm=False, pred_func=get_predictions_bce, return_out=False):
    net.to(device)
    net.eval()

    loss = 0
    num_batches = 0
    labels_ = []
    predictions_ = []
    outs_ = []
    dl_iter = tqdm(val_dl) if use_tqdm else val_dl
    with torch.no_grad():
        for batch_data, batch_labels in dl_iter:
            batch_data, batch_labels = batch_data.to(device), batch_labels.type(torch.float32).to(device)
            out = torch.flatten(net(batch_data))
            outs_.append(out.cpu().numpy())
            batch_loss = criterion(out, batch_labels)
                
            loss += batch_loss.item()
            num_batches += 1

            batch_predictions = pred_func(out)
            
            labels_.append(batch_labels.cpu().numpy())
            predictions_.append(batch_predictions.cpu().numpy())
    
    labels_ = np.concatenate(labels_)
    predictions_ = np.concatenate(predictions_)

    # loss, (predictions, label)
    if return_out:
        return loss / num_batches, (labels_, predictions_), np.concatenate(outs_)
    return loss / num_batches, (labels_, predictions_)


def train_lp(device, train_ds, val_ds, train_kwargs, seed, use_tqdm=False):

    set_seeds(seed)

    train_dl = DataLoader(train_ds, batch_size=train_kwargs['bs'], shuffle=True, num_workers=16)
    val_dl = DataLoader(val_ds, batch_size=train_kwargs['bs'], shuffle=False, num_workers=16)

    emb_dim = train_ds.embeddings.shape[1]
    layer = nn.Linear(emb_dim, train_kwargs['output_dim'], bias=True)
    optimizer = OPTIMIZERS[train_kwargs['optimizer']](layer.parameters(), lr=train_kwargs['lr'], weight_decay=train_kwargs['wd'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_kwargs['n_epochs'], eta_min=train_kwargs['lr_min'])

    loss_fn = LOSS_FNS[train_kwargs['loss_fn']]
    pred_fn = PRED_FNS[train_kwargs['pred_fn']]

    epoch_iter = range(train_kwargs['n_epochs'])
    if use_tqdm:
        epoch_iter = tqdm(epoch_iter)

    train_f1s = []
    val_f1s = []
    best_val_f1 = 0
    best_st = None ## best state dict
    for epoch in epoch_iter:
        
        _, (train_labels_, train_preds_) = train_lp_one_epoch(layer, device, train_dl, loss_fn, optimizer, pred_func=pred_fn)
        train_f1s.append(f1_score(train_labels_, train_preds_))
        _, (val_labels_, val_preds_) = validate_lp(layer, device, val_dl, loss_fn, pred_func=pred_fn)
        val_f1s.append(f1_score(val_labels_, val_preds_))
        
        if val_f1s[-1] > best_val_f1:
            best_val_f1 = val_f1s[-1]
            best_st = layer.cpu().state_dict()
        
        scheduler.step()
    return best_st, best_val_f1  ## return the layer st and the f1 score

def train_lp_one_epoch_balanced_class_loss(net, device, train_dl, criterion, optimizer, scheduler=None, use_tqdm=False, pred_func=get_predictions_bce):
    net.to(device)
    net.train()

    epoch_losses = []
    labels_ = []
    predictions_ = []
    net.zero_grad()
    optimizer.zero_grad()
    dl_iter = tqdm(train_dl) if use_tqdm else train_dl
    for batch_data, batch_labels in dl_iter:
        batch_data, batch_labels = batch_data.to(device), batch_labels.type(torch.float32).to(device)
        out = torch.flatten(net(batch_data))

        batch_losses = criterion(out, batch_labels, reduction='none')
        batch_losses = [torch.mean(batch_losses[batch_labels == label]) for label in batch_labels.unique()]
        batch_loss = sum(batch_losses) / len(batch_losses)

        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        epoch_losses.append(batch_loss.item())
        batch_predictions = pred_func(out.detach())
        labels_.append(batch_labels.cpu().numpy())
        predictions_.append(batch_predictions.cpu().numpy())
    
        if scheduler is not None:
            scheduler.step()
    labels_ = np.concatenate(labels_)
    predictions_ = np.concatenate(predictions_)

    # losses, (labels, predictions)
    return epoch_losses, (labels_, predictions_)

def train_lp_balanced_class_loss(device, train_ds, val_ds, train_kwargs, seed, use_tqdm=False, return_val_f1s=False):

    set_seeds(seed)

    train_dl = DataLoader(train_ds, batch_size=train_kwargs['bs'], shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=train_kwargs['bs'], shuffle=False, num_workers=0)
    emb_dim = train_ds.embeddings.shape[1]
    layer = nn.Linear(emb_dim, train_kwargs['output_dim'], bias=True)
    optimizer = OPTIMIZERS[train_kwargs['optimizer']](layer.parameters(), lr=train_kwargs['lr'], weight_decay=train_kwargs['wd'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_kwargs['n_epochs'], eta_min=train_kwargs['lr_min'])

    loss_fn = LOSS_FNS[train_kwargs['loss_fn']]
    pred_fn = PRED_FNS[train_kwargs['pred_fn']]

    epoch_iter = range(train_kwargs['n_epochs'])
    if use_tqdm:
        epoch_iter = tqdm(epoch_iter)

    train_f1s = []
    val_f1s = []
    best_val_f1 = 0
    best_st = None ## best state dict
    for epoch in epoch_iter:
        
        _, (train_labels_, train_preds_) = train_lp_one_epoch_balanced_class_loss(layer, device, train_dl, loss_fn, optimizer, pred_func=pred_fn)
        train_f1s.append(f1_score(train_labels_, train_preds_))
        _, (val_labels_, val_preds_) = validate_lp(layer, device, val_dl, loss_fn, pred_func=pred_fn)
        val_f1s.append(f1_score(val_labels_, val_preds_))
        
        if val_f1s[-1] > best_val_f1:
            best_val_f1 = val_f1s[-1]
            best_st = layer.state_dict()
            
        scheduler.step()
    if return_val_f1s:
        return best_st, best_val_f1, val_f1s
    return best_st, best_val_f1  ## return the layer st and the f1 score


def worst_group_accuracy(envs, labels, predictions):
    wga_ = 1
    envs = np.array(envs)
    labels = np.array(labels)
    predictions = np.array(predictions)

    for l in np.unique(labels):
        mask1 = labels == l
        for e in np.unique(envs):
            mask2 = envs == e
            mask = mask1 & mask2
            if sum(mask) > 0:
                wga_ = min(wga_, np.mean(labels[mask] == predictions[mask]))
    
    return wga_




