import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import utils
import datasets_local
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


LANGS = ['eng', 'deu', 'esp', 'deu-esp', 'deu-esp-eng']
N_LAYERS = {
    'lealla-large': 24,
}
EMB_DIM = {
    'lealla-large': 256,
}
SEEDS = [1007, 1013, 1019]
train_kwargs = {
    'output_dim': 1,
    'optimizer': 'adamw',
    'loss_fn': 'bce_wll',
    'pred_fn': 'bce',
    'lr': 1e-3,
    'lr_min': 1e-5,
    'n_epochs': 50,
    'wd': 1e-3,
    'bs': 512,
}
model_name = "setu4993/LEALLA-large"
model_path_name = model_name.lower().split('/')[-1]
device = torch.device('cuda')


all_layer_ids = list(range(1,N_LAYERS[model_path_name]+1)) + [None]
lns = list(range(1,N_LAYERS[model_path_name]+1)) + ['emb']
idxs = list(range(1, N_LAYERS[model_path_name]+2))
assert len(idxs) == len(lns)
assert len(idxs) == len(all_layer_ids)

for lang in LANGS:
    print(lang)

    path = f'./classifiers/{lang}/'
    output_path = f'./results/{model_path_name}/{lang}/'
    os.makedirs(output_path, exist_ok=True) 

    emb_dim = EMB_DIM[model_path_name]

    mtd = pd.read_csv(f'./data/track_a/train/{lang}.csv')
    targets = [c for c in mtd.columns if c not in ['id', 'text']]
    target_labels = {
        c: mtd[c].to_numpy() for c in targets
    }

    train_indices, val_indices = utils.load_split_indices(model_name, lang)
    train_labels = {
        c: target_labels[c][train_indices] for c in targets
    }
    val_labels = {
        c: target_labels[c][val_indices] for c in targets
    }
    
    # label -1 means there is no label for that sample for emotion c
    train_masks = {
        c: train_labels[c] > -0.5 for c in train_labels.keys()
    }
    val_masks = {
        c: val_labels[c] > -0.5 for c in val_labels.keys()
    }

    train_datasets = {
        c: datasets_local.EmbeddingsDataset(None, train_labels[c][train_masks[c]]) for c in targets 
    }
    val_datasets = {
        c: datasets_local.EmbeddingsDataset(None, val_labels[c][val_masks[c]]) for c in targets
    }

    ## for each layer - do all emotions at once -> one ds per sentiment (due to labels); switch the embeddings at each layer
    ## just the simple linear probe - decide on a layer/depth that is ok

    layer_f1s = {
        c: [] for c in targets # list[tuple[float, float, float]]
    }
    layer_best_f1s = {
        c: [] for c in targets # list[float]
    }
    ## at the end make a table in a file: c: mean+std for each layer

    macro_f1s = []
    for layer_id, ln in tqdm(list(zip(all_layer_ids, lns))):
        path_layer = path + f'{ln}/'
        os.makedirs(path_layer, exist_ok=True)
        
        embeddings = utils.load_embeddings(model_name, lang, layer_id)
        train_embeddings = torch.tensor(embeddings[train_indices]).to(device)
        val_embeddings = torch.tensor(embeddings[val_indices]).to(device)

        current_layer_f1s = []

        for c in targets:
            st_path = path_layer + f'model_{c}.pt'
            if os.path.exists(st_path):# and False:
                best_net_st = torch.load(st_path, map_location='cpu')
                f1s = best_net_st['f1s']
                best_val_f1 = best_net_st['f1'] ## :=: max(f1s)
            else:
                train_datasets[c].embeddings = train_embeddings[train_masks[c]]
                val_datasets[c].embeddings = val_embeddings[val_masks[c]]
                f1s = []
                best_f1 = 0
                best_net_st = None
                for seed in SEEDS:
                    net_st, best_val_f1 = utils.train_lp_balanced_class_loss(device, train_datasets[c], val_datasets[c], train_kwargs, seed, use_tqdm=False)
                    f1s.append(best_val_f1)
                    if best_val_f1 > best_f1:
                        best_f1 = best_val_f1
                        best_net_st = net_st
                        best_net_st['f1'] = best_f1 # add f1 score to the state dict -> can retrieve data from saved st if training stops midway
                
                best_net_st['f1s'] = f1s # save all f1s as well for final plots and stats
                torch.save(best_net_st, st_path) # best classifier for this emotion & layer
    
            layer_f1s[c].append(f1s)
            current_layer_f1s.append(best_val_f1)
            layer_best_f1s[c].append(best_val_f1)
        
        macro_f1s.append(np.mean(current_layer_f1s))
        
        for c in targets:
            plt.plot(layer_best_f1s[c], label=c)
        plt.legend()
        plt.ylabel('F1')
        plt.xlabel('Layer')
        plt.tight_layout()
        plt.savefig(output_path + 'per_emotion_f1.png')    
        plt.close()

    
    plt.plot(idxs, macro_f1s)
    plt.ylabel('Macro F1')
    plt.xlabel('Layer')
    plt.tight_layout()
    plt.savefig(output_path + 'macro_f1.png')
    plt.close()

    np.save(output_path + 'layer_best_f1s.npy', layer_best_f1s, allow_pickle=True)
    np.save(output_path + 'layer_f1s.npy', layer_f1s)
    args = [np.argmax(layer_best_f1s[c]) for c in targets]
    best_f1s_ = [layer_best_f1s[c][arg] for c,arg in zip(targets, args)]
    best_macro_f1 = np.mean(best_f1s_) # max f1 across layer for each column; mean of these f1s

    summary = {
        c: {
            'f1': best_f1s_[i],
            'layer': args[i],
        } for i, c in enumerate(targets)
    }
    summary['macro_f1'] = best_macro_f1
    np.save(output_path + 'summary.npy', summary)
    print("BEST f1", best_macro_f1)


    for c in targets:
        plt.plot(idxs, layer_best_f1s[c], label=c)
    plt.legend()
    plt.ylabel('F1')
    plt.xlabel('Layer')
    plt.tight_layout()
    plt.savefig(output_path + 'per_emotion_f1.png')
    plt.close()

    for c in targets:
        for idx, f1s in enumerate(layer_f1s[c], start=1):
            plt.scatter([idx for _ in SEEDS], f1s, marker='^')
        
        plt.ylabel('F1')
        plt.xlabel('Layer')
        plt.tight_layout()
        plt.savefig(output_path + f'f1s_{c}.png')
        plt.close()

    