from torch.utils.data import Dataset
import numpy as np

class EmbeddingsDataset(Dataset):

    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class CBEmbeddingsDataset(Dataset):

    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = np.array(labels)
        self.classes = np.unique(labels)
        self.n_classes = len(self.classes)

        self.class_indices = []
        for cls in self.classes:
            mask = self.labels == cls
            idxs = np.nonzero(mask)[0]
            self.class_indices.append(idxs)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx_):
        cls = np.random.randint(0, self.n_classes)
        idx = np.random.choice(self.class_indices[cls], size=1)[0]
        return self.embeddings[idx], self.labels[idx]

class CBDatasestMTB(Dataset):

    def __init__(self, data, task_labels, batch_size):
        self.data = np.array(data)

        self.task_labels = np.array(task_labels)
        self.classes = [0, 1]
        self.n_classes = 2 # 0/1
        self.n_tasks = self.task_labels.shape[1]

        self.batch_size = batch_size

        self.class_indices = {}
        for task_idx in range(self.n_tasks):
            class_indices_ = []
            for cls in self.classes:
                mask = self.task_labels[:, task_idx] == cls
                idxs = np.nonzero(mask)[0]
                class_indices_.append(idxs)
            self.class_indices[task_idx] = class_indices_

    def __len__(self):
        return len(self.task_labels)
    
    def __getitem__(self, idx_):
        tasks = np.random.randint(0, self.n_tasks, size=self.batch_size)
        cls = np.random.randint(0, self.n_classes, size=self.batch_size)
        idxs = []
        for i in range(self.batch_size):
            idxs.append(np.random.choice(self.class_indices[tasks[i]][cls[i]], size=1)[0])
        
        return list(self.data[idxs]), self.task_labels[idxs]


class LangAwareEmbeddingsDataset(Dataset):

    def __init__(self, embeddings, labels, langs):
        self.embeddings = embeddings
        self.labels = labels
        self.langs = langs

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx], self.langs[idx]
    

