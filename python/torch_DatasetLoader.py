import pandas as pd

class DatasetLoader():
    def __init__(self, fileName):
        dataset = pd.read_pickle(fileName)
        self.label = list(dataset.label)
        self.feature = list(dataset.feature)

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        sample = {'feature': self.feature[idx:idx+1], 
                  'label': self.label[idx:idx+1]}
        return sample
    
    def getBatches(self,batch_size):
        batches = []
        nBatch = int(len(self.label)/batch_size)
        for i in range(nBatch):
            istart, iend = i*batch_size, (i+1)*batch_size
            
            batch = {'feature': self.feature[istart:iend],
                     'label': self.label[istart:iend]}
            yield batch