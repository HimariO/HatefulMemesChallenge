import random
import functools


class KFoldWrapper:

    def __init__(self, source_dataset, k=4, seed=1234):
        random.seed(seed)
        assert k > 1
        
        self.dataset = source_dataset
        self.k = k
        self.fold = 0
        self.get_fold_indies(self.fold)
    
    def get_fold_indies(self, n):
        self.fold_indies = list(range(len(self.dataset)))
        random.shuffle(self.fold_indies)
        self.fold_indies = [self.fold_indies[i::self.k] for i in range(self.k) if i != n]
        self.fold_indies = functools.reduce(lambda a, b: a + b, self.fold_indies)
    
    def set_fold(self, i):
        assert 0 <= i < self.k
        self.fold = i
        self.get_fold_indies(self.fold)
    
    def __len__(self):
        return len(self.dataset) - len(self.dataset) // 4
    
    def __getitem__(self, index):
        ture_index = self.fold_indies[index]
        return self.dataset[ture_index]
    
    @property
    def weights_by_class(self):
        return [self.dataset.weights_by_class[i] for i in self.fold_indies]
    
    @property
    def data_names(self):
        return self.dataset.data_names
    
    @property
    def test_mode(self):
        return self.dataset.test_mode
    
