
import torch
from torch.utils import data
try:
    from transform import *
except:
    from .transform import *
    
from os import path as osp
import glob
import collections

class DataCacheWrapper(data.Dataset):
    def __init__(self, data_set, cache_prob=0.6, cache_cap=1.0, *args, **kargs):
        self.stub = data_set
        self.cache_prob = cache_prob
        self.cache_cap = int(cache_cap if cache_cap > 1 else len(data_set)*cache_cap + 1)
        self.dt_caches = collections.deque(maxlen=self.cache_cap)

    def __getitem__(self, index, *args, **kargs):
        if self.cache_prob < 1e-6:
            return self.stub.__getitem__(index)
            
        if random.uniform(0, 1.0) < self.cache_prob: #cache hit, ask from cache
            try:
                return self.dt_caches.pop() 
            except	IndexError: #cach is empty, load from disk
                R = self.stub.__getitem__(index)
        else: #load from disk
            R = self.stub.__getitem__(index)
        self.dt_caches.appendleft(R)
        return R

    def __len__(self):
        return len(self.stub)