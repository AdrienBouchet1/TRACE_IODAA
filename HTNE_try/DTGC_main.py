import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import sys

sys.path.insert(0,'TRACE_module' ) # insert path to be able to import file afterwards, allows for imports wherever the script is called from

from DTGC import HTNEDataSet, HTNE_a

if __name__ == "__main__":
    # Load the data
    
    print(sys.path)
    
    model = HTNE_a(
        file_path="HTNE_try/cows.txt", 
        emb_size=128,
        neg_size=10,
        hist_len=2,
        directed=False,
        epoch_num=50
    )

    # Train the model
    model.train()
