from datasets import load_dataset
import os

def load_templerun_dataset(data_dir):
    ds = load_dataset('imagefolder', data_dir=data_dir)
    return ds
