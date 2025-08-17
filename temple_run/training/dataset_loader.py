from datasets import load_dataset
import os

def load_templerun_dataset(data_dir):
    ds = load_dataset('imagefolder', data_dir=data_dir)
    return ds

if __name__ == '__main__':
    # Example usage - run from project root
    dataset = load_templerun_dataset("data_collector/templerun_dataset")
    print(dataset["train"][0].keys())
