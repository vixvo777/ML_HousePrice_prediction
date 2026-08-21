import pandas as pd
import numpy as np

def load_data():
 data = pd.read_csv("data\Housing.csv")
 return data

def train():
    print("data loading\n")
    data = load_data()
    print("..........Training model..........\n")
    

    print("..........Training completed in branch feature-preprocessing.......... :)")

if __name__ == "__main__" :
    train()