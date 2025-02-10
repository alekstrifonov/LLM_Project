from preprocess import json_to_df
import json
import matplotlib.pyplot as plt
import pandas as pd

def main(): 
    df = json_to_df(False)
    
    print(df)                    

if __name__ == '__main__':
    main()