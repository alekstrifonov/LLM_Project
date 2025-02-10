import preprocess
from gensim.models import Word2Vec

def main():
    df = preprocess.json_to_df()
    df = preprocess.tokenize_data(df, tokeniz)
    

if __name__ == '__main__':
    main()