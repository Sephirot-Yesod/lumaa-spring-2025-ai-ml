import pandas as pd
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import MatrixSimilarity
import argparse
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_numeric, remove_stopwords

def load_dataset():
    """Loads dataset from contentData (500 top IMDB movies)."""
    df = pd.read_csv("./contentData.csv")
    df = df[['Title', 'Description']].dropna()
    return df

def preprocess_text(text):
    """Processes away redundant elements with gensim"""
    filters = [strip_tags, strip_punctuation, strip_numeric, remove_stopwords]
    return preprocess_string(text, filters)

def prepare_corpus(df):
    """convert text data into vectors"""
    df['Processed'] = df['Description'].apply(preprocess_text)
    dictionary = Dictionary(df['Processed'])
    corpus = [dictionary.doc2bow(text) for text in df['Processed']]
    tfidf = TfidfModel(corpus)
    index = MatrixSimilarity(tfidf[corpus], num_features=len(dictionary))
    return dictionary, tfidf, index, corpus

def recommend_movies(query, dictionary, tfidf, index, df, top_n=5):
    """Uses matrix similarity to find the most appropriate match"""
    query_tokens = preprocess_text(query)
    query_bow = dictionary.doc2bow(query_tokens)
    query_tfidf = tfidf[query_bow]
    similarities = index[query_tfidf]
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    recommendations = df.iloc[top_indices][['Title']].copy()
    recommendations['Similarity'] = similarities[top_indices]
    return recommendations

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="User's preference description") #adds a parser to the argument - query for user input 
    args = parser.parse_args()
    
    df = load_dataset()
    dictionary, tfidf, index, corpus = prepare_corpus(df)
    recommendations = recommend_movies(args.query, dictionary, tfidf, index, df)
    
    print("\nTop Recommendations:")
    print(recommendations.to_string(index=False))

if __name__ == "__main__":
    main()
