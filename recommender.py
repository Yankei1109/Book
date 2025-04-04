import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Dataset, Reader, KNNWithMeans, SVD 
from scipy.sparse import csr_matrix
import os
import pickle

def content_based_recommendations(book_title, books, n=5):
    """Memory-efficient content-based recommendations"""
    if 'combined_features' not in books.columns:
        books['combined_features'] = (
            books['Book-Title'].str.lower() + ' ' + 
            books['Book-Author'].str.lower().fillna('') + ' ' + 
            books['Publisher'].str.lower().fillna('')
        )
    
    # Use smaller max_features and binary=True to reduce memory
    tfidf = TfidfVectorizer(stop_words='english', max_features=2000, binary=True)
    tfidf_matrix = tfidf.fit_transform(books['combined_features'])
    
    try:
        idx = books.index[books['Book-Title'].str.lower() == book_title.lower()].tolist()[0]
    except IndexError:
        return pd.DataFrame(columns=['Book-Title', 'Book-Author', 'Publisher'])
    
    # Compute only the similarities we need
    cosine_sim = linear_kernel(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
    
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    book_indices = [i[0] for i in sim_scores]
    
    return books.iloc[book_indices][['Book-Title', 'Book-Author', 'Publisher']]

def collaborative_filtering_recommendations(user_id, books, ratings, n=5):
    """Memory-efficient collaborative filtering using SVD"""
    # Take a sample of the data
    ratings_sample = ratings.sample(frac=0.2) if len(ratings) > 50000 else ratings
    
    # Prepare data
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(ratings_sample[['User-ID', 'ISBN', 'Book-Rating']], reader)
    
    # Use SVD which is much more memory efficient than KNN
    algo = SVD(n_factors=20, n_epochs=10, verbose=False)
    trainset = data.build_full_trainset()
    algo.fit(trainset)
    
    # Get books not rated by user
    rated_books = ratings_sample[ratings_sample['User-ID'] == user_id]['ISBN'].values
    books_to_predict = [book for book in books['ISBN'].unique() if book not in rated_books]
    
    # Predict in small batches
    predictions = []
    batch_size = 50
    for i in range(0, len(books_to_predict), batch_size):
        batch = books_to_predict[i:i+batch_size]
        predictions += [algo.predict(user_id, book) for book in batch]
    
    # Get top recommendations
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_n = predictions[:n]
    
    recommended_books = []
    for pred in top_n:
        try:
            book_info = books[books['ISBN'] == pred.iid].iloc[0]
            recommended_books.append({
                'Title': book_info['Book-Title'],
                'Author': book_info['Book-Author'],
                'Publisher': book_info['Publisher'],
                'Estimated Rating': round(pred.est, 2)
            })
        except IndexError:
            continue
    
    return pd.DataFrame(recommended_books)

def hybrid_recommendations(user_id, book_title, books, ratings, n=5):
    """Memory-efficient hybrid approach"""
    # First get content-based recommendations
    content_recs = content_based_recommendations(book_title, books, min(n*2, 20))  # Limit to 20
    
    if len(content_recs) == 0:
        return pd.DataFrame(columns=['Title', 'Author', 'Publisher', 'Estimated Rating'])
    
    # Prepare collaborative filtering data with sample
    ratings_sample = ratings.sample(frac=0.2) if len(ratings) > 50000 else ratings
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(ratings_sample[['User-ID', 'ISBN', 'Book-Rating']], reader)
    
    # Use SVD for memory efficiency
    algo = SVD(n_factors=20, n_epochs=10, verbose=False)
    trainset = data.build_full_trainset()
    algo.fit(trainset)
    
    # Predict ratings for content-based recommendations
    predictions = []
    for _, row in content_recs.iterrows():
        isbn = books[books['Book-Title'] == row['Book-Title']]['ISBN'].values[0]
        pred = algo.predict(user_id, isbn)
        predictions.append((row['Book-Title'], row['Book-Author'], row['Publisher'], round(pred.est, 2)))
    
    # Sort by predicted rating and return top n
    predictions.sort(key=lambda x: x[3], reverse=True)
    return pd.DataFrame(predictions[:n], columns=['Title', 'Author', 'Publisher', 'Estimated Rating'])
