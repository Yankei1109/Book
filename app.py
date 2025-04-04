import streamlit as st
import pandas as pd
from recommender import (
    content_based_recommendations,
    collaborative_filtering_recommendations,
    hybrid_recommendations
)

@st.cache_data
def load_data():
    try:
        books = pd.read_csv('data/Books.csv', low_memory=False, encoding='latin-1')
        ratings = pd.read_csv('data/Ratings.csv', low_memory=False, encoding='latin-1')
        users = pd.read_csv('data/Users.csv', low_memory=False, encoding='latin-1')
        
        # Basic cleaning
        books = books.dropna(subset=['Book-Title', 'Book-Author'])
        books = books.drop_duplicates(subset=['ISBN'])
        
        # Filter ratings
        ratings = ratings[ratings['ISBN'].isin(books['ISBN'])]
        ratings = ratings[ratings['Book-Rating'] != 0]
        
        # Reduce memory
        books = books[['ISBN', 'Book-Title', 'Book-Author', 'Publisher']]
        ratings = ratings[['User-ID', 'ISBN', 'Book-Rating']]
        
        return books, ratings, users
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

books, ratings, users = load_data()

# Streamlit app
st.title("📚 Book Recommender System")

# Sidebar
approach = st.sidebar.selectbox(
    "Recommendation Approach",
    ["Content-Based", "Collaborative Filtering", "Hybrid"]
)

st.sidebar.markdown("### Dataset Stats")
st.sidebar.write(f"Books: {len(books)}")
st.sidebar.write(f"Users: {len(users)}")
st.sidebar.write(f"Ratings: {len(ratings)}")

if approach == "Content-Based":
    st.header("Content-Based Recommendations")
    book_title = st.selectbox(
        "Select a book you like",
        books['Book-Title'].unique()
    )
    if st.button("Get Recommendations"):
        with st.spinner('Finding similar books...'):
            recs = content_based_recommendations(book_title, books)
        st.dataframe(recs)

elif approach == "Collaborative Filtering":
    st.header("Collaborative Filtering (SVD)")
    user_id = st.number_input(
        "Enter User ID",
        min_value=1,
        max_value=int(users['User-ID'].max()) if len(users) > 0 else 1000,
        value=1
    )
    if st.button("Get Recommendations"):
        with st.spinner('Finding personalized recommendations...'):
            recs = collaborative_filtering_recommendations(user_id, books, ratings)
        st.dataframe(recs)

elif approach == "Hybrid":
    st.header("Hybrid Recommendations")
    col1, col2 = st.columns(2)
    with col1:
        user_id = st.number_input(
            "User ID",
            min_value=1,
            max_value=int(users['User-ID'].max()) if len(users) > 0 else 1000,
            value=1
        )
    with col2:
        book_title = st.selectbox(
            "Book Title",
            books['Book-Title'].unique()
        )
    if st.button("Get Recommendations"):
        with st.spinner('Combining approaches...'):
            recs = hybrid_recommendations(user_id, book_title, books, ratings)
        st.dataframe(recs)

