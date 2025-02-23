# Movie Recommendation System

This project implements a **content-based movie recommendation system** using **TF-IDF** (Term Frequency-Inverse Document Frequency) and **cosine similarity**. It takes a dataset of **top 500 IMDB movies**, processes their descriptions, and recommends the most relevant movies based on a user's input query.

## Features

- **Preprocess movie descriptions** using `gensim` text processing tools.
- **Build a TF-IDF model** for feature extraction.
- **Compute similarities** using a **cosine similarity matrix**.
- **Recommend movies** based on user input.

## Installation

### Prerequisites

Ensure you have Python 3 installed along with the required dependencies:

```bash
pip install pandas numpy gensim argparse
```

### Dataset

The script requires `contentData.csv`, which should contain at least:

| Title | Description |
|-------|------------|
| Movie Name 1 | Movie description 1 |
| Movie Name 2 | Movie description 2 |
| ...   | ...  |

Make sure the CSV file is present in the same directory as the script.

## Usage

To run the recommendation system, execute the script with a **query describing your movie preferences**:

```bash
python movie_recommender.py "A thrilling adventure with futuristic themes"
```

### Example Output

```
Top Recommendations:
    Title               Similarity
1   Inception             0.768
2   The Matrix            0.745
3   Interstellar          0.723
4   Blade Runner 2049     0.700
5   Minority Report       0.678
```

## Code Overview

- **`load_dataset()`**: Loads the movie dataset from `contentData.csv`.
- **`preprocess_text(text)`**: Cleans and tokenizes text using `gensim` preprocessing filters.
- **`prepare_corpus(df)`**: Converts movie descriptions into TF-IDF vectors.
- **`recommend_movies(query, dictionary, tfidf, index, df, top_n=5)`**: Finds the top `N` most similar movies based on cosine similarity.
- **`main()`**: Parses the user query and provides recommendations.

## Notes

- The recommendations are based on text similarity, so results depend on how well the descriptions capture the movie's themes.
- The system works best when descriptions are **well-written and descriptive**.

## Future Improvements

- Expand dataset to include **more movies**.
- Implement **word embeddings (e.g., Word2Vec, BERT)** for better similarity matching.
- Allow filtering recommendations by **genres, year, or director**.

---
