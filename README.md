# Movie Title Finder: Identify Your Movie Based on a Description
Have you ever had those times when you are **really** trying to recomend a movie to a friend? but you happened to **FORGET** its name? No worries, this project is to solve exactly that problem! As long as you have a **vague memory** of what the movie is about, we will help you find its title!

This project implements a **content-based movie recommendation system** using **TF-IDF** (Term Frequency-Inverse Document Frequency) and **cosine similarity**. It takes a dataset of **top 500 IMDB movies**, processes the, and recommends the most relevant movies based on a user's input query.

## Features

- **Preprocess movie descriptions** using `gensim` text processing tools.
- **Build a TF-IDF model** for feature extraction.
- **Compute similarities** using a **cosine similarity matrix**.
- **Provide movie title** based on user input.

## Installation

### Prerequisites

Ensure you have Python 3.10 (gensim is **not yet compatible** with the newest version of Python!) installed along with the required dependencies:

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

To run the title finding system, execute the script with a **query describing your movie's plot description**:

```bash
python contentRecommender.py "A wealthy stockbroker’s lavish life unravels amid crime and corruption."
```

This is a description of Wolf of Wallstreet if you haven't noticed! I **LOVE** that movie!

### Example Output

```
Top Recommendations:
    Title               Similarity
1   The Wolf of Wall Street             0.376509
2   Touch of Evil            0.165567
3   Sin City          0.156059
4   The Game     0.154993
5   Chinatown       0.147820
```

## Code Overview

- **`load_dataset()`**: Loads the movie dataset from `contentData.csv`.
- **`preprocess_text(text)`**: Cleans and tokenizes text using `gensim` preprocessing filters.
- **`prepare_corpus(df)`**: Converts movie descriptions into TF-IDF vectors.
- **`recommend_movies(query, dictionary, tfidf, index, df, top_n=5)`**: Finds the top `N` most similar movies based on cosine similarity.
- **`main()`**: Parses the user query and provides recommendations.

## Notes

- The recommendations are based on text similarity, so results depend on how well the descriptions capture the movie's plot.
- The system works best when descriptions are **well-written and descriptive**.


## Future Improvements

- Expand dataset to include **more movies**.
- Implement **word embeddings (e.g., Word2Vec, BERT)** for better similarity matching.
- Allow filtering recommendations by **genres, year, or director**.

## Expected Salary
- 1600$ per month
---
