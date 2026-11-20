# Exercise Semantic Search

Describe what you want to train in plain English and get the 5 most relevant exercises back, matched by meaning instead of exact keywords.

**Live app:** [exercise-finder.streamlit.app](https://exercise-finder.streamlit.app)

## Data
Gym Exercise Dataset (Kaggle, niharika41298): 2,918 exercises with title, description, body part, equipment, type, and level.

## How it works
1. **Chunking:** one exercise = one chunk. Title, body part, equipment, type, and description are joined into a single string. Missing fields are skipped.
2. **Embedding:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
3. **Storage:** ChromaDB, persisted to `chroma_db/`.
4. **Search:** the query is embedded with the same model, and the 5 nearest exercises are returned.
5. **Interface:** Streamlit.

## Run locally
1. `pip install -r requirements.txt`
2. `streamlit run app.py`

To rebuild the database, download the CSV into `data/` and run `semantic_search_starter_notebook.ipynb`.

## Limitations
1. About 53% of exercises have no description, so those rows are embedded from the title and labels only. Short title-only rows can rank high on one-word queries.
2. Slang (e.g. "swole") is weakly understood by a small embedding model.
3. Some source labels are inaccurate (e.g. a pec stretch labeled as Abdominals).

## Possible improvements
Drop the rows with no description, try a larger embedding model, or add hybrid keyword and semantic search.