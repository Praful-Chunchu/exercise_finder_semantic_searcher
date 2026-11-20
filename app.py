import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import streamlit as st


# Streamlit reruns this whole file on every user interaction.
# @st.cache_resource runs the function once and reuses the result,
# so the model is not reloaded on every search.
@st.cache_resource
def load_model():
    # Must be the SAME model used to build the database
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

@st.cache_resource
def load_collection():
    # Open the database saved by the notebook 
    client = chromadb.PersistentClient(path='chroma_db', settings=Settings(anonymized_telemetry=False))
    return client.get_collection("exercises")

def semantic_search(query, k=5):
    query_embedding = model.encode([query])
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=k)
    return results['documents'][0]

# Load once (cached across reruns)
model = load_model()
collection = load_collection()

st.title("Exercise Finder")
st.write("Describe what you want to train and get the 5 closest matching exercises.")

query = st.text_input("What do you want to train?")

# Only search once the user has typed something (an empty string counts as False)
if query:
    # enumerate numbers the results 1 to 5 instead of 0 to 4
    for i, doc in enumerate(semantic_search(query), start=1):
        st.markdown(f"**{i}.** {doc}")