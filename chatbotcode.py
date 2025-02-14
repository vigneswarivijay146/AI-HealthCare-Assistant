import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util
import pandas as pd

CSV_FILE = "datasetforapp.csv"

# Detect GPU availability for acceleration
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Placeholder for "Loading..." message
loading_placeholder = st.empty()

# Show "Loading..." + spinner
with loading_placeholder, st.spinner("Loading..."):
    # Cache model loading
    @st.cache_resource(show_spinner=False)
    def load_model():
        return SentenceTransformer('distilbert-base-nli-stsb-mean-tokens', device=device)

    model = load_model()

    # Cache dataset loading
    @st.cache_data(show_spinner=False)
    def load_csv(file_path):
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip().str.lower()
            column_mapping = {df.columns[0]: "question", df.columns[1]: "answer"}
            df = df.rename(columns=column_mapping)

            if "question" not in df.columns or "answer" not in df.columns:
                st.error("CSV file must have 'Question' and 'Answer' columns!")
                return [], []

            return df["question"].tolist(), df["answer"].tolist()
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            return [], []

    medical_queries, medical_answers = load_csv(CSV_FILE)

    # Cache query embeddings
    @st.cache_data(show_spinner=False)
    def compute_query_embeddings(queries):
        with torch.no_grad():  # Disable gradient tracking for speed
            return model.encode(queries, convert_to_tensor=True, batch_size=64, device=device, dtype=dtype)

    query_embeddings = compute_query_embeddings(medical_queries)

# Remove "Loading..." once everything is initialized
loading_placeholder.empty()

# Streamlit UI - Chat Interface
st.title("AI-Powered Health Assistant🩺")
st.write("Hey hello!!👋I am your Health Assistant..Ask any health-related question, and I'll provide the best possible answer!")

# Initialize chat history if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history using st.chat_message for better UI
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(message)

# User Input
user_query = st.text_input("🔍 Type your message:")

# Define similarity threshold (higher value prevents incorrect answers)
SIMILARITY_THRESHOLD = 0.8

if st.button("Send"):
    if user_query.strip():
        st.session_state.chat_history.append(("user", user_query))

        with st.spinner("Thinking..."):
            with torch.no_grad():
                user_embedding = model.encode([user_query], convert_to_tensor=True, device=device, dtype=dtype)

                # Compute cosine similarity
                similarity_scores = util.cos_sim(user_embedding, query_embeddings)

                # Get top 3 matches
                top_matches = torch.topk(similarity_scores, k=3, dim=1)
                top_indices = top_matches.indices[0].tolist()
                top_scores = top_matches.values[0].tolist()

                # Ensure the best match meets the similarity threshold
                if top_scores[0] >= SIMILARITY_THRESHOLD:
                    best_match_index = top_indices[0]
                    best_response = medical_answers[best_match_index]
                else:
                    best_response = (
                        "I'm sorry, but I don't have information on that specific disease. "
                        "If you have questions about other diseases or health topics, feel free to ask!"
                    )

        # Store assistant's response in chat history
        st.session_state.chat_history.append(("assistant", best_response))

        # Rerun Streamlit to update UI
        st.rerun()
