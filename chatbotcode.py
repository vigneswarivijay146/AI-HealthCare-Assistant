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
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip().str.lower()
        column_mapping = {df.columns[0]: "question", df.columns[1]: "answer"}
        df = df.rename(columns=column_mapping)

        if "question" not in df.columns or "answer" not in df.columns:
            st.error("CSV file must have 'Question' and 'Answer' columns!")
            st.stop()

        return df["question"].tolist(), df["answer"].tolist()

    medical_queries, medical_answers = load_csv(CSV_FILE)

    # Cache query embeddings
    @st.cache_data(show_spinner=False)
    def compute_query_embeddings(queries):
        with torch.no_grad():  # Disable gradient tracking for speed
            return model.encode(queries, convert_to_tensor=True, batch_size=32, device=device, dtype=dtype)

    query_embeddings = compute_query_embeddings(medical_queries)

# Remove "Loading..." once everything is initialized
loading_placeholder.empty()

# Streamlit UI - Chat Interface
st.title("AI-Powered Health Assistant🩺")
st.write("Hey hello!!👋I am your Health Assistant..Ask any health-related question, and I'll provide the best possible answer!")

# Initialize chat history if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for role, message in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"**👤 You:** {message}")
    else:
        st.markdown(f"**🤖 Assistant:** {message}")

# User Input
user_query = st.text_input("🔍 Type your message:")

# Submit Button
if st.button("Send"):
    if user_query:
        st.session_state.chat_history.append(("user", user_query))

        with st.spinner("Thinking..."):
            with torch.no_grad():  # Speed up inference
                user_embedding = model.encode(user_query, convert_to_tensor=True, device=device, dtype=dtype)
                similarity_scores = util.dot_score(user_embedding, query_embeddings)
                best_match_index = torch.argmax(similarity_scores).item()
                best_response = medical_answers[best_match_index]

        # Store assistant's response in chat history
        st.session_state.chat_history.append(("assistant", best_response))

        # Rerun Streamlit to update UI
        st.rerun()
