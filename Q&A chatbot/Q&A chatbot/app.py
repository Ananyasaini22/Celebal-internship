import streamlit as st
from retriever import DataRetriever
from llm_interface import generate_answer

st.title("RAG Q&A Chatbot - Loan Approval Data")

query = st.text_input("Ask a question about the loan dataset (yes or no questions are preferred):")

if query:
    retriever = DataRetriever("data/Training Dataset.csv")
    context = " ".join(retriever.get_relevant_chunks(query))
    answer = generate_answer(context, query)

    st.subheader("Answer:")
    st.write(answer)

    st.subheader("Context Used:")
    st.write(context)
