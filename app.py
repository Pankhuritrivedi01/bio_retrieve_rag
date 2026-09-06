import streamlit as st
from bio_retrieve_rag import agent  # Make sure this imports your agent

st.set_page_config(page_title="BioRetrieve RAG", layout="wide")
st.title("BioRetrieve RAG – Healthcare Assistant")
st.write("Ask questions according to WHO guidelines.")

# User input
query = st.text_input("Enter your question here:")

if st.button("Ask"):
    if query.strip() != "" :

       #Response output 
        response = agent.run(query)
        st.write("**Answer:**")
        st.write(response.content)
    else:
        st.warning("Please enter a question.")

