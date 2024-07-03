import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pickle

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configure Google Generative AI with the API key
genai.configure(api_key=api_key)

# Define the path to your shared drive
SHARED_DRIVE_PATH = "G:/Shared drives/BRP Internal"

def list_directories(drive_path):
    """
    List all directories in the specified path.
    
    Args:
    drive_path (str): The path to list directories from.
    
    Returns:
    list: List of directory names.
    """
    return [f.name for f in os.scandir(drive_path) if f.is_dir()]

def list_files(drive_path):
    """
    List all PDF files in the specified path.
    
    Args:
    drive_path (str): The path to list files from.
    
    Returns:
    list: List of PDF file names.
    """
    return [f.name for f in os.scandir(drive_path) if f.is_file() and f.name.endswith('.pdf')]

def get_pdf_text(file_path):
    """
    Extract text from a PDF file along with page numbers.
    
    Args:
    file_path (str): The path to the PDF file.
    
    Returns:
    list: List of tuples containing text and page number.
    """
    text = []
    try:
        pdf_reader = PdfReader(file_path)
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text.append((page_text, page_num))
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

def get_text_chunks(text):
    """
    Split the text into chunks for processing, keeping track of page numbers.
    
    Args:
    text (list): List of tuples containing text and page number.
    
    Returns:
    list: List of tuples containing text chunks and corresponding page numbers.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    chunks = []
    for page_text, page_num in text:
        splits = text_splitter.split_text(page_text)
        chunks.extend([(split, page_num) for split in splits])
    return chunks

def get_vector_store(text_chunks):
    """
    Create a vector store from text chunks using embeddings and save page numbers.
    
    Args:
    text_chunks (list): List of tuples containing text chunks and page numbers.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    texts = [chunk[0] for chunk in text_chunks]
    vector_store = FAISS.from_texts(texts, embedding=embeddings)
    vector_store.save_local("faiss_index")
    with open("faiss_index/page_numbers.pkl", "wb") as f:
        pickle.dump(text_chunks, f)

def get_conversational_chain():
    """
    Create a conversational chain for question answering.
    
    Returns:
    chain: The question-answering chain configured with the prompt and model.
    """
    prompt_template = """
    Use the provided context to answer the question as accurately as possible. Provide detailed responses and if the information is not available in the context, state that the answer is not available in the context.\n\n
    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    """
    Process user input, perform similarity search, and generate a response.
    
    Args:
    user_question (str): The question asked by the user.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load the FAISS index and perform similarity search
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=5)  # Retrieve more documents for better context

    # Get the conversational chain and generate a response
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    # Load text chunks with page numbers
    with open("faiss_index/page_numbers.pkl", "rb") as f:
        text_chunks = pickle.load(f)

    # Map responses to page numbers
    response_with_pages = []
    for doc in docs:
        for chunk, page_number in text_chunks:
            if doc.page_content in chunk:
                response_with_pages.append(page_number)
                break

    # Display the response and the corresponding page numbers
    st.write("Reply: ", response["output_text"])
    st.write("Page Number(s): ", sorted(set(response_with_pages)))  # Display unique page numbers in sorted order

def main():
    """
    Main function to set up the Streamlit interface and handle user interactions.
    """
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    # Input field for user's question
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        
        # Persistent state to store the current folder path
        if 'current_folder' not in st.session_state:
            st.session_state.current_folder = SHARED_DRIVE_PATH

        folder_path = st.session_state.current_folder

        # Navigation through folders
        st.write(f"Current folder: {folder_path}")
        subfolders = list_directories(folder_path)
        subfolders.insert(0, "..")  # Option to go up one level
        selected_subfolder = st.selectbox("Select Folder", subfolders, key='folder_select')

        if st.button("Change Folder"):
            if selected_subfolder == "..":
                st.session_state.current_folder = os.path.dirname(folder_path)
            else:
                st.session_state.current_folder = os.path.join(folder_path, selected_subfolder)
            st.experimental_rerun()

        # List PDF files in the selected folder
        files = list_files(folder_path)
        st.write("Files in Folder:")
        for file_name in files:
            st.write(file_name)

        # File selector
        selected_file_name = st.text_input("Enter File Name to Process")
        if st.button("Process"):
            with st.spinner("Processing..."):
                # Access and process the selected file from the local Google Drive directory
                file_path = os.path.join(folder_path, selected_file_name)
                if os.path.exists(file_path):
                    raw_text = get_pdf_text(file_path)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
                else:
                    st.error(f"File {file_path} not found.")

if __name__ == "__main__":
    main()
