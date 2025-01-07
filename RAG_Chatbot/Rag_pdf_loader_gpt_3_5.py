from langchain_community.document_loaders import PyPDFLoader
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader,TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.output_parsers import StrOutputParser
from bs4 import BeautifulSoup
from colorama import Fore, Back, Style, init
init(autoreset=True)


""" Input: 

- The script processes PDF files in a specified folder. 
- Queries entered by the user in the terminal. 

Output: 

- Relevant documents retrieved from the database. 
- Answers to user queries based on the retrieved documents. 

Functionality: 

1. Load and process PDF documents from a specified folder. 
2. Store the processed documents in ChromaDB for retrieval. 
3. Use OpenAI GPT-3.5 model to answer user queries based on the retrieved documents.

"""

# Set your OpenAI API key
os.environ["openai_apikey"] = "Please provide the OpenAPI Key Here"


openai_model = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.environ.get("openai_apikey"), temperature=0.5, max_tokens=500)

persistent_directory = os.path.join(os.getcwd(), "db", "chroma_manuals")

# Check if the database directory exists
if os.path.exists(persistent_directory):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.environ.get("openai_apikey"))
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 10})
else:
    print(f"Creating new ChromaDB at {persistent_directory}")
    
    folder_path = "Manuals"  # Replace with the path to your folder containing PDF files
    all_documents = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing file: {file_path}")
            
            loader = PyPDFLoader(file_path)
            documents = loader.load()

            # Split the text into chunks
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)
            all_documents.extend(docs)

    # Initialize embeddings model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.environ.get("openai_apikey"))

    # Store vectors in ChromaDB
    db = Chroma.from_documents(all_documents, embeddings, persist_directory=persistent_directory)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 13})

# Query for similarity search
exit = True
while exit:
    query = input(Fore.WHITE + Style.BRIGHT + "\nEnter the Query: ")

    # Retrieve relevant documents
    relevant_docs = retriever.invoke(query)

    # Prepare input for the OpenAI model
    combined_input = (
        "Here are some documents that might help answer the question: "
        + query
        + "\n\nRelevant Documents:\n"
        + "\n\n".join([doc.page_content for doc in relevant_docs])
        + "\n\nPlease provide the answer based on the provided documents. If the answer is not found, don't answer the question. Provide a concise answer."
    )

    messages = [
        SystemMessage(content="You are a helpful assistant"),
        HumanMessage(content=combined_input),
    ]

    # Invoke OpenAI model
    output = openai_model.invoke(messages)
    parser = StrOutputParser()
    result = parser.invoke(output)

    print(Fore.GREEN + Style.BRIGHT + "\nOutput:\n")
    print(Fore.GREEN + Style.BRIGHT + result)

    value = input(Fore.BLUE + Style.BRIGHT + "\nIF you want to ask one more question: PRESS Y ,if you want to exit PRESS N :")

    if value.lower() == "y":
        exit = True
    if value.lower() == "n":
        exit = False

    print("\n=========== End of Output ====================================================\n")
