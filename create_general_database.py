from langchain_community.document_loaders import WikipediaLoader, DirectoryLoader, MergedDataLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
# from getpass import getpass
import os
import subprocess
import shutil
# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()

CHROMA_PATH = "chroma_general"
DATA_PATH = "./Data_General"

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
# huggingface_token = os.getenv("HF_TOKEN")
hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

def login_to_huggingface():
    """
    Logs into Hugging Face Hub using a token from an environment variable.
    This function executes the command in a PowerShell subprocess.
    """
    # 1. Get the Hugging Face token from environment variables
    

    if not hf_token:
        print("❌ Error: HF_TOKEN environment variable not set.")
        print("Please set the HF_TOKEN environment variable before running this script.")
        return

    print("🔑 Token found. Preparing to log in...")

    # 2. Construct the PowerShell command
    # We pass the token directly into the command string.
    command = f"hf auth login --token {hf_token}"

    try:
        # 3. Execute the command using PowerShell
        #    - 'powershell': Specifies the executable to run.
        #    - '-Command': Tells PowerShell to execute the following string.
        #    - check=True: Raises an error if the command fails (returns a non-zero exit code).
        #    - capture_output=True: Captures stdout and stderr.
        #    - text=True: Decodes stdout/stderr as text.
        result = subprocess.run(
            ["powershell", "-Command", command],
            check=True,
            capture_output=True,
            text=True
        )

        # 4. Print the successful output
        print("\n✅ Login successful!")
        print("\n--- PowerShell Output ---")
        print(result.stdout)
        if result.stderr:
            print("\n--- PowerShell Error Stream (may be empty) ---")
            print(result.stderr)

    except FileNotFoundError:
        print("❌ Error: 'powershell.exe' not found. Is PowerShell installed and in your system's PATH?")
    except subprocess.CalledProcessError as e:
        # This block runs if the command returns a non-zero exit code (i.e., an error)
        print(f"❌ An error occurred during login.")
        print(f"Return Code: {e.returncode}")
        print("\n--- PowerShell Output (stdout) ---")
        print(e.stdout)
        print("\n--- PowerShell Error (stderr) ---")
        print(e.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        

def main():
    generate_data_store()


def generate_data_store():
    login_to_huggingface()
    generate_and_store_chunks_lazily()


def generate_and_store_chunks_lazily():
    """
    Loads documents one by one, splits them, and adds them to ChromaDB
    to keep memory usage low.
    """
    # 1. Initialize the loader and splitter
    loader = WikipediaLoader(query="Due Diligence", load_max_docs=20, lang="en")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    
    loader_2 = (DirectoryLoader(DATA_PATH, glob="*.pdf"))
    merged_loader = MergedDataLoader(loaders=[loader, loader_2])
    # 2. Clear out the database directory
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # 3. Create a persistent ChromaDB client
    # We will add documents to this iteratively.
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embedding # Ensure your embedding function is defined
    )

    chunks_added = 0
    # 4. Use lazy_load() to get an iterator
    print("Starting lazy load and processing...")
    for doc in loader.lazy_load():
        # 5. Split the single document into chunks
        chunks = text_splitter.split_documents([doc]) # Note: split_documents expects a list
        
        # 6. Add the chunks from this one document to ChromaDB
        db.add_documents(chunks)
        chunks_added += len(chunks)
        print(f"-> Loaded & processed '{doc.metadata['title']}'. Added {len(chunks)} chunks.")

    print(f"\n✅ Finished processing.")
    print(f"Saved a total of {chunks_added} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()