from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
# from getpass import getpass
import os
import subprocess
import shutil
# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()

CHROMA_PATH = "chroma_investor"
DATA_PATH = "./Data"

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
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf")
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # document = chunks[10]
    # print(document.page_content)
    # print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, embedding, persist_directory=CHROMA_PATH
    )
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()