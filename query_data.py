from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
from getpass import getpass
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
import subprocess

load_dotenv()     

huggingface_token = os.getenv("HF_TOKEN")
hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
CHROMA_PATH = "chroma_investor"

PROMPT_TEMPLATE = """
A question will be asked based on financial data.
You are an expert in financial analysis and due diligence.
You will be provided with relevant context from a database of financial documents.
Your task is to answer the question based on the provided context.
If the context does not contain enough information, you should indicate that you cannot answer the question.
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
llm = HuggingFaceEndpoint(
        repo_id='moonshotai/Kimi-K2-Instruct',
        task='text-generation'
    )
    
model = ChatHuggingFace(llm=llm)
#intfloat/e5-large-v2
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


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
    # 1. Get query from user
    # Note: The login_to_huggingface() function is no longer needed.
    login_to_huggingface()
    query_text = input("Enter your query: ")

    # 2. Prepare the DB
    # Fix for the deprecation warning: Use langchain_chroma
    # You'll need to run: pip install langchain-chroma
    # And change your import to: from langchain_chroma import Chroma
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # 3. Search the DB
    results = db.similarity_search_with_relevance_scores(query_text, k=30)
    if not results or results[0][1] < 0.5:
        print("Unable to find matching results.")
        return

    # 4. Create the prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print("--- Generated Prompt ---")
    print(prompt)
    print("------------------------")

    # 5. Initialize the model (it will automatically find the environment variable)
    # llx
    
    # 6. Get the response
    print("\n⏳ Awaiting response from model...")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"\n✅ Response: {response_text}\n   Sources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    # Make sure to set the HUGGINGFACEHUB_API_TOKEN environment variable first!
    main()