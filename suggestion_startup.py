import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain.schema.output_parser import StrOutputParser
from datetime import datetime
import json
load_dotenv()
# Define the path to your JSON file
file_path = 'temp_2.json'
PROMPT_SUMMARY = """
You are an excellent 
"""
# Initialize an empty list to store the data
data_list = []

def load_json():
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
            return data_list
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{file_path}'. Please ensure it is a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    

prompt_template1 = PromptTemplate(
    template = """
    You are a Senior Due Diligence Analyst at a Venture Capital firm. You have been given a series of evaluations based on a comprehensive financial questionnaire for a target startup. Each evaluation includes a topic, a score, and the reasoning for that score.

    Your task is to synthesize these evaluations into a concise, professional, and actionable summary report for the Investment Committee. The report should focus on identifying critical flaws and necessary improvements in the startup's financial documentation and processes.

    **Input Data Provided:**
    A list of evaluation summaries, each containing:
    - The diligence topic (e.g., "Revenue Recognition," "Internal Controls").
    - A brief reasoning for the score with the score (1-10) at the end.
    
    The scores were decided based on following criteria:
    - **1-3 (High Risk / Red Flag):** Significant issues were found. This is a potential deal-breaker that requires immediate resolution.
    - **4-6 (Moderate Risk / Caution):** The findings raise concerns that require serious follow-up. The investment has notable challenges.
    - **7-8 (Positive Outlook / Minor Concerns):** Generally positive findings with minor, manageable risks. Appears to be a solid company.
    - **9-10 (Highly Appealing / Green Light):** Excellent findings, minimal risks, and strong indicators of a healthy, well-run business in this area.
    **Your Report Structure:**
    
    1.  **Executive Summary:**
        - Start with a high-level assessment. What is the overall state of the company's financial documentation and discipline based on the provided evaluations?
    
    2.  **Key Areas of Concern:**
        - Identify the top 3-5 most critical flaws revealed by the evaluations (prioritize those with the lowest scores or most severe reasoning).
        - For each flaw, briefly explain the risk it poses to the business or to a potential investment.
    
    3.  **Recommendations for Improvement:**
        - For each key concern, provide a clear, actionable recommendation. What specific steps should the company take to remedy the identified flaw?
    
    **Instructions:**
    - Base your report ONLY on the provided list of evaluations.
    - Do not invent information or make assumptions beyond the reasoning given.
    - Focus on synthesizing information into a coherent narrative, not just listing the low scores.
    - Maintain a professional and objective tone suitable for an investment committee.

    **Evaluations with score :**
    {string_evaluation}
    
    **Generate the Due Diligence Summary Report:**
    """,
    input_variables = ['string_evaluation']
    )

investor_template = PromptTemplate(
    template="""
    You are a Managing Director at a top-tier Private Equity firm responsible for authoring the final financial due diligence report for the Investment Committee. Your reputation rests on your ability to be thorough, insightful, and decisive.
    # CONTEXT
    The initial analysis of the company is complete. You have been provided with a list of key diligence questions and the corresponding analyst evaluations, which include a summary of findings, identified risks, and a score.

    # CORE TASK
    Synthesize the entire collection of provided analyst evaluations into a single, comprehensive, and professional Financial Due diligence Report. Your primary function is to identify patterns and systemic risks from these evaluations to form a final investment recommendation.

    # GUIDING PRINCIPLES
    1.  **Synthesize, Do Not Just List:** Identify patterns from the analyst notes. A theme of "Poor Internal Controls," for example, is more valuable than listing three separate low scores.
    2.  **Bottom Line Up Front (BLUF):** The Executive Summary must contain your final recommendation and its primary justifications.
    3.  **Reference the Findings:** Base your claims on the content of the analyst evaluations provided in the INPUT DATA section.
    4.  **Balance Strengths and Weaknesses:** Acknowledge positive findings (high scores) to provide a credible, balanced view.
    5.  **Focus on Materiality:** Prioritize risks that have a tangible impact on valuation, future cash flow, or investor control.

    # REQUIRED REPORT STRUCTURE
    (The structure remains the same as your original)
    ---
    ### **Financial Due Diligence Report: PowerLight Dynamics**
    **Date:** August 22, 2025
    **Prepared for:** Investment Committee
    **Version:** v1.5.43 - Final

    **1.0 Executive Summary**
    * **1.1 Overall Recommendation:** (Choose: Proceed with Investment / Proceed with Caution / Halt Process & Re-evaluate)
    * **1.2 Synopsis of Findings:** ...
    * **1.3 Key Strengths:** ...
    * **1.4 Critical Risks & Red Flags:** ...

    **2.0 Detailed Findings by Category**
    * **2.1 Financial Reporting & Integrity:** ...
    * **2.2 Revenue Quality & Unit Economics:** ...
    * (etc., for all categories)

    **3.0 Risk Assessment Matrix**
    | Risk Theme | Description of Finding | Potential Impact | Severity (High/Medium/Low) |
    |------------|------------------------|------------------|----------------------------|
    | ...        | ...                    | ...              | ...                        |

    **4.0 Final Conclusion & Justification**
    ...
    ---

    # INPUT DATA:
    You will now be provided with a list of diligence topics and their corresponding evaluations to synthesize.
    {string_evaluation}

    # Begin Report Generation.
    """,
    input_variables=['string_evaluation']
)

GROQ_MODEL_NAME = "openai/gpt-oss-20b"

def process_data(data_list):
    if not data_list:
        print("wrong data type, no list exists")
        return
    print("Initializing the main model:")
    model = ChatGroq(model=GROQ_MODEL_NAME)
    print("Extracting the evaluation and score from the temp.json")
    string_evaluation = "\n\n---\n\n".join([
        f"Analyst Evaluation:\n{d['key_points']}"
        for d in data_list
    ])
    print("Preparing the chain")
    chain = RunnableParallel(
        {
            "suggestion": prompt_template1 | model | StrOutputParser(),
            "final_report": investor_template | model | StrOutputParser()
        }
    )
    
    print("Structure of the chain: ")
    chain.get_graph().print_ascii()
    print("Waiting for response")
    final_output = chain.invoke({'string_evaluation': string_evaluation})
    
   

    print(f"""
          company suggestions:
          {final_output['suggestion']}
          
          investor side report:
          {final_output['final_report']}
          """) 
    
    return final_output

def store_in_file(final_output):
    # (Previous code from your process_data function remains the same)
    
    # --- NEW: WRITE TO TWO SEPARATE MARKDOWN FILES ---
    
    # 1. Prepare and save the company suggestions report
    company_suggestions_content = f"""
    # Suggestions for PowerLight Dynamics

    * **Date Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    * **Report Version:** v1.5.43

    ---

    This report outlines critical flaws and actionable recommendations for the company to improve its financial documentation and processes.

    {final_output['suggestion']}
    """
    company_file_name = "company_suggestions.md"
    with open(company_file_name, "w", encoding="utf-8") as f:
        f.write(company_suggestions_content)
    print(f"\n✅ Successfully saved company suggestions to '{company_file_name}'")

    # 2. Prepare and save the investor report
    investor_report_content = f"""
    # Due Diligence Report for the Investment Committee

    * **Date Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    

    ---

    This is the comprehensive internal report synthesizing all findings into a final investment recommendation.

    {final_output['final_report']}
    """
    investor_file_name = "investor_report.md"
    with open(investor_file_name, "w", encoding="utf-8") as f:
        f.write(investor_report_content)
    print(f"✅ Successfully saved investor report to '{investor_file_name}'")
    
    
    
def main():
    data_list = load_json()
    final_data = process_data(data_list)
    store_in_file(final_data)
    

    
if __name__ == '__main__':
    main()