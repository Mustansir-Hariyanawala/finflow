import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain.schema.output_parser import StrOutputParser
import subprocess
import json
# Load environment variables from .env file
load_dotenv()

financial_due_diligence_questions = [
    "Analyze the complete, audited annual financial statements (Balance Sheet, Income Statement, Statement of Cash Flows) for the three most recent fiscal years. Synthesize the findings from the full, unabridged auditor's opinion letter for each year to assess the integrity of financial reporting and identify any qualifications, emphasis of matter, or going concern paragraphs that indicate underlying financial or operational risks.",
    "Review all unaudited interim financial statements (P&L, Balance Sheet, Cash Flow) for all completed quarters since the last audited period and for the current year-to-date period. Identify and summarize any significant deviations or trends in financial performance that have emerged since the last formal audit, focusing on changes in revenue growth, margin stability, and cash flow generation to assess real-time financial visibility.",
    "Examine all auditor management letters, reports on internal controls, and similar correspondence from the past three fiscal years, along with the company's written responses. Synthesize these communications to identify any recurring 'significant deficiencies' or 'material weaknesses' in internal controls. Assess the effectiveness and timeliness of management's remediation efforts to gauge the company's commitment to strong financial governance.",
    "Provide a detailed analysis of all significant accounting policies. Identify and quantify the impact of any changes to these policies within the last three years to detect any accounting manipulations aimed at obscuring performance trends. The analysis should ensure a true 'apples-to-apples' comparison of financial results over the period.",
    "Generate a detailed, segmented revenue schedule for the past 36 months, breaking down revenue by primary streams (e.g., subscription, usage-based, professional services, hardware). Analyze this data to identify trends in revenue composition, assess the quality and predictability of the top-line, and flag any concentration risks that could impact future stability.",
    "Reconstruct the reconciliation from non-GAAP metrics, such as Bookings and ARR/MRR, to GAAP/IFRS revenue for each of the past 12 quarters. Analyze this reconciliation to determine if management's preferred 'vanity metrics' are providing a misleading picture of performance and to verify the true, recognized revenue trend.",
    "Compile and analyze a schedule of the top 20 customers by revenue for each of the last three fiscal years. For each customer, calculate their percentage of total revenue to identify and quantify the level of customer concentration risk. Assess the stability and trend of this concentration to determine the business's vulnerability to the loss of a key account.",
    "Analyze the complete deferred revenue waterfall schedule as of the most recent month-end. Evaluate the projected recognition schedule for all contracted but unearned revenue to assess the quality and predictability of the future revenue stream and to verify the integrity of the company's revenue recognition practices.",
    "Investigate and disclose any unbilled receivables or revenue recognized from contracts where cash has not yet been invoiced. Based on the provided customer contracts, assess the contractual justification for this recognition to identify any aggressive or non-compliant revenue recognition policies that could overstate performance.",
    "Deconstruct the Cost of Goods Sold (COGS) or Cost of Revenue for the past 12 quarters, providing a detailed, itemized breakdown of all components (e.g., hosting, third-party software/API fees, customer support salaries, implementation costs). Analyze this breakdown to assess the integrity of the reported gross margin and understand its primary drivers.",
    "Calculate and present the Gross Margin percentage on a quarterly basis for the past 12 quarters, both for the company overall and segmented by each major product or service line. Analyze these trends to evaluate the scalability of the business model and the profitability of its core offerings.",
    "If the Gross Margin has declined in any of the past 8 quarters, generate a detailed narrative analysis explaining the specific drivers of this decline. The analysis should investigate potential causes such as increased support costs, pricing pressure, or architectural inefficiencies to identify any underlying technical or operational debt.",
    "Formulate a 'per-customer' cost analysis by estimating the average monthly COGS attributable to a single customer. Detail the methodology and assumptions used in this calculation to assess the scalability of the customer support model and the per-unit profitability of the service.",
    "Provide a detailed, trended analysis of all operating expenses (OpEx) for the past 12 quarters, categorized by department (Sales & Marketing, R&D, G&A). Evaluate the spending in each area to identify operational inefficiencies and understand the company's strategic investment priorities.",
    "Conduct a variance analysis by comparing the budget vs. actual spending for each department over the past two fiscal years and the current year-to-date. Generate a report explaining the root causes for any variances greater than 10% to assess the company's financial discipline and forecasting accuracy.",
    "Analyze the schedule of headcount by department for each of the past 12 quarter-ends. Correlate this data with departmental spending to identify the primary drivers of operating expenses and assess the company's workforce efficiency and allocation of human capital.",
    "Evaluate the return on investment (ROI) of R&D spending over the past 12 quarters. Correlate the trend of R&D expenses as a percentage of revenue with a list of all patents filed and major product features shipped during the same period to determine if the investment in innovation is yielding tangible, value-creating results.",
    "Scrutinize the schedule of all related-party transactions over the past three years. Analyze these transactions to identify any potential conflicts of interest, self-dealing, or non-arm's-length arrangements between the company and its officers, directors, or significant shareholders that could indicate poor corporate governance or value leakage.",
    "Deconstruct and validate the company's Customer Acquisition Cost (CAC) calculation methodology. The analysis must identify every general ledger account and the percentage of that account's expense allocated to the CAC numerator for a given period to ensure the calculation is 'fully loaded' and not manipulated to understate costs, thereby confirming the sustainability of the unit economics.",
    "Generate a quarterly trended analysis of the company's blended CAC for the past 12 quarters. Evaluate this trend to identify any decline in marketing efficiency or signs of market saturation, which could threaten the future profitability of customer acquisition.",
    "Analyze the breakdown of CAC by major acquisition channel (e.g., Paid Digital, Content/SEO, Direct Sales, Channel Partners) for the past 8 quarters. Evaluate the efficiency and scalability of each channel to identify any saturation risks or over-reliance on a single, potentially volatile, acquisition source.",
    "Audit the 'new customers' denominator used in the CAC calculation for each quarter by referencing the provided CRM reports and product analytics data. Assess the validity of the company's definition of a 'new customer' to detect any manipulation intended to artificially lower the reported CAC.",
    "Investigate and justify the exclusion of any sales or marketing-related expenses (e.g., salaries of onboarding specialists, a portion of executive salaries) from the CAC calculation. This analysis is critical to uncover any hidden acquisition costs and arrive at a true, fully-loaded CAC.",
    "Deconstruct and validate the company's calculation for Customer Lifetime Value (LTV). The analysis must clearly identify and assess the reasonableness of the inputs used for Average Revenue Per Account (ARPA), Gross Margin %, and Churn Rate to ensure the LTV projection is accurate and reliable.",
    "Present a quarterly trended analysis of LTV for the past 8 quarters. Generate a narrative explaining the drivers behind any significant changes in the underlying inputs (ARPA, Gross Margin, Churn) to assess whether the value of an average customer is improving or declining over time.",
    "Conduct a comprehensive cohort analysis of Net Revenue Retention (NRR) for each monthly or quarterly customer vintage acquired since inception. Track the revenue trajectory of each cohort over time to assess product-market fit, identify trends in customer value, and determine if the company's ability to retain and expand revenue from its customers is improving.",
    "Perform a cohort analysis of Logo Retention for the same vintages and periods as the NRR analysis. Compare the two analyses to determine if the company is retaining customers but losing revenue, or if both customer count and revenue are stable, providing a deeper insight into customer satisfaction.",
    "Generate a detailed, quantitative breakdown of the drivers of Net Revenue Retention (NRR) for each of the last 8 quarters. The analysis should quantify the specific revenue contributions from expansion, contraction, and churn to distinguish between sustainable growth from up-sells versus a 'leaky bucket' scenario masked by a few large expansions.",
    "Synthesize data from churn surveys, CRM notes, and customer support logs to compile a qualitative analysis of the primary reasons cited by customers for both logo churn and revenue contraction over the past 24 months. This analysis should identify any recurring themes related to competitive threats, product gaps, or service deficiencies.",
    "Calculate and present the LTV:CAC ratio on a quarterly trended basis for the past 12 quarters. Analyze this trend to determine if the fundamental health of the business model is improving or deteriorating. A ratio consistently below 3:1 should be flagged as a significant risk to long-term sustainability.",
    "Calculate and present the CAC Payback Period (in months) on a quarterly trended basis for the past 12 quarters. Analyze this trend to assess the company's capital efficiency. A lengthening payback period may indicate declining sales efficiency or an unsustainable cash burn required to fuel growth.",
    "Generate a narrative analysis explaining the underlying business drivers of any significant changes (positive or negative) in the LTV:CAC ratio and Payback Period. This analysis should connect the financial metrics to strategic decisions, such as changes in pricing, marketing strategy, or product focus, to demonstrate a clear understanding of the business's economic levers.",
    "Based on the company's current and trended unit economics, generate a projection of the monthly recurring revenue (MRR) level at which the company is expected to achieve cash flow break-even. The response must include the financial model and key assumptions (e.g., future CAC, LTV, and overhead costs) supporting this projection to assess the company's path to profitability.",
    "Analyze the company's detailed, bottom-up operating and financial model for the next 36 months, including the monthly P&L, Balance Sheet, and Cash Flow Statement projections. Evaluate the internal consistency and mechanical correctness of the model to identify any unrealistic financial planning or flawed logic.",
    "Deconstruct the 'Assumptions' tab of the financial model. For all key operational drivers (e.g., new logo acquisition, ARPA growth, churn rates, sales rep quota, hiring plan), evaluate the reasonableness of the projections by comparing them to historical performance and industry benchmarks to identify any unsubstantiated or overly optimistic forecasts.",
    "Verify the credibility of the financial model's forward-looking projections by providing the historical data for the past 12-24 months for each key assumption. This direct comparison is essential to challenge any 'hockey stick' projections that are not grounded in past performance, thereby assessing the realism of the forecast.",
    "Perform a sensitivity or scenario analysis on the financial model. The analysis should quantify the impact on revenue projections and cash runway based on a +/- 20% variance in the top three most sensitive assumptions (e.g., churn rate, new customer acquisition rate, ARPA). This will stress-test the model and reveal the company's vulnerability to key operational risks.",
    "Calculate and provide the Net Burn and Gross Burn for each of the past 12 months, based on the provided Statements of Cash Flow and bank statements. Analyze the trend in these burn rates to assess whether the company's cash consumption is accelerating or decelerating.",
    "Based on the most recent month's Net Burn rate and the current cash balance, calculate the company's current cash runway in months. This calculation is a critical indicator of the company's immediate solvency and the urgency of the current financing round.",
    "Generate a detailed breakdown of the company's average monthly Net Burn into three strategic categories: 1) Growth Investment (all costs included in a fully-loaded CAC), 2) Product Investment (all R&D expenses), and 3) Overhead (all G&A and other non-growth/product expenses). This analysis will assess the 'quality' of the burn and how efficiently capital is being deployed.",
    "Evaluate the return on investment for the 'Growth' and 'Product' portions of the company's cash burn over the past 12 months. Justify the analysis with specific KPIs, such as new ARR added per dollar of growth investment and key features shipped per dollar of product investment, to determine if capital is being allocated efficiently.",
    "Identify the minimum cash balance required to operate the business and review the board-approved contingency plan for a scenario where the cash balance approaches this critical level. This will assess management's foresight and preparedness for potential adverse financial situations.",
    "Analyze the Accounts Receivable (AR) aging schedule as of the most recent month-end. Identify the percentage of total AR that is more than 90 days past due and evaluate the trend over the last four quarters to assess the effectiveness of the company's cash collection process.",
    "From the AR aging report, identify any specific customers that represent more than 10% of the balance that is over 90 days past due. Provide a summary of the status of collection efforts and any related disputes for these accounts to assess the risk of bad debt and potential customer satisfaction issues.",
    "Analyze the Accounts Payable (AP) aging schedule as of the most recent month-end. Evaluate the company's payment patterns to its suppliers and identify any risks to key supplier relationships that could be indicated by significantly delayed payments.",
    "Based on the standard customer agreement and contracts with major customers, describe the company's typical payment terms. Identify any significant deviations for large customers to assess the company's negotiating power and its impact on working capital.",
    "Compile and analyze a schedule of the company's top 10 vendors by spend. For each vendor, detail their standard payment terms to identify any dependencies or working capital strains that may arise from these key relationships.",
    "Calculate and provide the Cash Conversion Cycle (CCC) for each of the last four quarters. Analyze the trend in the CCC to assess whether the company's operational cash efficiency is improving or deteriorating over time.",
    "Validate the current, fully-diluted capitalization table, ensuring it is certified by a company officer. The analysis must confirm that it includes all classes of stock, all issued shares, all outstanding options, warrants, and other convertible securities to provide a definitive view of the ownership structure and identify all potential sources of dilution.",
    "Examine the employee stock option pool detailed in the cap table. The analysis should verify the total number of shares authorized, granted, and remaining available for grant to determine if there is a sufficient incentive pool to attract and retain future talent, or if an expansion (and further dilution) will be necessary.",
    "Generate a pro-forma capitalization table that models the impact of the current proposed financing round. The model must show the conversion of all outstanding convertible instruments (e.g., SAFEs, convertible notes) and the issuance of new shares at the proposed valuation to accurately project the post-money ownership for all stakeholders and avoid any misunderstanding of dilution.",
    "Uncover any 'shadow' dilution risk by identifying and disclosing any written or verbal commitments to issue equity (shares, options, or otherwise) that are not yet reflected on the current capitalization table. This requires a review of board minutes, offer letters, and consulting agreements.",
    "Review all 409A valuation reports commissioned by the company to date. Verify that the exercise price for all stock option grants was set at or above the fair market value established in these reports to ensure compliance with tax regulations and avoid any 'cheap stock' issues that could create liabilities for employees and the company.",
    "Compile a list of all security holders with special rights, such as pro-rata rights, information rights, or board observation rights. Analyze the agreements granting these rights to identify any terms that could encumber the company's governance or restrict the rights of a new investor.",
    "Analyze all agreements related to any past or present indebtedness (including loan agreements, credit agreements, venture debt facilities, and security agreements). The goal is to identify any hidden covenants, liens, or other obligations that could pose a risk to the company or the investor's position.",
    "Generate a summary of all financial, affirmative, and negative covenants contained within the company's debt agreements. Report any instances of non-compliance in the past 24 months to assess the risk of a covenant breach, which could lead to a default.",
    "Detail any security interests, liens, or other encumbrances that have been granted over any company assets, with a specific focus on intellectual property. This analysis is crucial to determine if an investor's position could be subordinated to a lender in a liquidation scenario.",
    "Uncover and quantify all off-balance-sheet liabilities by reviewing lease agreements, guarantees, and contracts with indemnification obligations. Provide a schedule of all future minimum payments for these commitments to reveal any hidden financial liabilities that could impact valuation and future cash flow.",
    "Identify and analyze any 'change of control' provisions within debt or material commercial agreements. Determine if the proposed financing or a future acquisition would trigger these provisions, potentially creating transaction obstacles or requiring costly consent payments.",
    "Provide a detailed schedule of all capital leases and their associated assets and liabilities. This analysis is necessary to ensure all capital obligations are properly disclosed and accounted for on the balance sheet.",
    "Evaluate the company's tax compliance risk by reviewing complete copies of all filed federal, state, and local income tax returns for the three most recent fiscal years. Identify any inconsistencies or potential audit red flags.",
    "Assess the company's potential payroll and sales tax liabilities by reviewing all sales & use tax and payroll tax filings for the past 12 quarters. Identify any jurisdictions where filings may be required but are not being made.",
    "Map the company's operational footprint by compiling a list of all jurisdictions (states and foreign countries) where the company has or has had employees, contractors, physical property, or significant revenue in the last three years. For each jurisdiction, confirm whether all relevant taxes are being filed to identify any unrecorded 'nexus' tax liabilities.",
    "For any jurisdiction where the company has a taxable presence ('nexus') but does not file taxes, provide and analyze the legal memo or opinion that forms the basis for this position. This will assess the aggressiveness of the company's tax positions and the associated risk of future penalties.",
    "Review all correspondence with any tax authority (IRS, state, local, foreign) regarding any audits, inquiries, or disputes within the past five years. Synthesize these communications to identify any ongoing or unresolved tax disputes that could result in a material liability.",
    "Validate any R&D tax credits claimed in the past three years by reviewing the supporting documentation and technical studies. Assess the risk that these credits could be disallowed upon audit, resulting in a repayment obligation.",
    "Evaluate the risk of fraud or unauthorized spending by analyzing the company's written internal control process for cash disbursements over $5,000. The analysis should confirm that there is a clear approval workflow with designated authorized individuals.",
    "Assess the risk of expense reimbursement abuse by reviewing the company's formal expense policy and the process for submitting, reviewing, and approving expense reports, particularly for the executive team.",
    "Evaluate the risk of inaccurate financial reporting by analyzing the company's documented process for the monthly financial close. The analysis should confirm that financial statements are formally reviewed and approved by management before being presented to the board.",
    "Assess the risk of internal fraud by determining if there is a formal segregation of duties in the finance/accounting function (e.g., confirming the person who approves payments is different from the person who executes them). If not, evaluate the effectiveness of any compensating controls.",
    "Generate a summary and analysis of all insurance policies currently in force. The analysis should evaluate the adequacy of coverage types (D&O, E&O, Cyber, General Liability), coverage limits, and annual premiums to ensure there is sufficient mitigation for key business risks.",
    "Investigate the company's claims history by identifying if any insurance claims have been denied in the past three years. Provide details on any denied claims to identify potentially uninsurable or poorly managed risks.",
    "Assess the risk of voided cyber insurance coverage by reviewing the application submitted to the carrier. The analysis should identify any potential misrepresentations regarding the company's security posture that could be used to deny a future claim.",
    "Compile a comprehensive summary of any pending, threatened, or historical litigation, arbitration, or governmental investigation involving the company or its officers. Analyze this summary to quantify any contingent legal liabilities that could impact the company's financial health.",
    "Verify the company's regulatory compliance by providing a list of all government licenses and permits required to operate the business. Confirm that all are current and in good standing to ensure there are no impediments to continued operation.",
    "Analyze the monthly cash flow statements for the last 24 months. Deconstruct the cash flows from operations, investing, and financing activities to identify patterns of cash flow volatility and assess the company's ability to self-fund its operations and growth.",
    "Disclose and analyze any use of receivable factoring or other non-traditional financing arrangements over the past 24 months. Evaluate the terms of these arrangements to identify any costly or predatory financing practices that may signal underlying financial distress.",
    "Review and analyze the company's fixed asset register. The analysis should include a summary of assets by purchase date, cost, accumulated depreciation, and net book value to assess the age and condition of the asset base and identify any potential needs for near-term capital expenditure.",
    "Evaluate the adequacy of the company's bad debt reserve by reviewing its accounting policy and the specific calculation used for the reserve for the last four quarters. This analysis will determine if there is a risk of understated bad debt, which could overstate the value of accounts receivable.",
    "Analyze the trend in the average sales cycle length (from initial contact to closed-won deal) for a new customer over the past 8 quarters. A lengthening sales cycle can be a leading indicator of declining sales efficiency, increased competition, or a weakening product-market fit.",
    "Provide a detailed schedule and analysis of sales commissions for the last 8 quarters. The analysis should detail the commission structure (e.g., % of TCV, upfront vs. recurring) and the total expense to assess the cost of sales and its impact on profitability.",
    "Assess the quality of customer support by analyzing the performance metrics (e.g., CSAT, NPS, first response time) for the past 12 months. Poor customer satisfaction can be a leading indicator of future churn and revenue loss.",
    "Review all material contracts with suppliers or partners and identify any with exclusivity clauses or minimum purchase commitments. Analyze these clauses to identify any restrictive commercial agreements that could limit the company's operational flexibility or create future financial burdens.",
    "Generate a list of all software used to run the business (e.g., CRM, ERP, HRIS) and the annual cost for each. Analyze this list to assess the level of G&A overhead and identify any potential for cost efficiencies.",
    "Disclose and analyze any severance or deferred compensation plans or agreements in place for any employees or executives. This is critical to uncover hidden employee liabilities that could be triggered by a change of control or future terminations.",
    "Provide a breakdown of the workforce by department, distinguishing between independent contractors and full-time employees. Analyze this data to assess the risk of worker misclassification, which could lead to significant back taxes, penalties, and legal liabilities.",
    "Compile a list of all bank accounts, credit card accounts, and investment accounts held by the company, including the authorized signatories for each. Review this list to identify any potential financial control gaps or unauthorized accounts.",
    "Disclose if the company has ever undergone a 'down round' financing. If so, provide a detailed analysis of the terms and the impact on the capitalization table to understand the company's historical valuation issues and any associated complex investor rights.",
    "Provide and evaluate the company's disaster recovery and business continuity plan from both a financial and operational perspective. Assess the plan's adequacy to mitigate the risk of significant business disruption from unforeseen events.",
    "Analyze the company's revenue recognition policy, specifically in the context of multi-element arrangements (e.g., contracts that include software subscription, implementation, and support). Assess the policy for compliance with ASC 606/IFRS 15 to ensure revenue is being recognized appropriately over the life of the contract.",
    "Generate a bridge analysis that explains the variance between the revenue projected in the prior year's budget and the actual revenue achieved. This analysis will assess the accuracy of the company's forecasting and its ability to execute against its financial plan.",
    "For any usage-based revenue streams, analyze the key metrics tracked (e.g., API calls, data storage) and the historical usage data for the past 12 months. This analysis will assess the volatility and predictability of these revenue streams.",
    "Review and analyze all shareholder agreements, voting agreements, or other arrangements that govern the control and management of the company. The goal is to identify any complex governance structures or shareholder rights that could impact a new investor's influence or control.",
    "Assess the company's talent retention risk by analyzing the average employee tenure and attrition rates by department (specifically engineering vs. sales) for the last two years. High attrition in key departments can signal underlying cultural, management, or compensation issues.",
    "Provide a comprehensive list of any company assets that have been pledged as collateral for any loans or other obligations. This analysis is critical to understand the extent of asset encumbrance and the seniority of lenders' claims.",
    "Disclose and analyze any instances where the company has repurchased shares from any founder, employee, or investor. Review the details of the transaction, including the price and justification, to identify any unusual capital transactions or potential self-dealing.",
    "Provide a detailed breakdown of marketing spend by category (e.g., paid ads, content, events, PR) for the last 8 quarters. Analyze this data to identify any over-reliance on a single marketing channel and to assess the diversification and sustainability of the company's lead generation strategy.",
    "Review all real estate lease agreements for any physical locations (offices, data centers). Analyze these agreements to identify and quantify any long-term lease liabilities, rent escalation clauses, or restrictive covenants.",
    "If the company operates internationally, provide and analyze its formal transfer pricing policy for all inter-company transactions between domestic and foreign subsidiaries. This is to assess the risk of an international tax audit and potential penalties from tax authorities.",
    "Generate a summary and analysis of the key terms of the employee stock option plan. The analysis should focus on vesting schedules (e.g., 4-year with 1-year cliff), the post-termination exercise period, and any acceleration provisions upon a change of control to assess the competitiveness and employee-friendliness of the plan."
]

# You can verify the number of questions with:
# print(len(financial_due_diligence_questions))
huggingface_token = os.getenv("HF_TOKEN")
hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

# --- Configuration ---
CHROMA_PATH_1 = "chroma_investor"
CHROMA_PATH_2 = "chroma_general"
# Using the same embedding model as in the database creation script
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Specify the model to use from Groq's offerings
GROQ_MODEL_NAME = "llama3-8b-8192"

PROMPT_TEMPLATE_RETRIEVER = """
    You are a meticulous financial data extractor. Your sole task is to answer the user's question based *exclusively* on the provided financial context. Do not infer, speculate, or use any outside knowledge. If the context does not contain the information to answer the question, you must state: "The provided context does not contain sufficient information to answer this question."

    **Context:**
    {context}

    **Question:**
    {question}

    **Factual Answer:**
    """


PROMPT_TEMPLATE_WHY = """
    You are a seasoned Senior Investment Analyst conducting late-stage financial due diligence. You are highly skeptical, detail-oriented, and your reputation is on the line.

    You have been provided with:
    1.  **The Original Diligence Question:** The specific query being investigated.
    2.  **The Extracted Answer:** A factual answer pulled directly from the company's documents.
    3.  **A General Diligence Framework:** A set of best practices for what to look for.

    Your task is to write a critical evaluation of the **Extracted Answer**. Do not simply repeat the information. Your goal is to identify underlying risks, strengths, and what might be missing.

    **Original Diligence Question:** {question}
    **Extracted Answer from Company Docs:** {answer}
    **General Due Diligence Framework:** {context2}

    ---
    **CRITICAL EVALUATION MEMO**

    Provide your analysis in the following structured format. Be professional, concise, and direct.

    **1. Key Findings:**
    - Summarize the most critical positive and negative facts presented in the 'Extracted Answer'. What is the most important takeaway?

    **2. Potential Risks & Red Flags:**
    - Based on the 'Diligence Framework', what potential risks, inconsistencies, or red flags does the 'Extracted Answer' reveal or hint at?
    - Are there signs of aggressive accounting, poor financial health, or operational inefficiency?
    - What crucial information appears to be missing that would be required for a complete picture?

    **3. Overall Assessment & Next Steps:**
    - Based on this single data point, what is your preliminary assessment of the company's health *in this specific area*?
    - What are the top 2-3 follow-up questions you would immediately ask the management team based on this analysis?

    Begin your evaluation now:
"""

PROMPT_TEMPLATE_SENTIMENT = """
    You are an Investment Committee scoring model. Your function is to analyze a due diligence memo and assign a numerical score representing the investment's attractiveness based *only* on that evaluation.

    **Analyst's Memo to Score:**
    {answer2}

    ---
    **INSTRUCTIONS**

    1.  **Adopt a critical investor's perspective.** Focus heavily on the identified risks, red flags, and unanswered questions in the memo. Do not be swayed by positive but unsupported statements.
    2.  **Follow a Chain-of-Thought process.** First, write a brief, step-by-step reasoning. Analyze the severity of the identified risks versus the significance of the strengths.
    3.  **Assign a final score.** After your reasoning, provide a single numerical score from 1 to 10.
    4.  **Format your response EXACTLY as follows, with no additional text or pleasantries:**

    **Reasoning:** [Your step-by-step analysis here. Be concise and focus on the risk/reward balance.]
    **Score:** [A single integer from 1 to 10]

    ---
    **SCORING GUIDE**
    - **1-3 (High Risk / Red Flag):** Significant issues were found. This is a potential deal-breaker that requires immediate resolution.
    - **4-6 (Moderate Risk / Caution):** The findings raise concerns that require serious follow-up. The investment has notable challenges.
    - **7-8 (Positive Outlook / Minor Concerns):** Generally positive findings with minor, manageable risks. Appears to be a solid company.
    - **9-10 (Highly Appealing / Green Light):** Excellent findings, minimal risks, and strong indicators of a healthy, well-run business in this area.

    Begin your analysis now.
"""

Role_Lawyer = "You are a highly skilled and experienced corporate lawyer specializing in financial due diligence. Your task is to analyze the provided financial documents and answer a series of specific questions.\n\n**Your Role:**\n\nYou are to act as a legal and financial expert. Your responses should be:\n\n* **Accurate and Factual:** Base your answers *solely* on the information contained within the provided documents retrieved by the RAG model.\n* **Clear and Concise:** Provide direct and to-the-point answers. Avoid jargon where possible, but use precise legal and financial terminology when necessary.\n* **Professional in Tone:** Maintain a formal and analytical tone at all times.\n* **Cautious and Principled:** If the provided documents do not contain the information needed to answer a question, you must explicitly state that. Do not make assumptions or use any external knowledge.\n\n**Instructions:**\n\n1.  **Analyze the context:** Carefully review the provided financial documents (the \"context\") to understand the company's financial situation.\n2.  **Answer the Question:** For each question, formulate a response based *only* on the information present in the context.\n3.  **Cite Your Source (if applicable):** When you provide a specific data point or piece of information, you can (but are not required to) mention the document or section where you found it.\n4.  **Handle Missing Information:** If the answer to a question cannot be found in the provided context, you MUST respond with: \"The provided documents do not contain sufficient information to answer this question.\"\n\n**Example Interaction:**\n\n**Question:** What was the company's total revenue for the fiscal year 2023?\n\n***(Model analyzes the provided financial statements)***\n\n**Your Answer:** The company's total revenue for the fiscal year 2023 was $15.2 million, as stated in the \"Consolidated Statement of Operations.\"\n\n**Question:** What is the CEO's opinion on the current market trends?\n\n***(Model finds no information about the CEO's opinions in the documents)***\n\n**Your Answer:** The provided documents do not contain sufficient information to answer this question.\n\nYou will now be given a series of questions and the relevant financial documents. Proceed with your analysis." 



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
# ... other imports


# Define the desired data structure for the sentiment output.


def process_questions(processed_data: list[str]):
    """
    Reads questions from a list, gets answers and evaluations,
    and returns a list of result dictionaries.
    """
    new_processed_data = []
    # ✨ 1. DEFINE THE HELPER FUNCTION HERE ✨
    def format_docs(docs):
        """Combines the page_content of a list of Document objects into a single string."""
        return "\n\n".join(doc.page_content for doc in docs)

    # --- Model and Retriever Setup ---
    print(f"🧠 Initializing Groq model: '{GROQ_MODEL_NAME}'...")
    model = ChatGroq(model=GROQ_MODEL_NAME)

    print(f"🧠 Initializing embedding function: '{EMBEDDING_MODEL_NAME}'...")
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print(f"🔍 Initializing vector database at '{CHROMA_PATH_1}'...")
    db = Chroma(persist_directory=CHROMA_PATH_1, embedding_function=embedding_function)
    retriever = db.as_retriever(search_kwargs={'k': 3})
    print("🔍 Vector database initialized.")

    print(f"🔍 Initializing vector database at '{CHROMA_PATH_2}'...")
    db1 = Chroma(persist_directory=CHROMA_PATH_2, embedding_function=embedding_function)
    retriever1 = db1.as_retriever(search_kwargs={'k': 3})
    print("🔍 Vector database initialized.")

    # --- Prompt Templates ---
    print("📝 Creating prompt templates...")
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_RETRIEVER)
    evaluator_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_WHY)
    sentiment_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_SENTIMENT)

    # --- ✨ 2. UPDATE CHAIN DEFINITIONS ✨ ---
    generation_chain = (
        RunnablePassthrough.assign(
            # Pipe the retriever output into format_docs
            context=(lambda x: x["question"]) | retriever | format_docs)
        | RunnablePassthrough.assign(
            answer=(prompt_template | model | StrOutputParser())
        )
    )

    # generation_chain = (
    #     RunnablePassthrough.assign(
    #         # Pipe the retriever output into format_docs
    #         answer=(lambda x: x["question"]) | retriever | format_docs) | prompt_template | model | StrOutputParser()
    # )
    
    evaluate_chain = (
        RunnablePassthrough.assign(
            # Pipe the retriever output into format_docs
            context2=(lambda x: x["question"]) | retriever1 | format_docs
        )
        | RunnablePassthrough.assign(
            answer2=(evaluator_prompt | model | StrOutputParser())
        )
    )

    # evaluate_chain = (
    #     RunnablePassthrough.assign(
    #         # Pipe the retriever output into format_docs
    #         answer2=(lambda x: x["question"]) | retriever1 | format_docs | evaluator_prompt | model | StrOutputParser())
    # )
    
    final_chain = generation_chain | evaluate_chain | RunnablePassthrough.assign(
        sentiment = sentiment_prompt | model | StrOutputParser()
    ) 

    final_chain.get_graph().print_ascii()
    i = 1
    for item in processed_data:
        print(f"\n⏳ Generating response with Groq for  Q{i}...")
        response_dict = final_chain.invoke({"question": item})
        
        print("\n" + "="*50 + "\n✅ Response Dictionary:")
        # Now the 'context' keys will hold simple strings, not Document objects
        print(response_dict) 
        print("="*50 + "\n")
        i += 1
        new_processed_data.append(response_dict)

    return new_processed_data
    
def export_to_json(data: list[str], output_file: str):
    """
    Exports the processed data to a JSON file.
    """
    with open(output_file, 'w') as f:
        # Use indent for pretty-printing the JSON
        json.dump(data, f, indent=2)
        
    print(f"\nProcessing complete. The new data has been saved to {output_file}")  

def main():
    """
    Main function to query the vector database and generate a response using Groq.
    """
    login_to_huggingface() 
    processed_answers = process_questions(financial_due_diligence_questions)
    export_to_json(processed_answers, output_file="temp.json")

    # Write the updated data to the output JSON file
    


if __name__ == "__main__":
    main()