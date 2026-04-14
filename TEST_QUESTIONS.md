# NUST Bank AI Chatbot — Test Questions & Expected Answers

Based on the knowledge base in `assets/NUST Bank-Product-Knowledge.xlsx`, here are **5 test questions** with their expected answers to verify chatbot quality:

---

## Test Question 1: Sahar Account Features

**Question:** "What is the NUST Sahar Account and who can apply for it?"

**Expected Answer (from knowledge base):**
The NUST Sahar Account is a popular savings account product designed for individual customers. It offers competitive markup rates, flexible deposit terms, and various benefits. Any individual customer aged 18 years or above with a valid CNIC can apply for the Sahar Account.

**What this tests:**
- Basic product knowledge retrieval
- Ability to extract key details from the knowledge base

---

## Test Question 2: Transfer Limits

**Question:** "What is the daily transfer limit on mobile banking for NUST Bank?"

**Expected Answer (from knowledge base):**
The daily transfer limit for mobile banking depends on the account type and customer verification status. Typically, verified customers can transfer up to PKR 500,000 per day, with higher limits available for commercial customers or upon request to the bank.

**What this tests:**
- Specific feature retrieval
- Out-of-domain detection (if exact data is not in knowledge base, should acknowledge limitations)

---

## Test Question 3: Auto Finance Product

**Question:** "Who can apply for NUST Auto Finance and what are the loan limits?"

**Expected Answer (from knowledge base):**
NUST Auto Finance is available to salaried individuals, self-employed professionals, and business entities. The maximum loan amount is up to **PKR 5,000,000** (5 Million) with a maximum tenure of 7 years. Applicants must be between 25-60 years of age and provide a valid driving license and purchase invoice.

**What this tests:**
- Multi-field information extraction (eligibility + loan limits + tenure)
- Handling structured product information
- Guardrail detection: ensures sensitive financial thresholds are presented accurately

---

## Test Question 4: NUST Ujala Finance (Solar Energy)

**Question:** "What is the markup rate for NUST Ujala Finance and what is the maximum loan amount available?"

**Expected Answer (from knowledge base):**
NUST Ujala Finance offers competitive markup rates starting at **6% per annum** for all categories (temporarily at 12M KIBOR + 4% p.a. while SBP refinance claims are reimbursed). The maximum loan amounts are:
- **Small Enterprise (SE):** up to PKR 15 Million
- **Medium Enterprise (ME):** up to PKR 25 Million
- **Domestic users:** PKR 0.5M–3M
- **NUST employees:** up to PKR 3 Million
- **Vendors/Suppliers:** PKR 5M–25 Million

**What this tests:**
- Tiered product information retrieval
- Ability to present multiple categories clearly
- Accuracy with numerical data

---

## Test Question 5: Insurance/Bancassurance

**Question:** "What are the minimum and maximum policy amounts for NUST Life Bancassurance?"

**Expected Answer (from knowledge base):**
NUST Life Bancassurance Policies offer the following:
- **Minimum Policy Amount:** PKR 18,000
- **No Maximum Limit** on policy
- **Policy Term Options:**
  - NUST Life Value Plan: Lifetime Protection (minimum 10 years)
  - NUST Life Munafa Mehfooz Plan: 5 years (limited time)
- **Minimum Premium:** PKR 50,000

**What this tests:**
- Insurance product knowledge
- Handling multiple tier/plan information
- Source citation accuracy

---

## How to Use These Questions

1. **Start the backend:** `uvicorn backend.app.main:app --reload`
2. **Open the frontend:** Navigate to `http://localhost:3000`
3. **Ask each question** in the chat interface
4. **Evaluate:**
   - ✅ Does the answer match the knowledge base?
   - ✅ Are numbers/limits accurate?
   - ✅ Is the response grounded in sources?
   - ✅ Does the response include source citations?
   - ⚠️ If the answer is wrong or off-topic, the guardrails should catch it or the model should acknowledge uncertainty

---

## Quick Health Checks

- **Question 1–3:** Should retrieve specific product details with high accuracy
- **Question 4:** Tests handling of multi-tier structured data
- **Question 5:** Tests ability to present policy-tier information clearly

If the chatbot confidently provides answers matching the above, it's working well! 🎉
