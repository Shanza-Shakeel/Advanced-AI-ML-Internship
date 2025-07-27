\# 🏷️ Support Ticket Tagging System



\## 🎯 Objective of the Task

The objective of this project is to build an intelligent system capable of automatically tagging support tickets based on their descriptions. The system aims to:



\- 🤖 Automate the tagging process for support tickets

\- ⚡ Improve efficiency in categorizing and routing tickets

\- 📊 Evaluate model performance using accuracy, precision, recall, and F1-score



\## 🔍 Methodology / Approach



\### 1. 📂 Dataset

\*\*Source\*\*:  

\- Kaggle dataset containing support tickets with:

&nbsp; - 📝 Ticket descriptions

&nbsp; - 🏷️ Predefined categories (ground truth)



\*\*Preprocessing\*\*:

\- 🧹 Cleaned placeholder values (e.g., `{product\_purchased}`)

\- ✂️ Split into training/testing sets (80/20 ratio)



\### 2. 🤖 Zero-Shot Classification Model

\*\*Model\*\*: `facebook/bart-large-mnli` from Hugging Face  

\*\*Candidate Labels\*\*:

\- 🛠️ Technical Issue

\- 💰 Billing Inquiry

\- 🔑 Account Access

\- 💾 Data Loss



\*\*Prediction Process\*\*:

```python

\# Example prediction

Input: "I forgot my password"

Output: \["Account Access", "Technical Issue", "Password Reset"]

