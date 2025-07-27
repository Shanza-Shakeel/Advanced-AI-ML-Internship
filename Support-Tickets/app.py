import streamlit as st
from transformers import pipeline
import re
from collections import defaultdict

# --- APP CONFIG ---
st.set_page_config(
    page_title="Ticket Analyzer Pro",
    page_icon="🏷️",
    layout="wide"
)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        return pipeline(
            "zero-shot-classification",
            model="typeform/distilbert-base-uncased-mnli",
            device=-1
        )
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

classifier = load_model()

# --- TAGGING SYSTEM ---
DEFAULT_TAGS = {
    "Software Bug": ["crash", "error", "not working", "bug", "freeze"],
    "Performance Issue": ["slow", "lag", "unresponsive", "timeout"],
    "Login Problem": ["can't log in", "login failed", "password", "authentication"],
    "Payment Issue": ["charge", "payment", "invoice", "refund", "$"],
    "Data Problem": ["missing data", "deleted", "lost", "corrupted"],
    "Feature Request": ["how to", "can you add", "feature request"]
}

SOLUTIONS = {
    "Software Bug": [
        "1. Restart the application",
        "2. Check for updates",
        "3. Clear cache/data",
        "4. Collect error logs",
        "5. Escalate to engineering team"
    ],
    "Performance Issue": [
        "1. Check internet connection",
        "2. Try during non-peak hours",
        "3. Disable browser extensions"
    ],
    "Login Problem": [
        "1. Reset password",
        "2. Check account status",
        "3. Verify email/phone"
    ],
    "Payment Issue": [
        "1. Verify transaction ID",
        "2. Check payment processor",
        "3. Process refund if needed"
    ],
    "General": [
        "1. Acknowledge ticket receipt",
        "2. Gather more details",
        "3. Route to appropriate team"
    ]
}

# --- UTILITY FUNCTIONS ---
def clean_text(text):
    text = str(text).lower()
    return re.sub(r'\{.*?\}|[^\w\s]', ' ', text).strip()

def analyze_ticket(text):
    text = clean_text(text)
    urgent = any(word in text for word in ["urgent", "immediately", "critical", "emergency"])
    
    try:
        if classifier:
            model_result = classifier(text, list(DEFAULT_TAGS.keys()), multi_label=True)
            primary_tag = model_result['labels'][0]
            confidence = model_result['scores'][0]
            
            # Only use model prediction if confident
            if confidence > 0.7:
                return {
                    "primary_tag": primary_tag,
                    "solutions": SOLUTIONS.get(primary_tag, SOLUTIONS["General"]),
                    "urgent": urgent
                }
        
        # Rule-based fallback
        scores = {tag: 0 for tag in DEFAULT_TAGS}
        for tag, keywords in DEFAULT_TAGS.items():
            for keyword in keywords:
                if keyword in text:
                    scores[tag] += 1
        
        primary_tag = max(scores.items(), key=lambda x: x[1])[0]
        return {
            "primary_tag": primary_tag,
            "solutions": SOLUTIONS.get(primary_tag, SOLUTIONS["General"]),
            "urgent": urgent
        }
    
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return {
            "primary_tag": "General",
            "solutions": SOLUTIONS["General"],
            "urgent": urgent
        }

# --- MAIN APP ---
st.title("🏷️ Ticket Analyzer Pro")

ticket_text = st.text_area(
    "Enter ticket description:",
    height=150,
    placeholder="Example: The app crashes when uploading PDF files...",
    key="ticket_input"
)

if st.button("Analyze Ticket", type="primary"):
    if not ticket_text.strip():
        st.error("Please enter ticket content")
    else:
        with st.spinner("Analyzing..."):
            analysis = analyze_ticket(ticket_text)
            
            # Display results
            if analysis["urgent"]:
                st.error(f"🚨 URGENT: {analysis['primary_tag']}")
            else:
                st.success(f"🏷️ {analysis['primary_tag']}")
            
            st.markdown("**Recommended Actions:**")
            for step in analysis["solutions"]:
                st.write(step)

st.caption("Ticket Analyzer Pro v2.3 | Model: DistilBERT-MNLI")