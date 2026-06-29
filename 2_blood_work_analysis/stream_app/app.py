import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Blood Work Analyzer",
    page_icon="🩸",
    layout="wide"
)

st.title("🩸 Blood Work Analysis")
st.caption("AI Powered Blood Report Analyzer & Indian Nutritionist")

# -----------------------------
# LLM
# -----------------------------
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    reasoning_format="parsed"
)

# -----------------------------
# Upload File
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Blood Report (.txt)",
    type=["txt"]
)

if uploaded_file:

    blood_report = uploaded_file.read().decode("utf-8")

    st.success("Blood report uploaded successfully.")

    with st.expander("View Blood Report"):
        st.text(blood_report)

    if st.button("Analyze Report", use_container_width=True):

        with st.spinner("Analyzing Blood Report..."):

            # -----------------------------
            # Stage 1
            # -----------------------------
            extraction_prompt = f"""
You are a medical data extraction assistant.

From the blood report below, extract all the test values and classify each one as HIGH, LOW, or NORMAL based on the reference ranges provided in the report.

Format your response as:

- Test Name : Value | Status: HIGH/LOW/NORMAL | Reference: Range

Blood Report:
{blood_report}
"""

            extraction_response = llm.invoke(extraction_prompt)
            extracted_values = extraction_response.content

            # -----------------------------
            # Stage 2
            # -----------------------------
            diet_prompt = f"""
You are an Indian nutritionist.

Based on the following Blood Work Analysis:

{extracted_values}

Provide only:

1. Short Health Summary (4-5 lines)

2. Diet Recommendations

- What to Avoid
- What to Eat More

Do not include meal plans, recipes, calories, exercise, or any other advice.
"""

            diet_response = llm.invoke(diet_prompt)

        # -----------------------------
        # Output
        # -----------------------------
        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🧪 Extracted Blood Values")
            st.code(extracted_values)

        with col2:
            st.subheader("🥗 Health Summary & Diet")
            st.markdown(diet_response.content)