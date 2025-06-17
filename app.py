import streamlit as st
from backendd import process_user_input
from backendd import TextCleaner
st.title("News Authenticity & Fact-Check Demo")

user_input = st.text_area("Paste a news article here:")

if st.button("Analyze"):
    if user_input.strip():
        # Call the process_user_input function defined in your notebook
        result = process_user_input(user_input)
        
        st.write(f"**Prediction:** {result['prediction']}")
        st.write("**Extracted Claims:**")
        for claim in result['claims']:
            st.write(f"- {claim}")
        st.write("**Fact Checks:**")
        for check in result['fact_checks']:
            st.write(f"**Claim:** {check['claim']}")
            for fact in check['fact_check']:
                st.write(f"  - Verdict: {fact.get('textualRating', 'N/A')}")
    else:
        st.warning("Please paste a news article before clicking Analyze.")
