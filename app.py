import streamlit as st
from mfapi import MFAPI

# Page config
st.set_page_config(
    page_title="MF Return Decomposition Analyzer",
    page_icon="📊",
    layout="wide"
)

# Initialize
mf = MFAPI()

# Title
st.title("Return Decomposition Analysis")

# Sidebar
with st.sidebar:
    st.header("Fund Selection")
    
    # Get schemes
    with st.spinner("Loading schemes..."):
        scheme_codes = mf.get_scheme_codes()
    
    if scheme_codes:
        scheme_names = {v: k for k, v in scheme_codes.items()}
        
        # Search
        search_term = st.text_input("Search fund name", "")
        
        if search_term:
            filtered_names = [name for name in scheme_names.keys() 
                            if search_term.lower() in name.lower()]
        else:
            filtered_names = list(scheme_names.keys())[:100]  # Show first 100
        
        # Select scheme
        if filtered_names:
            selected_name = st.selectbox(
                "Select Fund",
                options=filtered_names,
                key="fund_select"
            )
            
            if selected_name:
                scheme_code = scheme_names[selected_name]
                
                # Show basic details
                details = mf.get_scheme_details(scheme_code)
                if details:
                    st.divider()
                    st.subheader("Fund Details")
                    st.write(f"**Fund House:** {details['fund_house']}")
                    st.write(f"**Category:** {details['scheme_category']}")
                    st.write(f"**Latest NAV:** ₹{details['nav']}")
        else:
            st.warning("No funds found. Try different search term.")
    else:
        st.error("Failed to load schemes")

# Main area
st.info("👈 Select a fund from the sidebar to begin analysis")