import streamlit as st
import awesome_streamlit as ast
import home
import data1 
import rawplot
import pred
import postplots
import insights
import uploader
ast.core.services.other.set_logging_format()

# create the pages
PAGES = {
    "Home":home,
    "Raw Data":data1,
    "Raw Data visualisations":rawplot,
    "Run Predictions":pred,
    "Predicted data +  visualisations":postplots,
    "Insights":insights,
    "File uploader": uploader
}

# render the pages
def main():
    """Main function of the App"""
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]

    with st.spinner(f"Loading {selection} ..."):
        ast.shared.components.write_page(page)

    st.sidebar.title("About")
    st.sidebar.info(
        """
        This App is an end-to-end Web_app that transcribes Amharic speech.
        """
    )

# run it
if __name__ == "__main__":
    main()
    