import streamlit as st
from murder import page1
from spetial import page2

# Initialize the session state for the current page
if "page" not in st.session_state:
    st.session_state.page = "main"

def set_background_gradient(start_color, end_color):
  """Sets the background gradient for the Streamlit app.

  Args:
      start_color (str): The starting color for the gradient. (e.g., "#FF0000")
      end_color (str): The ending color for the gradient. (e.g., "#0000FF")
  """
  st.markdown(f"""<style>
      .reportview-container {{
          background-image: linear-gradient(to right, {start_color}, {end_color});
      }}
  </style>""", unsafe_allow_html=True)

# Main page content
def main():
    st.title("Crime in India")

    # Set background gradient for the main page
    set_background_gradient("#000000", "#111111")  # Replace with your desired colors

    st.write("Welcome to the Crime in India Portal!")
    st.write("Explore different facets of crime in India, focusing on murder and special crimes.")

    # Buttons to navigate to other pages
    if st.button("Explore Murder Cases"):
        st.session_state.page = "page1"
    if st.button("Discover Special Crimes"):
        st.session_state.page = "page2"

# Check the current page and display the appropriate content
if st.session_state.page == "main":
    main()
elif st.session_state.page == "page1":
    page1()
elif st.session_state.page == "page2":
    page2()