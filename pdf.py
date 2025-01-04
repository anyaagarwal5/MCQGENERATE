import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
import base64
import tempfile
from fpdf import FPDF  # To generate a PDF for saving the summary

# Set up Streamlit page configuration
st.set_page_config(layout="wide", page_title="PDF Summarization App")

# Load model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def summarize(text, max_length=500):
    # Summarize the document with the user-defined max_length
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(inputs, max_length=max_length, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

def summarize_sections(text, section_length=1500, max_length=500):
    """
    Summarize the document in chunks (sections) and concatenate the summaries.
    """
    sections = [text[i:i + section_length] for i in range(0, len(text), section_length)]
    full_summary = ""
    
    for section in sections:
        summary = summarize(section, max_length)
        full_summary += summary + "\n\n"  # Add a space between summaries for clarity
    
    return full_summary

# PDF file preprocessing function with error handling
def file_preprocessing(file):
    try:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.getvalue())  # Use getvalue() for reading the uploaded file content
            temp_file_path = temp_file.name
        
        # Load and split the document using the file path
        loader = PyPDFLoader(temp_file_path)
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter()
        texts = text_splitter.split_documents(pages)
        final_texts = ""
        for text in texts:
            final_texts += text.page_content
        return final_texts
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        return ""

# Summarization pipeline function
def summarize_pdf(uploaded_file, max_length):
    input_text = file_preprocessing(uploaded_file)
    if input_text:
        summary = summarize_sections(input_text, max_length=max_length)
        return summary
    else:
        return "Could not generate summary due to an error."

# Display PDF in the app
def display_pdf(uploaded_file):
    uploaded_file.seek(0)  # Reset file pointer to the start
    content = uploaded_file.read()
    if not content:
        st.error("The uploaded file is empty. Please upload a valid PDF file.")
    else:
        # Display the PDF file
        base64_pdf = base64.b64encode(content).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

# Function to save summary as PDF
def save_summary_as_pdf(summary):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, summary)
    pdf_output = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    pdf.output(pdf_output.name)
    return pdf_output.name

# Main function for Streamlit app
def main():
    st.title('PDF Summarization App')
    st.write("Upload a PDF file, and the system will summarize it.")

    uploaded_file = st.file_uploader("Upload your PDF File", type=["pdf"])

    # Slider for user to select the maximum summary length
    max_summary_length = st.slider(
        "Select maximum summary length (words)",
        min_value=50,
        max_value=1000,
        value=500,  # Default value
        step=50
    )

    if uploaded_file is not None:
        if uploaded_file.size == 0:
            st.error("The uploaded file is empty. Please upload a valid PDF file.")
        else:
            # Display the PDF file
            st.info("Displaying the uploaded PDF:")
            display_pdf(uploaded_file)

            # Summarize the document
            if st.button("Summarize"):
                st.info("Generating Summary...")
                summary = summarize_pdf(uploaded_file, max_summary_length)
                st.write(summary)

                # Provide option to save summary as PDF
                if st.button("Save Summary as PDF"):
                    pdf_file_path = save_summary_as_pdf(summary)
                    st.success("Summary saved as PDF!")
                    with open(pdf_file_path, "rb") as pdf_file:
                        st.download_button(
                            label="Download PDF",
                            data=pdf_file,
                            file_name="summary.pdf",
                            mime="application/pdf"
                        )

if __name__ == "__main__":
    main()
