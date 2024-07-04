#import streamlit as st
#from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
#import torch

## Load the fine-tuned model and tokenizer
#model_name = "fine-tuned-model"
#model = DistilBertForSequenceClassification.from_pretrained(model_name)
#tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

## Function to classify text
#def classify_text(text):
#    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#    with torch.no_grad():
#        outputs = model(**inputs)
#    logits = outputs.logits
#    predicted_class_id = torch.argmax(logits, dim=1).item()
#    return "spam" if predicted_class_id == 1 else "ham"

## Streamlit app
#st.title("Text Message Classification")
#st.write("Enter a text message and see if it's classified as spam or ham.")

#user_input = st.text_area("Text Message", "")
#if st.button("Classify"):
#    if user_input:
#        prediction = classify_text(user_input)
#        st.write(f"The message is classified as: \n **{prediction}**")
#    else:
#        st.write("Please enter a text message.")
import streamlit as st
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch

# Load the fine-tuned model and tokenizer
model_name = "fine-tuned-model"
model = DistilBertForSequenceClassification.from_pretrained(model_name)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

# Function to classify text
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    return "spam" if predicted_class_id == 1 else "ham"

# Streamlit app
st.set_page_config(page_title="Text Message Classification", page_icon="📧")

# Header
st.title("📧 Text Message Classification")

# Text input area
#st.subheader("Enter a Text Message:")
user_input = st.text_area("Type your message here...", height=50)

# Classify button and result display
if st.button("Classify"):
    if user_input:
        prediction = classify_text(user_input)
        if prediction == "ham":
            st.success(f"The message is classified as: **{prediction}**")
        else:
            st.error(f"The message is classified as: **{prediction}**")
    else:
        st.warning("Please enter a text message.")

# Footer
st.markdown("""
    ---
    Built with ❤️ using [Streamlit](https://streamlit.io/) and [Transformers](https://huggingface.co/transformers/).
    """)
