import streamlit as st
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

@st.cache_resource
def load_model():
    model_name = "tuner007/pegasus_paraphrase"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

st.title("📝 Shakespearean Translator (Free)")
st.write("Convert modern English into Shakespearean-style old English — fully free, no API key required.")

user_text = st.text_area("Enter your text:")

def translate_to_shakespeare(text):
    batch = tokenizer([text], truncation=True, padding="longest", max_length=60, return_tensors="pt")
    translated = model.generate(**batch, max_length=60, num_beams=5, num_return_sequences=1)
    result = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

    # Enhance Shakespearean style
    replacements = {
        "you": "thou",
        "your": "thy",
        "are": "art",
        "before": "ere",
        "do": "doth",
        "does": "dost",
        "have": "hath",
        "has": "hath",
        "will": "shalt"
    }
    for k, v in replacements.items():
        result = result.replace(" " + k + " ", " " + v + " ")

    return result

if st.button("Translate"):
    if user_text.strip():
        with st.spinner("Translating into Shakespearean English..."):
            shakespeare_text = translate_to_shakespeare(user_text)
        st.subheader("Translated Text:")
        st.write(shakespeare_text)
    else:
        st.warning("Please enter some text.")
