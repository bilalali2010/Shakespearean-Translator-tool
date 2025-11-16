import streamlit as st
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
import re

@st.cache_resource
def load_model():
    model_name = "tuner007/pegasus_paraphrase"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

st.title("📝 Shakespearean Translator (Free)")
st.write("Convert modern English into Shakespearean-style English — fully free, no API key required.")

user_text = st.text_area("Enter your text:")

# ----------------------------------------
# SHAKESPEAREAN POST-PROCESSING FUNCTION
# ----------------------------------------

def shakespeare_enhance(text):
    replacements = {
        r"\byou\b": "thee",
        r"\byour\b": "thy",
        r"\byours\b": "thine",
        r"\bare\b": "art",
        r"\bbefore\b": "ere",
        r"\bdo\b": "doth",
        r"\bdoes\b": "doth",
        r"\bhave\b": "hath",
        r"\bhas\b": "hath",
        r"\bwill\b": "shall"
    }

    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    # Optional poetic enhancements
    text = text.replace(" very ", " most ")
    text = text.replace(" really ", " truly ")

    return text


# ----------------------------------------
# PEGASUS PARAPHRASE + STYLE MIX FUNCTION
# ----------------------------------------

def translate_to_shakespeare(text):
    batch = tokenizer(
        [text],
        truncation=True,
        padding="longest",
        max_length=128,
        return_tensors="pt"
    )

    generated = model.generate(
        **batch,
        max_length=128,
        num_beams=5,
        num_return_sequences=1,
        temperature=1.1
    )

    paraphrased = tokenizer.decode(generated[0], skip_special_tokens=True)

    # Apply Shakespearean enhancement
    final_text = shakespeare_enhance(paraphrased)

    # Capitalize first letter
    return final_text[0].upper() + final_text[1:]


# ----------------------------------------
# STREAMLIT UI
# ----------------------------------------

if st.button("Translate"):
    if user_text.strip():
        with st.spinner("Translating into Shakespearean English..."):
            output = translate_to_shakespeare(user_text)
        st.subheader("Translated Text:")
        st.write(output)
    else:
        st.warning("Please enter some text.")
