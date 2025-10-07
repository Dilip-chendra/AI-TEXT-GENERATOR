import os
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
import numpy as np
from utils import clean_text, postprocess_generated_text

# Set page config
st.set_page_config(
    page_title="AI Text Generator",
    page_icon="✍️",
    layout="centered"
)

# Initialize models
@st.cache_resource
def load_models():
    try:
        # Show loading message
        with st.spinner("Loading AI models (this may take a few minutes on first run)..."):
            # Sentiment analysis model (smaller model for better performance)
            sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            
            # Load with progress
            progress_text = st.empty()
            progress_text.text("Downloading sentiment analysis model...")
            
            sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
            sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
            
            progress_text.text("Downloading text generation model...")
            
            # Use a smaller model for text generation
            generator = pipeline('text-generation', 
                               model='gpt2',  # Using base GPT-2 model for better compatibility
                               device=-1)  # Force CPU usage
            
            progress_text.empty()
            
            return {
                'sentiment_tokenizer': sentiment_tokenizer,
                'sentiment_model': sentiment_model,
                'generator': generator
            }
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.warning("Please check your internet connection and try again.")
        return None

def analyze_sentiment(text, models):
    inputs = models['sentiment_tokenizer'](text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = models['sentiment_model'](**inputs).logits
    
    # Get predicted class (0 = negative, 1 = positive)
    predicted_class = torch.argmax(logits).item()
    sentiment = "positive" if predicted_class == 1 else "negative"
    
    # Get probability scores
    probs = torch.softmax(logits, dim=1)
    confidence = probs[0][predicted_class].item()
    
    return sentiment, confidence

def generate_text(prompt, sentiment, models, max_length=100):
    # Add sentiment guidance to the prompt
    prompt_with_sentiment = f"{prompt} [This text should be {sentiment} sentiment]"
    
    # Clean the prompt
    clean_prompt = clean_text(prompt_with_sentiment)
    
    # Generate text using the model
    try:
        with st.spinner("Generating text..."):
            generated = models['generator'](
                clean_prompt,
                max_length=min(max_length, 200),  # Limit max length
                num_return_sequences=1,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                no_repeat_ngram_size=2,
                do_sample=True,
                pad_token_id=models['generator'].tokenizer.eos_token_id,
                truncation=True
            )
            
            # Post-process the generated text
            if generated and len(generated) > 0:
                processed_text = postprocess_generated_text(generated[0]['generated_text'])
                return processed_text or "I couldn't generate any text. Please try again with a different prompt."
            return "I couldn't generate any text. Please try again with a different prompt."
            
    except Exception as e:
        st.error(f"An error occurred during text generation: {str(e)}")
        return "I'm sorry, I couldn't generate text at the moment. Please try again with a different prompt."

def main():
    st.title("✍️ AI Text Generator with Sentiment")
    st.markdown("""
    This AI-powered tool generates text based on the sentiment of your input. 
    - Enter any text prompt
    - The AI will detect the sentiment (positive/negative)
    - It will then generate a coherent paragraph matching that sentiment
    - Or manually select a sentiment to override the detection
    """)
    
    # Add a note about first-time setup
    with st.expander("⚠️ First time setup"):
        st.write("""
        On first run, the application needs to download AI models (about 1-2GB total). 
        This may take several minutes depending on your internet speed.
        Please ensure you have a stable internet connection.
        """)
    
    # Load models
    models = load_models()
    
    if models is None:
        st.error("Failed to load AI models. Please check your internet connection and try again.")
        if st.button("Retry loading models"):
            st.experimental_rerun()
        return
    
    # User input with example prompts (simpler examples)
    example_prompts = [
        "A beautiful sunny day",
        "I didn't like the food",
        "This is amazing!",
        "That was really bad"
    ]
    
    col1, col2 = st.columns([3, 1])
    with col1:
        user_input = st.text_area("Enter your prompt:", "I had a wonderful day at the park")
    with col2:
        st.write("Try these examples:")
        for i, example in enumerate(example_prompts):
            if st.button(f"Example {i+1}", key=f"example_{i}"):
                user_input = example
    
    # Optional parameters
    col1, col2 = st.columns(2)
    with col1:
        length = st.slider("Output length (words)", 50, 300, 100)
    with col2:
        manual_sentiment = st.radio("Or choose sentiment:", ["Auto-detect", "Positive", "Negative"])
    
    if st.button("Generate Text"):
        if not user_input.strip():
            st.warning("Please enter some text first!")
            return
            
        with st.spinner("Analyzing sentiment and generating text..."):
            # Determine sentiment
            if manual_sentiment == "Auto-detect":
                sentiment, confidence = analyze_sentiment(user_input, models)
                st.success(f"Detected sentiment: {sentiment.capitalize()} (Confidence: {confidence*100:.1f}%)")
            else:
                sentiment = manual_sentiment.lower()
                
            # Generate text
            generated_text = generate_text(
                user_input,
                sentiment,
                models,
                max_length=length
            )
            
            # Display results with better formatting
            st.subheader("Generated Text:")
            st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; color: black;'>
                <p style='font-size: 16px; line-height: 1.6; color: black;'>{generated_text}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add copy to clipboard button using Streamlit
            if st.button("Copy to Clipboard"):
                import pyperclip
                try:
                    pyperclip.copy(generated_text)
                    st.success("Copied to clipboard!")
                except Exception as e:
                    st.error(f"Failed to copy to clipboard: {str(e)}")
            
            # Add some spacing
            st.markdown("---")
            st.info("💡 Tip: Try different prompts or adjust the sentiment manually for varied results!")

if __name__ == "__main__":
    main()
