import re

def clean_text(text):
    """Clean and preprocess the input text."""
    if not text:
        return ""
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove special characters and numbers (keep basic punctuation)
    text = re.sub(r'[^\w\s\.\!\?\,\']', ' ', text)
    return text.strip()

def format_generated_text(text):
    """Format the generated text to improve readability."""
    if not text:
        return ""
    # Simple sentence splitting on punctuation followed by space
    sentences = re.split('([.!?] )', text)
    # Recombine sentences with proper spacing
    text = ''
    for i in range(0, len(sentences)-1, 2):
        if i+1 < len(sentences):
            text += sentences[i] + sentences[i+1]
        else:
            text += sentences[i]
    # Capitalize first letter
    return text[0].upper() + text[1:] if text else ""

def postprocess_generated_text(text):
    """Post-process the generated text to remove any artifacts."""
    if not text:
        return ""
    # Remove the sentiment instruction if it appears in the output
    text = re.sub(r'\[This text should be (positive|negative) sentiment\]', '', text, flags=re.IGNORECASE)
    # Clean up any remaining artifacts
    text = re.sub(r'\s+', ' ', text).strip()
    # Ensure proper sentence capitalization
    return format_generated_text(text)
