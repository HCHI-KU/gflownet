from huggingface_hub import login, whoami
from transformers import AutoTokenizer
import os

try:
    # Try with default auth
    print("Checking default auth...")
    user = whoami()
    print(f"Logged in as: {user['name']}")
    
    model_id = "google/gemma-2-9b-it"
    print(f"Attempting to load tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("Successfully loaded tokenizer!")
except Exception as e:
    print(f"Error: {e}")
