from llama_cpp import Llama

# Path to your local model (same as in llmClient.py)
MODEL_PATH = "models/qwen2.5-3b-instruct-q5_k_m.gguf"

# Initialize Qwen model
llm = Llama(model_path=MODEL_PATH)

# Questions to test (you can add more here)
questions = [
    "What is the time complexity of BogoSquared Sort?",
    "How many times was the word 'algorithm' misspelled in the document?",
    "Which algorithm involves deleting universes as part of its process?",
    "What is Yolo Sort?",
    "Explain Stalin Sort briefly."
]

def ask_qwen(question: str, max_tokens: int = 256):
    print(f"\n=== Question: {question} ===")
    try:
        output = llm(
            prompt=f"Answer the following question briefly:\n{question}\n",
            max_tokens=max_tokens,
            temperature=0.7
        )
        answer = output["choices"][0]["text"].strip()
        print(f"Answer: {answer}")
    except Exception as e:
        print(f"Error generating answer: {e}")

# Run all questions
for q in questions:
    ask_qwen(q)
