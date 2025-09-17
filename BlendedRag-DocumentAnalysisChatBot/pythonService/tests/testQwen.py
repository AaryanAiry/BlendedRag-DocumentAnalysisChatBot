from llama_cpp import Llama

# Correct path to your downloaded GGUF file
model_path = "models/qwen2.5-3b-instruct-q5_k_m.gguf"

# Load model
llm = Llama(model_path=model_path)

# Simple prompt test
prompt = "Explain in 2 sentences what quantum computing is."
output = llm(prompt=prompt, max_tokens=100)

print(output)
