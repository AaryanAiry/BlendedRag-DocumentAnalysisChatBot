# app/llm/llmClient.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from app.utils.logger import getLogger

logger = getLogger(__name__)

class LLMClient:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct"):
        """
        Initializes the LLM client.
        Automatically detects GPU if available, otherwise falls back to CPU.
        """
        self.model_name = model_name

        # Check device
        self.device = 0 if torch.cuda.is_available() else -1
        if self.device == 0:
            logger.info("GPU detected. Using GPU for inference.")
        else:
            logger.info("No GPU detected. Using CPU for inference. Performance will be slower.")

        # Load tokenizer
        logger.info(f"Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model
        logger.info(f"Loading model {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if self.device == 0 else None,
            torch_dtype=torch.float16 if self.device == 0 else torch.float32
        )

        # Initialize text-generation pipeline
        logger.info("Initializing text-generation pipeline...")
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )
        logger.info("LLMClient ready.")

    def generateAnswer(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        """
        Generate an answer for the given prompt.
        """
        try:
            response = self.generator(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True
            )
            return response[0]['generated_text']
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return "Error: Failed to generate answer."

# Singleton instance for reuse
llmClient = LLMClient()
