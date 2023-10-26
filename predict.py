# Prediction interface for Cog

from cog import BasePredictor, Input, Path
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "amazon/MistralLite"
MODEL_CACHE = "model-cache"
TOKEN_CACHE = "token-cache"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=TOKEN_CACHE,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            use_flash_attention_2=True,
            device_map="auto",
            cache_dir=MODEL_CACHE
        )
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
        )

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="What are the main challenges to support a long context for LLM?"
        ),
        max_new_tokens: int = Input(
            description="Max new tokens",
            default=400,
            le=16000,
            ge=0.0,
        ),
    ) -> str:
        """Run a single prediction on the model"""
        full_prompt = "<|prompter|>"+prompt+"</s><|assistant|>"
        sequence = self.pipeline(
            full_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_full_text=False,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
        )[0]
        return sequence['generated_text']
        
