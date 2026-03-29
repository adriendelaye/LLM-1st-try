# generator.py
from prompting import build_context, build_prompt
from huggingface_hub import InferenceClient

class LLMGenerator:
    def __init__(self, model="mistral", max_tokens=300, temperature=0.3, hf_token=None):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        if not hf_token:
            raise ValueError("hf_token needed")
        HF_TOKEN = "" #add your Hungging face token in the strings
        self.client = InferenceClient(token=HF_TOKEN)
        self._cache = {}

    def generate(self, prompts):
        results = []
        prompts_to_compute = []
        indices_to_compute = []

        # Verify cache
        for i, prompt in enumerate(prompts):
            if prompt in self._cache:
                results.append(self._cache[prompt])
            else:
                results.append(None)
                prompts_to_compute.append(prompt)
                indices_to_compute.append(i)

        # calls HF API
        for i, prompt in zip(indices_to_compute, prompts_to_compute):
            try:
                output = self.client.text_generation(
                    model=self.model,
                    prompt=prompt,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                text = output.generated_text.strip()
            except Exception as e:
                print(f"[ERROR] LLM generation failed for prompt: {e}")
                text = "[ERROR] LLM generation failed."

            results[i] = text
            self._cache[prompt] = text

        return results if len(results) > 1 else results[0]

    def answer(self, query: str, docs: list):
        context = build_context(docs)
        prompt = build_prompt(query, context)
        return self.generate([prompt])
