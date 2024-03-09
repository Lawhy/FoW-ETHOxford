from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class ChatAgent:

    def __init__(self, pretrained: str = "google/gemma-7b", load_in_4bit: bool = False, load_in_8bit: bool = False):
        self.pretrained_path = pretrained
        self.quantization_config = BitsAndBytesConfig(load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit)
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.pretrained_path, quantization_config=self.quantization_config
        )

    def response(self, query: str, max_new_tokens: int = 32):
        input_ids = self.tokenizer(query, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**input_ids, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0])
    
    def parse(self, query: str, max_new_tokens: int = 32):
        ...
