import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(device)


summarization_prompt = """
### Instruction:
You are a state-of-the-art language model specifically trained to create clear and concise summaries. Your task is to distill the provided text into a summary that highlights the essential information while maintaining accuracy and coherence. Follow these updated guidelines:

1. **Read and Understand**: Thoroughly comprehend the main ideas, context, and purpose of the text.
2. **Extract Essentials**: Identify key details such as primary events, core arguments, critical facts, and supporting information.
3. **Summarize with Precision**:
   - Be Concise: Eliminate superfluous details, redundancies, and minor examples.
   - Stay Accurate: Ensure the summary faithfully represents the source without introducing errors.
4. **Adapt Language for Clarity**: Write in clear, logical, and concise sentences while maintaining readability.
5. **Paraphrase Thoughtfully**: Rephrase ideas in your own words, avoiding direct quotes unless absolutely necessary.
6. **Maintain Neutrality**: Avoid subjective interpretations or opinions, presenting only factual information.

### Input:
{text}

### Response:
"""

class inf_model(object):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct-bnb-4bit")
        self.model = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B-Instruct-bnb-4bit")
    
    def llm_summarize(self, transcript):

        

        inputs = self.tokenizer(
            [
                summarization_prompt.format(
                    text= transcript, 
                )
            ], return_tensors="pt").to(device) 

        outputs = self.model.generate(
            **inputs,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=2048,
            no_repeat_ngram_size=3,  
            temperature=0.7,  
            top_p=0.9  
  
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)        
        
        
    