import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class perplexityCalculator:
    def __init__(self,model_name, device, data):
        self.model_name = model_name
        self.device = device
        self.data = data

    def compute_perplexity(self, log_probs):
        total_log_prob = 0
        for log_prob in log_probs:
            total_log_prob += log_prob
        perplexity = np.exp(-total_log_prob / len(log_probs))
        return perplexity

    def scores(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, 
                                                     torch_dtype=torch.float16).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        prompt = f'''Given the question provide answer in a detailed sentence.
                    Question: {self.data}
                    Answer:
                    '''
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
        output_ids = model.generate(
            **input_ids.to(model.device),
            max_new_tokens=256,
            do_sample=True,
            temperature=0.5,
            output_scores=True,
            return_dict_in_generate=True
            )
        
        transition_scores = model.compute_transition_scores(
            output_ids.sequences, output_ids.scores, normalize_logits=True
        )
        input_length = input_ids.input_ids.shape[1]
        generated_tokens = output_ids.sequences[:, input_length:]
        output = tokenizer.decode(generated_tokens.tolist()[0][:-1])

        return {'outputs':output,
                'perplexity':self.compute_perplexity(transition_scores[0].cpu().numpy())}