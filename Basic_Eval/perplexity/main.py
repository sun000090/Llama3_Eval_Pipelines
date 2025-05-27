import torch
import json
from perplexityCal import perplexityCalculator

model_name = 'task-aware/Llama_3.2_1B_Instruct'

device = 'mps' if torch.backends.mps.is_built()==True else 'cpu'

with open('mtbench101.jsonl') as f:
    data = [json.loads(line) for line in f]

print(perplexityCalculator(model_name,
                           device,
                           data[3]['history'][0]['user']).scores())