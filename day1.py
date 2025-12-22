import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2"  # 小模型，够用
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

model.eval()

text = "The factory production line"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

print("input_ids shape:", inputs["input_ids"].shape)
print("logits shape:", outputs.logits.shape)
print("hidden_states layers:", len(outputs.hidden_states))
print("last hidden state shape:", outputs.hidden_states[-1].shape)
