import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

prompt = "The factory production line optimization requires"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    # Prefill
    t0 = time.time()
    outputs = model(**inputs, use_cache=True)
    past = outputs.past_key_values
#
# past_key_values = [
#   (layer0_key, layer0_value),
#   (layer1_key, layer1_value),
#   ...
# ]  K, V shape:[batch_size, num_heads, seq_len, head_dim]

    t1 = time.time()

    # Decode 1 token
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

    t2 = time.time()
    # 这里模型化看到的Attention输入是 Q：当前一个token。KV，prompt的所有token（来自past_key_values） + 当前token
    outputs2 = model(input_ids=next_token,
                     past_key_values=past,
                     use_cache=True)
    t3 = time.time()

print(f"Prefill time: {(t1 - t0) * 1000:.2f} ms")
print(f"Decode 1 token time: {(t3 - t2) * 1000:.2f} ms")
