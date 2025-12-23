import time
import threading
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model.eval()

def hf_infer(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=32)

texts = ["The factory production line optimization requires"] * 4

t0 = time.time()
threads = []

for t in texts:
    th = threading.Thread(target=hf_infer, args=(t,))
    th.start()
    threads.append(th)

for th in threads:
    th.join()

print("HF total time:", time.time() - t0)

from vllm import LLM, SamplingParams
import time

llm = LLM(model="gpt2")
sampling_params = SamplingParams(max_tokens=32)

prompts = ["The factory production line optimization requires"] * 4

t0 = time.time()
llm.generate(prompts, sampling_params)
print("vLLM total time:", time.time() - t0)

