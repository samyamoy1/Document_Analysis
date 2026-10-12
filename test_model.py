import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

messages = [
    {
        "role": "user",
        "content": "Explain RAG in simple words in 3 sentences."
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Generating answer...")
outputs = model.generate(**inputs, max_new_tokens=120)

answer = outputs[0][inputs["input_ids"].shape[1]:]
print(tokenizer.decode(answer, skip_special_tokens=True))