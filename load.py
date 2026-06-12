import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

device = "mps" if torch.backends.mps.is_available() else "cpu"

model_path = "./fine_tuned_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

model.eval()

def generate_response(input_text):
    prompt = f"Question: {input_text}\nAnswer:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=128, num_return_sequences=1)

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response.split("Answer:")[1].strip() if "Answer:" in response else response.strip()

# Example usage
if __name__ == "__main__":
    user_input = "tell me more about pcc"
    response = generate_response(user_input)
    print(f"User: {user_input}")
    print(f"Model: {response}")