import os 
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_llama_inference(question, custom_cache_dir):
    
    # Load the tokenizer and model
   # tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct", cache_dir=custom_cache_dir)

   # model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-70B-Instruct", cache_dir=custom_cache_dir, device_map="auto", torch_dtype="auto")
    # Tokenize the input question
    inputs = tokenizer(question, return_tensors="pt").to(model.device)
    
    # Generate a response
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, top_p=0.9)
    
    # Decode the response
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

custom_cache_dir = "/cluster/scratch/gcardenal/LLM_models"

#model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", cache_dir=custom_cache_dir, device_map="auto", torch_dtype="auto")

#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B", cache_dir=custom_cache_dir)


question = "How is HIV diagnosed?"


answer = run_llama_inference(question, custom_cache_dir)
print(f'User: {question}\nAssistant: {answer}')