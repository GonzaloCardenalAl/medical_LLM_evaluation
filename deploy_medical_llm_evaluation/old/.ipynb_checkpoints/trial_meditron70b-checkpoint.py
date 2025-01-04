import os 
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_meditron_inference(question, custom_cache_dir):
    
    # Load the tokenizer and model
   # tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("OpenMeditron/Meditron3-70B", cache_dir=custom_cache_dir)

   # model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto")
    model = AutoModelForCausalLM.from_pretrained("OpenMeditron/Meditron3-70B", cache_dir=custom_cache_dir, device_map="auto", torch_dtype="auto")
    # Tokenize the input question
    inputs = tokenizer(question, return_tensors="pt").to(model.device)
    
    # Generate a response
    outputs = model.generate(**inputs, max_new_tokens = 2048)
    
    # Decode the response
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

custom_cache_dir = "/cluster/scratch/gcardenal/LLM_models"

question = "How is HIV diagnosed?"


answer = run_meditron_inference(question, custom_cache_dir)
print(f'User: {question}\nAssistant: {answer}')