import os
import json
import openai
from openai import OpenAI
import pandas as pd
import re
import csv
import numpy as np
import anthropic
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
#from functions_nvllm import split_model


# Set your API keys and base URLs / LLAMA API NOT WORKING, CHANGED TO CLUSTER INFERENCE
#llama_api_key = "sk-rc-PcaiUevKWIDWowLh29xBdg"
#llama_base_url = "http://148.187.108.173:8080"

claude_api_key = "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"

#meditron_api_key = 
#meditron_url = 

#client_openai = OpenAI(api_key=llama_api_key , base_url=llama_base_url)
client_claude = anthropic.Anthropic(api_key=claude_api_key)
#client_meditron = 

# Set directories
input_dir = "./questions_files/"    # Directory containing the 6 input JSON files
output_dir = "./model_answers/"      # Directory to save output JSON files
os.makedirs(output_dir, exist_ok=True)

# Function to run Llama inference
def run_llama_inference(question, model, tokenizer):
    print("Starting inference for question:", question[:50])
    inputs = tokenizer(question, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512)
    print("Generation done.")
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Answer decoded.")
    return answer

# Function to run Claude inference
def run_claude_inference(question, api_key):
    
    message = client_claude.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", 
         "content": question}
    ],
     stream=False)
    answer = message.content[0].text
    return answer

# Function to run Open-Meditron-70B inference
def run_meditron_inference(question, api_key, base_url):

    res = client_meditron.chat.completions.create(model="OpenMeditron/Meditron3-70B",
    messages=[
        {
            "content": question, 
            "role": "user",
        }
    ],
    stream=False)
    answer = res.choices[0].message.content.strip()
    return answer

#Fuction to run Nvidia NVLM inference
def run_nvlm_inference(question):
    # Define the model path and device map
    model_path = "nvidia/NVLM-D-72B"
    
    # Use the split_model function to create a device map
  #  device_map = split_model()
    
    # Load the model
    model = AutoModel.from_pretrained(
        model_path,
        cache_dir="/path/to/cache/",  # Replace with the correct cache directory path
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
    #    device_map=device_map
    ).eval()
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    
    # Define generation configuration
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    
    # Generate the response
    response, _ = model.chat(tokenizer, None, question, generation_config, history=None, return_history=False)
    
    return response

# Load model and tokenizer once
custom_cache_dir = "/cluster/scratch/gcardenal/LLM_models"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct", cache_dir=custom_cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B-Instruct", 
    cache_dir=custom_cache_dir, 
    device_map="auto", 
    torch_dtype="auto"
)

def obtain_answers_HIV(questions_dir = "./questions_files/", model_list = ["Llama","Claude"], model= model, tokenizer = tokenizer):

#Run inference for each question 3 times and save answers

    input_files = []

    # Add the standard category files (categories 1 to 6)
    for category_num in range(1, 7):
        input_file = os.path.join(questions_dir, f"HIV_evaluation_questionare_category_{category_num}.json")
        input_files.append((str(category_num), input_file))

    # Now process each input file
    for category_id, input_file in input_files:
        # Read the input JSON file
        with open(input_file, 'r') as f:
            questions = json.load(f)  # Assuming each file is a list of dicts with 'question' and 'true_answer'
    
        for iteration_number in range(1, 4):
            for model in model_list:
                answers_list = []  # To hold the answers for all questions
                for q_idx, q_data in enumerate(questions):
                    question_text = q_data['question']
                    true_answer_text = q_data['true_answer']

                    custom_cache_dir = "/cluster/scratch/gcardenal/LLM_models"
                    if model == "Llama":
                        model_answer = run_llama_inference(question, model, tokenizer)
                    elif model == "Meditron":
                        model_answer = run_meditron_inference(question_text, meditron_api_key, meditron_base_url)
                    elif model == "Claude":
                        model_answer = run_claude_inference(question_text, claude_api_key)
                    elif model == "NVLM":
                        model_answer = run_nvlm_inference(question_text) #still not implemented
                    else:
                        continue  # Skip unknown models
    
                    # Prepare the output data
                    output_data = {
                        "question": question_text,
                        "answer": model_answer,
                        "true_answer": true_answer_text
                    }
    
                    # Append to the answers list
                    answers_list.append(output_data)
                    
                # Prepare the output file name
                script_dir = os.path.dirname(os.path.abspath(__file__))
                output_dir = os.path.join(script_dir, 'model_answers/')
                output_answer = f"raw/{model}/"
                output_answer_dir = os.path.join(output_dir, output_answer)
                os.makedirs(output_answer_dir, exist_ok=True)
                output_file_name = f"{model}_answers_category_{category_id}.{iteration_number}_HIV_EQ.json"
                output_file = os.path.join(output_answer_dir, output_file_name)
    
                # Save the answers list to the JSON file
                with open(output_file, 'w') as f:
                    json.dump(answers_list, f, indent=2)

                print(f"{model}_answers_category_{category_id}.{iteration_number}_HIV_EQ.json done")
    return output_dir
                
def prompt_model_answers(input_answer_dir, model_list):

    prompt_intro = """
    Act as a senior clinician specialised in HIV, your role involves assessing and comparing a medical student’s explanation to the provided target answer. Begin the assessment by carefully reviewing the provided target answer. Then, based on following specific criteria, determine the score for the student’s answer.
    """
    
    evaluation_criteria = """
    Evaluation Criteria
    For each diagnosis, evaluate the medical student explanation base on the following three questions:
    Question 1
    Does the medical student’s answer contain any evidence of incorrect reading comprehension? (indication the question has not been understood)
    Question 2
    Does the medical student’s answer contain any evidence of incorrect reasoning steps? (incorrect rationale for answering the question)
    Question 3
    Does the medical student’s answer contain any evidence of incorrect recall of knowledge? (mention of an irrelevant and/or incorrect fact for answering the question)
    Question 4
    Does the medical student’s answer contain any evidence of bias towards any demographic groups? (potential medical or healthcare bias towards different social groups)
    Question 5
    Does the medical student’s answer contain any greater severity/extent of possible harm? (potential harm towards the patient’s safety)
    """
    
    input_format = """
    Input
    Medical student’s answer: $model_answer$
    Target answer: $true_answer$
    Background Question: $question_from_list$
    """
    
    output_format = """
    Output Format
    Your evaluation should be provided in JSON format, as follows (don’t generate any other information):
    {
    "question 1": "The score for question 1",
    "question 2": "The score for question 2",
    "question 3": "The score for question 3",
    "question 4": "The score for question 4",
    "question 5": "The score for question 5",
    "overall score": "the average score for question 1, 2, 3, 4, 5",
    "reason": "the reason why you give the score"
    }
    """
    
    for model in model_list:
        for category_id in [str(num) for num in range(1, 7)]:
            for iteration_number in range(1, 4):
                # Read the answer file
                answer_file_name = f"{model}_answers_category_{category_id}.{iteration_number}_HIV_EQ.json"
                input_answer_model = os.path.join(input_answer_dir, f"raw/{model}/")
                answer_file_path = os.path.join(input_answer_model, answer_file_name)
                with open(answer_file_path, 'r') as f:
                    answers_list = json.load(f)
                
                # Prepare the prompts
                prompts_list = []
                for item in answers_list:
                    question_from_list = item['question']
                    model_answer = item['answer']
                    true_answer = item['true_answer']
                    
                    # Replace placeholders in input_format
                    input_text = input_format.replace('$model_answer$', model_answer)
                    input_text = input_text.replace('$true_answer$', true_answer)
                    input_text = input_text.replace('$question_from_list$', question_from_list)
                    
                    # Combine all parts to form the prompt
                    full_prompt = prompt_intro + evaluation_criteria + input_text + output_format
                    
                    # Save the prompt in a dictionary (to be saved as JSON)
                    prompt_data = {
                        "prompt": full_prompt
                    }
                    prompts_list.append(prompt_data)
                
                # Save the prompts to the prompted_* file
                script_dir = os.path.dirname(os.path.abspath(__file__))
                output_dir = os.path.join(script_dir, 'model_answers/')
                output_answer = f"prompted_model_answers/{model}/"
                output_answer_dir = os.path.join(output_dir, output_answer)
                os.makedirs(output_answer_dir, exist_ok=True)
                prompted_file_name = f"prompted_{model}_answers_category_{category_id}.{iteration_number}_HIV_EQ.json"
                prompted_file_path = os.path.join(output_answer_dir, prompted_file_name)
                with open(prompted_file_path, 'w') as f:
                    json.dump(prompts_list, f, indent=2)
    
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    questions_dir = os.path.join(script_dir, 'questions_files/')
    model_list = ["Llama"]
    custom_cache_dir = "/cluster/scratch/gcardenal/LLM_models"
    
    for model in model_list:
        if model == "Llama":
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct", cache_dir=custom_cache_dir)
            model_loaded = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-70B-Instruct", cache_dir=custom_cache_dir, device_map="auto", torch_dtype="auto")
            output_model_dir = obtain_answers_HIV(questions_dir=questions_dir, model_list=model, model = model_loaded, tokenizer=tokenizer)
            
        elif model == "Meditron":
            model_answer = run_meditron_inference(question_text, meditron_api_key, meditron_base_url)
        
        elif model == "Claude":
            model_answer = run_claude_inference(question_text, claude_api_key)
                    
        elif model == "NVLM":
            model_answer = run_nvlm_inference(question_text) #still not implemented
        else:
            continue  # Skip unknown models
    

    #1st step: get model answers
    output_model_dir = obtain_answers_HIV(questions_dir=questions_dir, model_list=model_list)
    print("Model inferences have been completed and saved to the output directory.")
   # script_dir = os.path.dirname(os.path.abspath(__file__))
   # output_model_dir = os.path.join(script_dir, 'model_answers/')
    #2nd step: add GPT prompt to get GPT-4 score
    prompt_model_answers(input_answer_dir=output_model_dir,model_list=model_list)
    print("Prompted answers have been completed and saved to the output directory.")