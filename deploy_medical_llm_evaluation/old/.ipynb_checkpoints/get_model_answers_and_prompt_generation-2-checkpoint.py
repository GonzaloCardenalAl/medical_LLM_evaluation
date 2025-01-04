import os
import json
import openai
from openai import OpenAI
import pandas as pd
import re
import csv
import numpy as np
import anthropic
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import split_model

claude_api_key = "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"
client_claude = anthropic.Anthropic(api_key=claude_api_key)

llama_api_key = "sk-rc-COSy3IVB1YAE1-fbGyHwhg"
llama_base_url = "https://fmapi.swissai.cscs.ch"

meditron_api_key = "sk-rc-COSy3IVB1YAE1-fbGyHwhg"
meditron_base_url = "https://fmapi.swissai.cscs.ch"

client_openai = OpenAI(api_key=llama_api_key , base_url=llama_base_url) # For Llama and Meditron API calls

# Store references to loaded models (for fallback) so we don't load them multiple times
loaded_models = {
    "Llama": {"model": None, "tokenizer": None},
    "Meditron": {"model": None, "tokenizer": None},
    "NVLM": {"model": None, "tokenizer": None},
    "Med42": {"model": None, "tokenizer": None}
}

custom_cache_dir = "/cluster/scratch/gcardenal/LLM_models"

# Directories
input_dir = "./questions_files/"
output_dir = "./model_answers/"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------
# Helper function to try API inference
# ---------------------------------------
def try_api_inference(model_name, question):
    """
    Try to run the API inference for Llama or Meditron. 
    Returns the model answer if successful, or None if there's an error.
    """
    try:
        if model_name == "Llama":
            res = client_openai.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-70B-Instruct",
                messages=[{"content": question, "role": "user"}],
                stream=False
            )
            answer = res.choices[0].message.content.strip()
            return answer

        elif model_name == "Meditron":
            res = client_openai.chat.completions.create(
                model="OpenMeditron/Meditron3-70B",
                messages=[{"content": question, "role": "user"}],
                stream=False
            )
            answer = res.choices[0].message.content.strip()
            return answer

        return None
    except Exception as e:
        print(f"API call for {model_name} failed with error: {e}")
        return None

def load_llama_model():
    if loaded_models["Llama"]["model"] is None:
        print("Falling back to local Llama inference - Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-70B",
            cache_dir=custom_cache_dir
        )
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-70B",
            cache_dir=custom_cache_dir,
            device_map="auto",
            torch_dtype="auto"
        )
        loaded_models["Llama"]["model"] = model
        loaded_models["Llama"]["tokenizer"] = tokenizer

def load_meditron_model():
    if loaded_models["Meditron"]["model"] is None:
        print("Falling back to local Meditron inference - Loading model...")
        tokenizer = AutoTokenizer.from_pretrained("OpenMeditron/Meditron3-70B", cache_dir=custom_cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            "OpenMeditron/Meditron3-70B",
            cache_dir=custom_cache_dir,
            device_map="auto",
            torch_dtype="auto"
        )
        loaded_models["Meditron"]["model"] = model
        loaded_models["Meditron"]["tokenizer"] = tokenizer

# ---------------------------------------
# Functions for inference with each model
# ---------------------------------------
def run_llama_inference(question):
    # Try API first, no model loaded yet
 #   answer = try_api_inference("Llama", question)
    answer = None
    if answer is not None:
        return answer
    
    # If API failed, load model locally and run inference
    load_llama_model()
    model = loaded_models["Llama"]["model"]
    tokenizer = loaded_models["Llama"]["tokenizer"]

    print("Starting inference for question locally (Llama):", question[:50])
    inputs = tokenizer(question, return_tensors="pt").to(model.device)
   # outputs = model.generate(**inputs, max_new_tokens=512, temperature=1.0, do_sample=True)
    outputs = model.generate(**inputs, max_new_tokens=1024)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if answer.startswith(question):
        answer = answer[len(question):].strip()
    return answer

def run_meditron_inference(question):
    # Try API first, no model loaded yet
    answer = try_api_inference("Meditron", question)
    if answer is not None:
        return answer

    # If API failed, load model locally and run inference
    load_meditron_model()
    model = loaded_models["Meditron"]["model"]
    tokenizer = loaded_models["Meditron"]["tokenizer"]

    print("Starting inference for question locally (Meditron):", question[:50])
    inputs = tokenizer(question, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=1.0, do_sample=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if answer.startswith(question):
        answer = answer[len(question):].strip()
        
    return answer

def run_claude_inference(question, api_key):
    message = client_claude.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": question}
        ],
        stream=False
    )
    answer = message.content[0].text
    return answer

def load_nvlm_model():
    if loaded_models["NVLM"]["model"] is None:
        print("Loading NVLM model...")
        device_map = split_model()
        tokenizer = AutoTokenizer.from_pretrained("nvidia/NVLM-D-72B", cache_dir=custom_cache_dir, trust_remote_code=True, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
                "nvidia/NVLM-D-72B", cache_dir=custom_cache_dir, 
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=False,
                trust_remote_code=True,
                device_map=device_map).eval()


        loaded_models["NVLM"]["model"] = model
        loaded_models["NVLM"]["tokenizer"] = tokenizer

def run_nvlm_inference(question):
    load_nvlm_model()
    model = loaded_models["NVLM"]["model"]
    tokenizer = loaded_models["NVLM"]["tokenizer"]

    print("Starting inference for question (NVLM):", question[:50])
  #  inputs = tokenizer(question, return_tensors="pt").to(model.device)
  #  outputs = model.generate(**inputs, max_new_tokens=512, temperature=1.0, do_sample=True)
  #  answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    answer, history = model.chat(tokenizer, None, question, generation_config, history=None, return_history=True)
    print(answer)

    if answer.startswith(question):
        answer = answer[len(question):].strip()
         
    return answer

def load_med42_model():
    if loaded_models["Med42"]["model"] is None:
        print("Loading Med42 model...")
        tokenizer = AutoTokenizer.from_pretrained("m42-health/Llama3-Med42-70B", cache_dir=custom_cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            "m42-health/Llama3-Med42-70B",
            cache_dir=custom_cache_dir,
            device_map="auto",
            torch_dtype="auto"
        )
        loaded_models["Med42"]["model"] = model
        loaded_models["Med42"]["tokenizer"] = tokenizer

def run_med42_inference(question):
    load_med42_model()
    model = loaded_models["Med42"]["model"]
    tokenizer = loaded_models["Med42"]["tokenizer"]

    print("Starting inference for question (Med42):", question[:50])
    inputs = tokenizer(question, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=1.0, do_sample=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if answer.startswith(question):
        answer = answer[len(question):].strip()
        
    return answer

# ---------------------------------------
# Main functions to obtain answers and prompt them
# ---------------------------------------
def obtain_answers_HIV(
    questions_dir="./questions_files/", 
    model_list=["Llama"], 
    claude_api_key=claude_api_key
):
    input_files = []
    for category_num in range(1, 7):
        input_file = os.path.join(questions_dir, f"HIV_evaluation_questionare_category_{category_num}.json")
        input_files.append((str(category_num), input_file))

    for model_name in model_list:
        for category_id, input_file in input_files:
            with open(input_file, 'r') as f:
                questions = json.load(f)

            for iteration_number in range(1, 4):
                answers_list = []
                for q_data in questions:
                    question_text = q_data['question']
                    true_answer_text = q_data['true_answer']

                    if model_name == "Llama":
                        model_answer = run_llama_inference(question_text)
                    elif model_name == "Meditron":
                        model_answer = run_meditron_inference(question_text)
                    elif model_name == "Claude":
                        model_answer = run_claude_inference(question_text, claude_api_key)
                    elif model_name == "NVLM":
                        model_answer = run_nvlm_inference(question_text)
                    elif model_name == "Med42":
                        model_answer = run_med42_inference(question_text)
                    else:
                        continue

                    output_data = {
                        "question": question_text,
                        "answer": model_answer,
                        "true_answer": true_answer_text
                    }
                    answers_list.append(output_data)

                script_dir = os.path.dirname(os.path.abspath(__file__))
                output_dir = os.path.join(script_dir, 'model_answers/')
                output_answer_dir = os.path.join(output_dir, f"raw/{model_name}/")
                os.makedirs(output_answer_dir, exist_ok=True)
                output_file_name = f"{model_name}_answers_category_{category_id}.{iteration_number}_HIV_EQ.json"
                output_file = os.path.join(output_answer_dir, output_file_name)

                with open(output_file, 'w') as f:
                    json.dump(answers_list, f, indent=2)

                print(f"{model_name}_answers_category_{category_id}.{iteration_number}_HIV_EQ.json done")

        # After finishing all inferences for this model, if we loaded a local model, unload it
        if model_name in ["Llama", "Meditron", "NVLM", "Med42"]:
            if loaded_models[model_name]["model"] is not None:
                del loaded_models[model_name]["model"]
                del loaded_models[model_name]["tokenizer"]
                loaded_models[model_name]["model"] = None
                loaded_models[model_name]["tokenizer"] = None
                torch.cuda.empty_cache()

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
    
    for model_name in model_list:
        for category_id in [str(num) for num in range(1, 7)]:
            for iteration_number in range(1, 4):
                answer_file_name = f"{model_name}_answers_category_{category_id}.{iteration_number}_HIV_EQ.json"
                input_answer_model = os.path.join(input_answer_dir, f"raw/{model_name}/")
                answer_file_path = os.path.join(input_answer_model, answer_file_name)

                if not os.path.exists(answer_file_path):
                    continue

                with open(answer_file_path, 'r') as f:
                    answers_list = json.load(f)
                
                prompts_list = []
                for item in answers_list:
                    question_from_list = item['question']
                    model_answer = item['answer']
                    true_answer = item['true_answer']

                    input_text = input_format.replace('$model_answer$', model_answer)
                    input_text = input_text.replace('$true_answer$', true_answer)
                    input_text = input_text.replace('$question_from_list$', question_from_list)

                    full_prompt = prompt_intro + evaluation_criteria + input_text + output_format
                    prompt_data = {
                        "prompt": full_prompt
                    }
                    prompts_list.append(prompt_data)
                
                script_dir = os.path.dirname(os.path.abspath(__file__))
                output_dir = os.path.join(script_dir, 'model_answers/')
                output_answer_dir = os.path.join(output_dir, f"prompted_model_answers/{model_name}/")
                os.makedirs(output_answer_dir, exist_ok=True)
                prompted_file_name = f"prompted_{model_name}_answers_category_{category_id}.{iteration_number}_HIV_EQ.json"
                prompted_file_path = os.path.join(output_answer_dir, prompted_file_name)
                with open(prompted_file_path, 'w') as f:
                    json.dump(prompts_list, f, indent=2)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    questions_dir = os.path.join(script_dir, 'questions_files/')
    
 #   model_list = ["Llama", "Meditron","NVLM", "Med42"]
    model_list = ["NVLM", "Claude"]
    #model_list = ["Llama"]
    # 1st step: get model answers
    output_model_dir = obtain_answers_HIV(questions_dir=questions_dir, model_list=model_list)
    print("Model inferences have been completed and saved to the output directory.")

    # 2nd step: add GPT prompt to get GPT-4 score
    prompt_model_answers(input_answer_dir=output_model_dir, model_list=model_list)
    print("Prompted answers have been completed and saved to the output directory.")