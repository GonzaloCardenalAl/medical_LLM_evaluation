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
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from utils import split_model


claude_api_key = "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"
client_claude = anthropic.Anthropic(api_key=claude_api_key)

#llama_api_key = "sk-rc-COSy3IVB1YAE1-fbGyHwhg"
#llama_base_url = "https://fmapi.swissai.cscs.ch"

# Directories
input_dir = "./questions_files/"
output_dir = "./model_answers/"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------
# Functions for inference with each model
# ---------------------------------------
def run_llama_inference(question, model, tokenizer):
    print("Starting inference for question:", question[:50])
    inputs = tokenizer(question, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=1.0, do_sample=True)
    print("Generation done.")
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Answer decoded.")
    
    if answer.startswith(question):
        answer = answer[len(question):].strip()
    return answer

def run_claude_inference(question, api_key):
    message = client_claude.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {"role": "user",
             "content": question}
        ],
        stream=False
    )
    answer = message.content[0].text
    return answer

def run_meditron_inference(question, model, tokenizer):
    print("Starting inference for question:", question[:50])
    inputs = tokenizer(question, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=1.0, do_sample=True)
    print("Generation done.")
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Answer decoded.")
    return answer

def run_nvlm_inference(question, model, tokenizer):
    print("Starting inference for question:", question[:50])
    inputs = tokenizer(question, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=1.0, do_sample=True)
    print("Generation done.")
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Answer decoded.")
    return answer

def run_med42_inference(question, model, tokenizer):
    print("Starting inference for question:", question[:50])
    inputs = tokenizer(question, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=1.0, do_sample=True)
    print("Generation done.")
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Answer decoded.")
    return answer

# ---------------------------------------
# Main functions to obtain answers and prompt them
# ---------------------------------------
def obtain_answers_HIV(
    questions_dir="./questions_files/", 
    model_list=["Llama"], 
    claude_api_key=claude_api_key,
    custom_cache_dir="/cluster/scratch/gcardenal/LLM_models"
):
    input_files = []
    for category_num in range(1, 7):
        input_file = os.path.join(questions_dir, f"HIV_evaluation_questionare_category_{category_num}.json")
        input_files.append((str(category_num), input_file))

    # Process each model one at a time
    for model_name in model_list:
        # Load the model/tokenizer for the current model
        if model_name == "Llama":
            tokenizer = AutoTokenizer.from_pretrained( #"meta-llama/Llama-3.3-70B-Instruct", 
                                                      "meta-llama/Llama-3.1-70B",
                                                      cache_dir=custom_cache_dir)
            model = AutoModelForCausalLM.from_pretrained(
               # "meta-llama/Llama-3.3-70B-Instruct",
                "meta-llama/Llama-3.1-70B",
                cache_dir=custom_cache_dir,
                device_map="auto",
                torch_dtype="auto"
            )

        elif model_name == "Meditron":
            tokenizer = AutoTokenizer.from_pretrained("OpenMeditron/Meditron3-70B", cache_dir=custom_cache_dir)
            model = AutoModelForCausalLM.from_pretrained(
                "OpenMeditron/Meditron3-70B",
                cache_dir=custom_cache_dir,
                device_map="auto",
                torch_dtype="auto"
            )

        elif model_name == "Claude":
            # Claude is via API, no local model to load. Just set tokenizer/model = None
            tokenizer = None
            model = None

        elif model_name == "NVLM":

            device_map = split_model()
            tokenizer = AutoTokenizer.from_pretrained("nvidia/NVLM-D-72B", cache_dir=custom_cache_dir)
            model = AutoModelForCausalLM.from_pretrained(
                "nvidia/NVLM-D-72B",
                cache_dir=custom_cache_dir,
                device_map= device_map,
                torch_dtype="bfloat16"
            )

        elif model_name == "Med42":
            tokenizer = AutoTokenizer.from_pretrained("m42-health/Llama3-Med42-70B", cache_dir=custom_cache_dir)
            model = AutoModelForCausalLM.from_pretrained(
                "m42-health/Llama3-Med42-70B",
                cache_dir=custom_cache_dir,
                device_map="auto",
                torch_dtype="auto"
            )
        else:
            # Unknown model, skip
            continue

        # Run inference for the current model
        for category_id, input_file in input_files:
            with open(input_file, 'r') as f:
                questions = json.load(f)  # list of dicts with 'question' and 'true_answer'

            for iteration_number in range(1, 4):
                answers_list = []
                for q_data in questions:
                    question_text = q_data['question']
                    true_answer_text = q_data['true_answer']

                    if model_name == "Llama":
                        model_answer = run_llama_inference(question_text, model, tokenizer)
                    elif model_name == "Meditron":
                        model_answer = run_meditron_inference(question_text, model, tokenizer)
                    elif model_name == "Claude":
                        model_answer = run_claude_inference(question_text, claude_api_key)
                    elif model_name == "NVLM":
                        model_answer = run_nvlm_inference(question_text, model, tokenizer)
                    elif model_name == "Med42":
                         model_answer = run_med42_inference(question_text, model, tokenizer)
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
                output_answer = f"raw/{model_name}/"
                output_answer_dir = os.path.join(output_dir, output_answer)
                os.makedirs(output_answer_dir, exist_ok=True)
                output_file_name = f"{model_name}_answers_category_{category_id}.{iteration_number}_HIV_EQ.json"
                output_file = os.path.join(output_answer_dir, output_file_name)

                with open(output_file, 'w') as f:
                    json.dump(answers_list, f, indent=2)

                print(f"{model_name}_answers_category_{category_id}.{iteration_number}_HIV_EQ.json done")

        # After finishing with this model, unload it
        if model_name != "Claude":  # Claude was not loaded locally
            del model
            del tokenizer
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
                    # If file doesn't exist, skip
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
                output_answer = f"prompted_model_answers/{model_name}/"
                output_answer_dir = os.path.join(output_dir, output_answer)
                os.makedirs(output_answer_dir, exist_ok=True)
                prompted_file_name = f"prompted_{model_name}_answers_category_{category_id}.{iteration_number}_HIV_EQ.json"
                prompted_file_path = os.path.join(output_answer_dir, prompted_file_name)
                with open(prompted_file_path, 'w') as f:
                    json.dump(prompts_list, f, indent=2)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    questions_dir = os.path.join(script_dir, 'questions_files/')
    
    # Models to run: load and unload each model one by one
    model_list = ["Llama", "Meditron"]

    # 1st step: get model answers
    output_model_dir = obtain_answers_HIV(questions_dir=questions_dir, model_list=model_list)
    print("Model inferences have been completed and saved to the output directory.")

    # 2nd step: add GPT prompt to get GPT-4 score
    prompt_model_answers(input_answer_dir=output_model_dir, model_list=model_list)
    print("Prompted answers have been completed and saved to the output directory.")