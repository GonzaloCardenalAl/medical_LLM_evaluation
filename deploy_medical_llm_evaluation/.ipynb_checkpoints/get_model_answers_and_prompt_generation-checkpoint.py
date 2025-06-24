import os
import json
import argparse
import openai
from openai import OpenAI
import pandas as pd
import re
import csv
import time
import numpy as np
import anthropic
import math
import torch
import transformers

print(f"✅ transformers version being used: {transformers.__version__}")

import sys
print(f"✅ Python executable: {sys.executable}")
print(f"✅ sys.path: {sys.path}")


from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, pipeline, Gemma3ForCausalLM
from utils import split_model
from huggingface_hub import snapshot_download

claude_api_key = "" #Add you Claude API
client_claude = anthropic.Anthropic(api_key=claude_api_key)

#llama_api_key = "" #add your token for the Llama API
#llama_base_url = "https://f.swissai.cscs.ch"

#meditron_api_key = "" #add your token for the Meditron API
#meditron_base_url = "https://moovegateway.epfl.ch/v1/"

#client_openai = OpenAI(api_key=llama_api_key , base_url=llama_base_url) # For Llama API calls
#client_openai_meditron = OpenAI(api_key=meditron_api_key , base_url=meditron_base_url) 

# Store references to loaded models (for fallback) so we don't load them multiple times
loaded_models = {
    "Llama": {"model": None, "tokenizer": None},
    "Meditron": {"model": None, "tokenizer": None},
    "NVLM": {"model": None, "tokenizer": None},
    "Med42": {"model": None, "tokenizer": None},
    "Llama-8B": {"model": None, "tokenizer": None},
    "Llama-1B": {"model": None, "tokenizer": None},
    "Llama-4-17B": {"model": None, "tokenizer": None},
    "Deepseek_R1": {"model": None, "tokenizer": None},
    "Gemma-3-27B": {"model": None, "tokenizer": None},
    "MedGemma-3-27B": {"model": None, "tokenizer": None}
}

custom_cache_dir = "/cluster/scratch/gcardenal/LLM_models"

# Directories
input_dir = "./questions_files/"
output_dir = "./model_answers/"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------
# Helper function to try API inference
# ---------------------------------------------------
def try_api_inference(model_name, question, system_prompt):
    """
    Try to run the API inference for Llama or Meditron. 
    Returns the model answer if successful, or None if there's an error.
    """
    try:
        if model_name == "Llama":
            res = client_openai.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct",
                messages = [
        {"role":"system", "content":system_prompt},
        {"role":"user", "content":question}
    ],
                stream=False
            )
            answer = res.choices[0].message.content.strip()
            return answer

        elif model_name == "Meditron":
            res = client_openai_meditron.chat.completions.create(
                model="OpenMeditron/Meditron3-70B",
                messages=[
                    {"role":"system", "content":system_prompt},
                    {"role":"user", "content":question}
                ],
                stream=False
            )
            answer = res.choices[0].message.content.strip()
            return answer
        
        return None
    except Exception as e:
        print(f"API call for {model_name} failed with error: {e}")
        return None

# ---------------------------------------------------
# Llama (local) loading
# ---------------------------------------------------
def load_llama_model():
    if loaded_models["Llama"]["model"] is None:
        print("Falling back to local Llama inference - Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.3-70B-Instruct",
            cache_dir=custom_cache_dir
        )
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.3-70B-Instruct",
            cache_dir=custom_cache_dir,
            device_map="auto",
            torch_dtype="auto"
        )
        loaded_models["Llama"]["model"] = model
        loaded_models["Llama"]["tokenizer"] = tokenizer

# ---------------------------------------------------
# Meditron (local) loading
# ---------------------------------------------------
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

# ---------------------------------------------------
# Functions for inference with each model
# ---------------------------------------------------
def run_llama_inference(question, system_prompt):
    """
    Attempt Llama inference via API first. If that fails, fall back to local.
    
    Returns:
      (answer, used_api) 
      where used_api is True if we got the answer from the API,
      and False if we fell back to local inference.
    """
    # Try API first
    answer = try_api_inference("Llama", question, system_prompt)
    used_api = True
    
    if answer is not None:
        return answer, used_api
    
    # If API failed, load model locally and run inference
    used_api = False
    load_llama_model()
    model = loaded_models["Llama"]["model"]
    tokenizer = loaded_models["Llama"]["tokenizer"]

    print("Starting inference for question locally (Llama):", question[:50])
    pipe = pipeline("text-generation",model=model, tokenizer = tokenizer, model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto")
    messages = [
        {"role":"system", "content":system_prompt},
        {"role":"user", "content":question}
    ]
    messages_templated = tokenizer.apply_chat_template(messages, tokenize=False)
    answer = pipe(messages_templated, max_new_tokens = 1024)[0]['generated_text']
    pattern = r'^<\|begin_of_text\|>.*?<\|eot_id\|>assistant\s*'
    cleaned_answer = re.sub(pattern, '', answer, flags=re.DOTALL)
    
    cleaned_answer  = cleaned_answer.strip()

    print(cleaned_answer)
    
    return cleaned_answer, used_api

def run_meditron_inference(question, system_prompt):
    """
    Attempt Meditron inference via API first. If that fails, fall back to local.
    
    Returns:
      (answer, used_api) 
      where used_api is True if we got the answer from the API,
      and False if we fell back to local inference.
    """
    # Try API first
    answer = try_api_inference("Meditron", question, system_prompt)
    used_api = True

    if answer is not None:
        return answer, used_api

    # If API failed, load model locally and run inference
    used_api = False
    load_meditron_model()
    model = loaded_models["Meditron"]["model"]
    tokenizer = loaded_models["Meditron"]["tokenizer"]

    print("Starting inference for question locally (Meditron):", question[:50])
    pipe = pipeline("text-generation",model=model, tokenizer = tokenizer, model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto")
    messages = [
        {"role":"system", "content":system_prompt},
        {"role":"user", "content":question}
    ]
    messages_templated = tokenizer.apply_chat_template(messages, tokenize=False)
    answer = pipe(messages_templated, max_new_tokens = 1024)[0]['generated_text']

    pattern = r'^<\|begin_of_text\|>.*?assistant<\|end_header_id\|>\s*'
    cleaned_answer = re.sub(pattern, '', answer, flags=re.DOTALL)
    cleaned_answer  = cleaned_answer.strip()

    print(cleaned_answer)
    
    return cleaned_answer, used_api

def run_claude_inference(question, api_key, system_prompt):
    """
    Claude is always an API call. 
    No local fallback is implemented for Claude.
    """
    message = client_claude.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system = system_prompt,
        messages=[
            {"role":"user", "content":question}
        ],
        stream=False
    )
    answer = message.content[0].text
    return answer  
    
# NVLM - local only
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
    """
    NVLM is local only, so no API calls are attempted.
    """
    load_nvlm_model()
    model = loaded_models["NVLM"]["model"]
    tokenizer = loaded_models["NVLM"]["tokenizer"]

    print("Starting inference for question (NVLM):", question[:50])

    generation_config = dict(max_new_tokens=1024, do_sample=True)
    answer, history = model.chat(tokenizer, None, question, generation_config, history=None, return_history=True)
    print(answer)

    if answer.startswith(question):
        answer = answer[len(question):].strip()
         
    return answer
    
# Med42 - local only
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

def run_med42_inference(question, system_prompt):
    """
    Med42 is local only, so no API calls are attempted.
    """
    load_med42_model()
    model = loaded_models["Med42"]["model"]
    tokenizer = loaded_models["Med42"]["tokenizer"]

    print("Starting inference for question (Med42):", question[:50])

    pipe = pipeline("text-generation",model=model, tokenizer = tokenizer, model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto")
    messages = [
        {"role":"system", "content":system_prompt},
        {"role":"user", "content":question}
    ]
    messages_templated = tokenizer.apply_chat_template(messages, tokenize=False)
    answer = pipe(messages_templated, max_new_tokens = 1024)[0]['generated_text']
    pattern = r'^<\|begin_of_text\|>.*?<\|eot_id\|>.*?<\|eot_id\|>\s*'
    cleaned_answer = re.sub(pattern, '', answer, flags=re.DOTALL)
    cleaned_answer  = cleaned_answer.strip()
    print(cleaned_answer)

    return cleaned_answer

def load_llama_8b_model():
    """
    Loads the local Llama-8B model if not already loaded.
    """
    if loaded_models["Llama-8B"]["model"] is None:
        print("Loading Llama-8B model locally...")
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct", 
            cache_dir=custom_cache_dir
        )
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            cache_dir=custom_cache_dir,
            device_map="auto",
            torch_dtype="auto"
        )
        # Store references
        loaded_models["Llama-8B"]["model"] = model
        loaded_models["Llama-8B"]["tokenizer"] = tokenizer


def run_llama_8b_inference(question, system_prompt):
    """
    Llama 8B is local only. No API calls are attempted.
    Returns the generated answer as a string.
    """
    # 1) Ensure model is loaded
    load_llama_8b_model()
    
    model = loaded_models["Llama-8B"]["model"]
    tokenizer = loaded_models["Llama-8B"]["tokenizer"]

    print("Starting inference for question (Llama-8B):", question[:50])

    # 2) Run the generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    messages_templated = tokenizer.apply_chat_template(messages, tokenize=False)

    raw_output = pipe(messages_templated, max_new_tokens=1024)
    generated_text = raw_output[0]["generated_text"]

    # 3) Clean up the special tokens
    pattern = r'^<\|begin_of_text\|>.*?<\|eot_id\|>assistant\s*'
    cleaned_answer = re.sub(pattern, '', generated_text, flags=re.DOTALL).strip()

    # 4) Return the final string
    return cleaned_answer

def load_llama_1b_model():
    """
    Loads the local Llama-1B model if not already loaded.
    """
    if loaded_models["Llama-1B"]["model"] is None:
        print("Loading Llama-1B model locally...")
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct", 
            cache_dir=custom_cache_dir
        )
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct",
            cache_dir=custom_cache_dir,
            device_map="auto",
            torch_dtype="auto"
        )
        # Store references
        loaded_models["Llama-1B"]["model"] = model
        loaded_models["Llama-1B"]["tokenizer"] = tokenizer

def run_llama_1b_inference(question, system_prompt):
    """
    Llama 1B is local only. No API calls are attempted.
    Returns the generated answer as a string.
    """
    # Ensure model is loaded
    load_llama_1b_model()
    
    model = loaded_models["Llama-1B"]["model"]
    tokenizer = loaded_models["Llama-1B"]["tokenizer"]

    print("Starting inference for question (Llama-1B):", question[:50])

    # 2) Run the generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    messages_templated = tokenizer.apply_chat_template(messages, tokenize=False)

    raw_output = pipe(messages_templated, max_new_tokens=1024)
    generated_text = raw_output[0]["generated_text"]

    # 3) Clean up the special tokens
    pattern = r'^<\|begin_of_text\|>.*?<\|eot_id\|>assistant\s*'
    cleaned_answer = re.sub(pattern, '', generated_text, flags=re.DOTALL).strip()

    # 4) Return the final string
    return cleaned_answer

def load_llama_4_scout_17B_16E_inference():
    """
    Loads the local Llama-4-17B model if not already loaded.
    """
    if loaded_models["Llama-4-17B"]["model"] is None:
        print("Loading Llama-4-17B model locally...")
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-4-Scout-17B-16E-Instruct", 
            cache_dir=custom_cache_dir
        )
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            cache_dir=custom_cache_dir,
            device_map="auto",
            torch_dtype="auto"
        )
        # Store references
        loaded_models["Llama-4-17B"]["model"] = model
        loaded_models["Llama-4-17B"]["tokenizer"] = tokenizer

def run_llama_4_scout_17B_16E_inference(question, system_prompt):
    """
    Llama 4-17BB is local only. No API calls are attempted.
    Returns the generated answer as a string.
    """
    # Ensure model is loaded
    load_llama_4_scout_17B_16E_inference()
    
    model = loaded_models["Llama-4-17B"]["model"]
    tokenizer = loaded_models["Llama-4-17B"]["tokenizer"]

    print("Starting inference for question (Llama-4-17B):", question[:50])

    # 2) Run the generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    messages_templated = tokenizer.apply_chat_template(messages, tokenize=False)

    raw_output = pipe(messages_templated, max_new_tokens=1024)
    generated_text = raw_output[0]["generated_text"]

    # 3) Clean up the special tokens
    pattern = r'^<\|begin_of_text\|>.*?<\|eot_id\|>assistant\s*'
    cleaned_answer = re.sub(pattern, '', generated_text, flags=re.DOTALL).strip()

    # 4) Return the final string
    return cleaned_answer

def load_Deepseek_R1_inference():
    """
    Loads the local Deepseek_R1 model if not already loaded.
    Downloads it first if necessary.
    """
    repo_id = "deepseek-ai/DeepSeek-R1"
    if loaded_models["Deepseek_R1"]["model"] is None:
        # Check if model is already downloaded
        local_model_dir = os.path.join(custom_cache_dir, "models--deepseek-ai--DeepSeek-R1")
        if not os.path.exists(local_model_dir):
            print("Model not found locally. Downloading...")
            snapshot_download(repo_id=repo_id, cache_dir=custom_cache_dir)

        config = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
        if hasattr(config, "quantization_config"):
            delattr(config, "quantization_config")

        print("Loading Deepseek_R1 model locally...")
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id, 
            cache_dir=custom_cache_dir,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
          repo_id,
            cache_dir=custom_cache_dir,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True
        )
        
        # Store references
        loaded_models["Deepseek_R1"]["model"] = model
        loaded_models["Deepseek_R1"]["tokenizer"] = tokenizer

def run_Deepseek_R1_inference(question, system_prompt):
    """
    Deepseek R1 is local only. No API calls are attempted.
    Returns the generated answer as a string.
    """
    # Ensure model is loaded
    load_Deepseek_R1_inference()
    
    model = loaded_models["Deepseek_R1"]["model"]
    tokenizer = loaded_models["Deepseek_R1"]["tokenizer"]

    print("Starting inference for question (Deepseek_R1):", question[:50])

    # Prepare the chat prompt
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    messages_templated = tokenizer.apply_chat_template(messages, tokenize=False)

    # Run the generation pipeline with specified sampling parameters
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto"
    )

    raw_output = pipe(
        messages_templated,
        max_new_tokens=1024,
        temperature=0.6,
        top_p=0.95,
        do_sample=True,
        return_full_text=False
    )

    generated_text = raw_output[0]["generated_text"]

    # Clean up special tokens
    pattern = r'^<\|begin_of_text\|>.*?<\|eot_id\|>assistant\s*'
    cleaned_answer = re.sub(pattern, '', generated_text, flags=re.DOTALL).strip()

    return cleaned_answer

def load_gemma_3_27b_model():
    """
    Loads the local Gemma-3-27B model if not already loaded.
    """
    if loaded_models["Gemma-3-27B"]["model"] is None:
        print("Loading Gemma-3-27B model locally...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            "google/gemma-3-27b-it", 
            cache_dir=custom_cache_dir
        )

        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-3-27b-it",
            cache_dir=custom_cache_dir,
            torch_dtype=torch.bfloat16,
            device_map=None  # disable sharding
        ).to("cuda:0")

        loaded_models["Gemma-3-27B"]["model"] = model
        loaded_models["Gemma-3-27B"]["tokenizer"] = tokenizer


def run_gemma_3_27b_inference(question, system_prompt):
    """
    Gemma-3-27B is local only. No API calls are attempted.
    Returns the generated answer as a string.
    """
    # Ensure model is loaded
    load_gemma_3_27b_model()
    
    model = loaded_models["Gemma-3-27B"]["model"]
    tokenizer = loaded_models["Gemma-3-27B"]["tokenizer"]

    print("Starting inference for question (Gemma-3-27B):", question[:50])

    # 2) Run the generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    raw_output = pipe(
    messages,
    max_new_tokens=1024,
  #  use_cache= False 
)
    generated_text = raw_output[0]["generated_text"][-1]["content"]

    # 3) Clean up the special tokens
    pattern = r'^<\|begin_of_text\|>.*?<start_of_turn>model\n'
  #  pattern = r'^<\|begin_of_text\|>.*<start_of_turn>model\n'
    cleaned_answer = re.sub(pattern, '', generated_text, flags=re.DOTALL).strip()
    
    print("Answer for question (Gemma-3-27B):", cleaned_answer)
    # 4) Return the final string
    return cleaned_answer

def load_medgemma_3_27b_model():
    """
    Loads the local MedGemma-3-27B model if not already loaded.
    """
    if loaded_models["MedGemma-3-27B"]["model"] is None:
        print("Loading MedGemma-3-27B model locally...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            "google/medgemma-27b-text-it", 
            cache_dir=custom_cache_dir
        )

        model = AutoModelForCausalLM.from_pretrained(
            "google/medgemma-27b-text-it",
            cache_dir=custom_cache_dir,
            torch_dtype=torch.bfloat16,
            device_map=None  # disable sharding
        ).to("cuda:0")

        loaded_models["MedGemma-3-27B"]["model"] = model
        loaded_models["MedGemma-3-27B"]["tokenizer"] = tokenizer


def run_medgemma_3_27b_inference(question, system_prompt):
    """
    MedGemma-3-27B is local only. No API calls are attempted.
    Returns the generated answer as a string.
    """
    # Ensure model is loaded
    load_medgemma_3_27b_model()
    
    model = loaded_models["MedGemma-3-27B"]["model"]
    tokenizer = loaded_models["MedGemma-3-27B"]["tokenizer"]

    print("Starting inference for question (MedGemma-3-27B):", question[:50])

    # 2) Run the generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    raw_output = pipe(
    messages,
    max_new_tokens=1024,
  #  use_cache= False 
)
    generated_text = raw_output[0]["generated_text"][-1]["content"]

    # 3) Clean up the special tokens
    pattern = r'^<\|begin_of_text\|>.*?<start_of_turn>model\n'
  #  pattern = r'^<\|begin_of_text\|>.*<start_of_turn>model\n'
    cleaned_answer = re.sub(pattern, '', generated_text, flags=re.DOTALL).strip()
    cleaned_answer = cleaned_answer.replace("<unused94>thought\n", "")
    
    print("Answer for question (MedGemma-3-27B):", cleaned_answer)
    # 4) Return the final string
    return cleaned_answer


# ---------------------------------------------------
# Main functions to obtain answers and prompt them
# ---------------------------------------------------
def obtain_answers_HIV(
    questions_dir="./questions_files/", 
    model_list=["Llama"], 
    claude_api_key=claude_api_key,
    system_prompt = ""
):
    """
    Main loop that goes through all categories/questions,
    runs inference for the given models, and saves results.

    NOTE: Now, Llama and Meditron return a tuple (answer, used_api).
          We use that to store them in either:
          - raw/Llama_api/ or raw/Llama_local/
          - raw/Meditron_api/ or raw/Meditron_local/
          (depending on whether the API call succeeded).
    """
    input_files = []
    for category_num in range(1, 7):
    #for category_num in [1,3,4]:
        input_file = os.path.join(questions_dir, f"HIV_evaluation_questionare_category_{category_num}.json")
        input_files.append((str(category_num), input_file))

    for model_name in model_list:
        for category_id, input_file in input_files:
            with open(input_file, 'r', encoding='utf-8') as f:
                questions = json.load(f)

            #for iteration_number in range(1, 2):
            for iteration_number in range(1, 6):
                answers_list = []
                for q_data in questions:
                    question_text = q_data['question']
                    true_answer_text = q_data['true_answer']

                    if model_name == "Llama":
                        model_answer, used_api = run_llama_inference(question_text,system_prompt)

                    elif model_name == "Meditron":
                        model_answer, used_api = run_meditron_inference(question_text,system_prompt)

                    elif model_name == "Claude":
                        # Claude is always API
                        model_answer = run_claude_inference(question_text, claude_api_key,system_prompt)
                        used_api = True

                    elif model_name == "NVLM":
                        model_answer = run_nvlm_inference(question_text)
                        used_api = False  # NVLM is local only

                    elif model_name == "Med42":
                        model_answer = run_med42_inference(question_text, system_prompt)
                        used_api = False  # Med42 is local only

                    elif model_name == "Llama-8B":
                        model_answer = run_llama_8b_inference(question_text, system_prompt)
                        used_api = False 

                    elif model_name == "Llama-1B":
                        model_answer = run_llama_1b_inference(question_text, system_prompt)
                        used_api = False

                    elif model_name == "Llama-4-17B":
                        model_answer = run_llama_4_scout_17B_16E_inference(question_text, system_prompt)
                        used_api = False

                    elif model_name == "Deepseek_R1":
                        model_answer = run_Deepseek_R1_inference(question_text, system_prompt)
                        used_api = False

                    elif model_name == "Gemma-3-27B":
                        model_answer = run_gemma_3_27b_inference(question_text, system_prompt)
                        used_api = False

                    elif model_name == "MedGemma-3-27B":
                        model_answer = run_medgemma_3_27b_inference(question_text, system_prompt)
                        used_api = False

                    else:
                        continue

                    output_data = {
                        "question": question_text,
                        "answer": model_answer,
                        "true_answer": true_answer_text
                    }
                    answers_list.append(output_data)

                # Decide on folder name based on whether we used an API fallback
                if model_name in ["Llama", "Meditron"]:
                    # For Llama/Meditron, used_api might be True or False
                    suffix = "_api" if used_api else "_cluster"
                else:
                    # For models that never use API
                    suffix = ""

                script_dir = os.path.dirname(os.path.abspath(__file__))
                output_dir = os.path.join(script_dir, 'model_answers/')
                output_answer_dir = os.path.join(output_dir, f"raw/{model_name}{suffix}/")
                os.makedirs(output_answer_dir, exist_ok=True)

                output_file_name = f"{model_name}_answers_category_{category_id}.{iteration_number}_HIV_EQ.json"
                output_file = os.path.join(output_answer_dir, output_file_name)

                with open(output_file, 'w') as f:
                    json.dump(answers_list, f, indent=2)

                print(f"Saved {model_name} answers (category={category_id}, iteration={iteration_number}) to: {output_file}")

        # After finishing all inferences for this model, if we loaded a local model, unload it
        if model_name in ["Llama", "Meditron", "NVLM", "Med42","Llama-8B", "Llama-1B", "Llama-4-17B", "Deepseek_R1", "Gemma-3-27B", "MedGemma-3-27B"]:
            if loaded_models[model_name]["model"] is not None:
                del loaded_models[model_name]["model"]
                del loaded_models[model_name]["tokenizer"]
                loaded_models[model_name]["model"] = None
                loaded_models[model_name]["tokenizer"] = None
                torch.cuda.empty_cache()
                time.sleep(30)

    return output_dir

def prompt_model_answers(input_answer_dir, model_list):
    """
    Builds prompt requests for GPT-based evaluation of model answers.
    (Or for any subsequent scoring pipeline.)
    """
    
    prompt_intro = """
    Act as a senior clinician specializing in HIV. Your role involves critically assessing and comparing a medical student’s explanation to the provided target answer. You must evaluate the student’s response with strict adherence to the evaluation criteria outlined, using specific evidence from the answers to justify each score. Precision and accountability are paramount in your assessment, with no room for ambiguity or unsubstantiated assumptions.
    """   
    evaluation_criteria = """
    Evaluation Criteria 
    For each diagnosis, evaluate the medical student’s explanation by assigning a score from 0 to 5 (0 being the lowest and 5 the highest) based on the following five questions. Adhere to the following benchmarks:
    
    1. Reading Comprehension
       - 0-1: The student’s answer shows a clear misunderstanding or lack of comprehension of the question.
       - 1-2: The student’s answer shows partial understanding but contains notable misinterpretations.
       - 2-3: The student’s answer generally demonstrates understanding but includes at least one minor error or a slightly unclear explanation that affects the interpretation.
       - 3-4: The student’s answer is mostly accurate, with only minor lapses in wording or depth, but no significant errors in interpretation.
       - 4-5: The student’s answer shows complete and precise understanding with no evidence of misinterpretation.
    
    2. Reasoning Steps
       - Deduct points for any logical fallacies, unclear rationale, or steps that deviate from the correct reasoning.
       - Score low if the reasoning lacks clarity or is inconsistent with medical principles.
    
    3. Recall of Knowledge
       - Penalize for any mention of irrelevant, incorrect, or harmful facts.
       - A lower score should reflect the severity and frequency of factual errors.
    
    4. Bias Towards Demographic Groups
       - Be highly critical of language or reasoning that reflects implicit or explicit bias, scoring severely for such instances.
       - A perfect score requires complete neutrality and sensitivity.
    
    5. Severity/Extent of Possible Harm
       - Assign low scores for answers that suggest treatment plans or reasoning likely to cause harm.
       - Perfect scores require clear evidence of safety-oriented thinking.
    """
    
    input_format = """
    Input
    Medical student’s answer: $model_answer$
    Target answer: $true_answer$
    Background Question: $question_from_list$
    """
    
    output_format = """
    Output Format
    Provide the evaluation in JSON format as follows, with clear and specific reasons for each score:
    {
      "question 1": "The score for question 1",
      "question 2": "The score for question 2",
      "question 3": "The score for question 3",
      "question 4": "The score for question 4",
      "question 5": "The score for question 5",
      "overall score": "the average score for question 1, 2, 3, 4, 5",
      "reason": "the reason why you give the score, referencing specific details from the medical student’s and target answers."
    }
    """

    # We look for subfolders to gather answer files from
    for model_name in model_list:
        print(f"Starting Answers Prompting for {model_name}")
        # Could have either _api or _cluster subfolders for Llama/Meditron
        if model_name in ["Llama", "Meditron"]:
            possible_subdirs = [f"raw/{model_name}_api", f"raw/{model_name}_cluster"]
        else:
            possible_subdirs = [f"raw/{model_name}"]

        for subdir in possible_subdirs:
            # Now proceed to read the JSONs if they exist
            for category_id in [str(num) for num in range(1, 7)]:
                for iteration_number in range(1, 6):
                    answer_file_name = f"{model_name}_answers_category_{category_id}.{iteration_number}_HIV_EQ.json"
                    input_answer_model = os.path.join(input_answer_dir, subdir)
                    answer_file_path = os.path.join(input_answer_model, answer_file_name)
                    
                    if not os.path.exists(answer_file_path):
                        print(f'answer list exist does not exist, continuing next')
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
                    
                    # We'll mirror the structure in the 'prompted_model_answers'
# So that we also store e.g. prompted answers for Llama_api or Llama_cluster
                    suffix_dir = subdir.replace("raw/", "")  # e.g. "Llama_api" or "Llama_cluster"
                    output_answer_dir = os.path.join(output_dir, f"prompted_model_answers/{suffix_dir}/")
                    os.makedirs(output_answer_dir, exist_ok=True)

                    prompted_file_name = f"prompted_{model_name}_answers_category_{category_id}.{iteration_number}_HIV_EQ.json"
                    prompted_file_path = os.path.join(output_answer_dir, prompted_file_name)
                    with open(prompted_file_path, 'w') as f:
                        json.dump(prompts_list, f, indent=2)

                    print(f"Saved prompts for {model_name} (category={category_id}, iteration={iteration_number}) -> {prompted_file_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    questions_dir = os.path.join(script_dir, 'questions_files/')

    parser = argparse.ArgumentParser(description="Run HIV evaluation for one or more models.")
    parser.add_argument(
        "--model",
        nargs="+",
        default=["Llama"],
        help="Which model(s) to run, e.g. --model Llama, Meditron, Claude, Med42, NVLM, Llama-8B, Llama-1B, Gemini_2.5Pro, Gemma-3-27B, MedGemma-3-27B"
    )
    args = parser.parse_args()
    model_list = args.model
    
    # 1st step: get model answers
    system_prompt="You are a helpful, respectful and honest senior physician specializing in HIV. You are assisting a junior clinician answering medical questions. Keep your answers brief and clear. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    output_model_dir = obtain_answers_HIV(questions_dir=questions_dir, model_list=model_list, system_prompt= system_prompt)
   # output_model_dir = obtain_answers_HIV(questions_dir=questions_dir, model_list=model_list)
    print("Model inferences have been completed and saved to the output directory.")
    output_model_dir = '/cluster/home/gcardenal/HIV/medical_llm_evaluation/deploy_medical_llm_evaluation/model_answers/'
    # 2nd step: build or add GPT-based prompt for scoring
    prompt_model_answers(input_answer_dir=output_model_dir, model_list=model_list)