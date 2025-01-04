import os
import json
import openai
from openai import OpenAI
import pandas as pd
import re

def prompt_to_rephrase_true_answer(path_to_true_answers, model_with_answers):
    input_dir = path_to_true_answers 
    prompt_intro = """
    Rephrase the following answer given in True Answer with clinical terms, do not rephrase Question
    """
    input_format = """
    Input
    Question : $question$
    True answer: $true_answer$
    """
    output_format = """
    Output Format
    Your rephrased answer should be provided in JSON format, as follows (don’t generate any other information):
    {
    "question": "The original question"
    "true_answer": "The original true answer"
    "true_answer_rephrased": "The true answer rephrased"
    }
    """
    for model in model_with_answers:
        for category_id in [str(num) for num in range(1, 7)]:
            for iteration_number in range(1, 4):
                # Read the answer file
                answer_file_name = f"{model}_answers_category_{category_id}.{iteration_number}_HIV_EQ.json"
                answer_file_path = os.path.join(input_dir, model, answer_file_name)
                with open(answer_file_path, 'r') as f:
                    answers_list = json.load(f)
                
                # Prepare the prompts
                prompts_list = []
                for item in answers_list:
                    true_answer = item['true_answer']
                    question = item['question']
                    
                    # Replace placeholders in input_format
                    input_text = input_format.replace('$true_answer$', true_answer)
                    input_text = input_text.replace('$question$', question)
                    
                    # Combine all parts to form the prompt
                    full_prompt = prompt_intro + input_text + output_format
                    
                    # Save the prompt in a dictionary (to be saved as JSON)
                    prompt_data = {
                        "prompt": full_prompt
                    }
                    prompts_list.append(prompt_data)
                
                # Save the prompts to the prompted_* file
                prompted_file_name = f"prompted_to_rephrase_True_answer_answers_category_{category_id}.{iteration_number}_HIV_EQ.json"
                script_dir = os.path.dirname(os.path.abspath(__file__))
                output_dir = os.path.join(script_dir, 'prompted_to_rephrase/')
                prompted_file_path = os.path.join(output_dir, prompted_file_name)
                with open(prompted_file_path, 'w') as f:
                    json.dump(prompts_list, f, indent=2)


def get_GPT_scores_rephrased(gpt4_api_key,input_dir):
    
    client = OpenAI(api_key=gpt4_api_key)
    
    for model in ["True_answer"]:
        for category_id in [str(num) for num in range(1, 7)]:
            for iteration_number in range(1, 4):
                prompted_file_name = f"prompted_to_rephrase_{model}_answers_category_{category_id}.{iteration_number}_HIV_EQ.json"
                prompted_file_path = os.path.join(input_dir, prompted_file_name)
                with open(prompted_file_path, 'r') as f:
                    prompts_list = json.load(f)
                
                # Prepare to collect the GPT-4 responses
                responses_list = []
                for prompt_data in prompts_list:
                    prompt_text = prompt_data['prompt']
                    
                    # Send the prompt to GPT-4 via OpenAI API
                    res = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "user",
                                "content": prompt_text
                            }
                        ],
                        stream=False,
                        temperature=0,
                    )
                    # Extract the assistant's reply
                    assistant_reply = res.choices[0].message.content.strip()
                    
                    # Save the reply in the responses_list
                    response_data = {
                        "response": assistant_reply
                    }
                    responses_list.append(response_data)
                
                # Save the responses to the GPT-score_* file
                rephrased_file_name = f"GPT-rephrase_true_answer_answers_category_{category_id}.{iteration_number}_HIV_EQ.json"
                script_dir = os.path.dirname(os.path.abspath(__file__))
                output_dir = os.path.join(script_dir, 'rephrased_true_answers/')
                rephrased_file_path = os.path.join(output_dir, rephrased_file_name)
                with open(rephrased_file_path, 'w') as f:
                    json.dump(responses_list, f, indent=2)

if __name__ == "__main__":
    gpt4_api_key = "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"
    gpt4_base_url = "http://148.187.108.173:8080"

    path_to_true_answers = '../model_answers/raw/'
    model_with_answers = ['Claude'] #just need a list with the string of one model from whose json files we can get the 'true_answer'

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'prompted_to_rephrase/')
    
    prompt_to_rephrase_true_answer(path_to_true_answers=path_to_true_answers, model_with_answers=model_with_answers)
    get_GPT_scores_rephrased(gpt4_api_key = gpt4_api_key, input_dir=output_dir)