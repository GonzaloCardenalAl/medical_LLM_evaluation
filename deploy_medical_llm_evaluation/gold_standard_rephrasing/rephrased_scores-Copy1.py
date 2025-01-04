import os
import json
import openai
from openai import OpenAI
import pandas as pd
import re
import csv
import numpy as np

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
                answer_file_name = f"GPT-rephrase_true_answer_answers_category_{category_id}.{iteration_number}_HIV_EQ.json"
                input_answer_model = os.path.join(input_answer_dir, model)
                print(f"input_answer_model {input_answer_model}")
                answer_file_path = os.path.join(input_answer_model, answer_file_name)
                with open(answer_file_path, 'r') as f:
                    answers_list = json.load(f)
                
                # Prepare the prompts
                prompts_list = []

                for item in answers_list:
                    # Extract and parse the JSON inside the "response" key
                    try:
                        response_data = json.loads(item['response'].strip("```json\n").strip("```"))
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in response: {e}")
                        continue
                
                    # Access the parsed response data
                    question_from_list = response_data.get('question', '')
                    model_answer = response_data.get('true_answer_rephrased', '')
                    true_answer = response_data.get('true_answer', '')
                
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
                output_dir = os.path.join(script_dir, f'prompted_rephrased/{model}/')
                os.makedirs(output_dir, exist_ok=True)
                prompted_file_name = f"prompted_{model}_answers_category_{category_id}.{iteration_number}_HIV_EQ.json"
                prompted_file_path = os.path.join(output_dir, prompted_file_name)
                with open(prompted_file_path, 'w') as f:
                    json.dump(prompts_list, f, indent=2)

def get_GPT_scores_rephrased(model_list,gpt4_api_key, model_answers_files_path):
    client = OpenAI(api_key=gpt4_api_key)
    
    for model in model_list:
        for category_id in [str(num) for num in range(1, 7)]:
            for iteration_number in range(1, 4):
                prompted_file_name = f"prompted_{model}_answers_category_{category_id}.{iteration_number}_HIV_EQ.json"
                prompted_files_path_model= os.path.join(model_answers_files_path, f'prompted_rephrased/{model}/')
                prompted_file_path = os.path.join(prompted_files_path_model, prompted_file_name)
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
                gpt_score_file_name = f"GPT-score_{model}_answers_category_{category_id}.{iteration_number}_HIV_EQ.json"
                gpt_score_path = os.path.join(model_answers_files_path, 'results-GPT-score-rephrased/')
                os.makedirs(gpt_score_path, exist_ok=True)
                gpt_score_file_path = os.path.join(gpt_score_path, gpt_score_file_name)
                with open(gpt_score_file_path, 'w') as f:
                    json.dump(responses_list, f, indent=2)



def json_to_df(model_list, model_answers_files_path):
    data_records = []
    
    for model in model_list:
        for category_id in [str(num) for num in range(1, 7)]:
            for iteration_number in range(1, 4):
                gpt_score_file_name = f"GPT-score_{model}_answers_category_{category_id}.{iteration_number}_HIV_EQ.json"
                gpt_score_path_model = os.path.join(model_answers_files_path, 'results-GPT-score-rephrased/')
                gpt_score_file_path = os.path.join(gpt_score_path_model, gpt_score_file_name)
                with open(gpt_score_file_path, 'r') as f:
                    responses_list = json.load(f)
                
                for idx, response_data in enumerate(responses_list):
                    assistant_reply = response_data['response']
                    # Parse the JSON in the assistant_reply
                    try:
                        # Find the JSON string in the reply
                        json_text = assistant_reply.strip()
                        # Remove any text before or after the JSON
                        json_match = re.search(r'\{.*\}', json_text, re.DOTALL)
                        if json_match:
                            json_str = json_match.group()
                            scores = json.loads(json_str)
                            
                            # Extract the scores
                            q1 = scores.get('question 1', None)
                            q2 = scores.get('question 2', None)
                            q3 = scores.get('question 3', None)
                            q4 = scores.get('question 4', None)
                            q5 = scores.get('question 5', None)
                            overall_score = scores.get('overall score', None)
                            
                            # Convert scores to numeric if possible
                            q1 = float(q1) if q1 is not None else None
                            q2 = float(q2) if q2 is not None else None
                            q3 = float(q3) if q3 is not None else None
                            q4 = float(q4) if q4 is not None else None
                            q5 = float(q5) if q5 is not None else None
                            overall_score = float(overall_score) if overall_score is not None else None
                            
                            # Calculate overall_score_wo_Q4&5
                            q_scores = [q1, q2, q3]
                            q_scores = [q for q in q_scores if q is not None]
                            if q_scores:
                                overall_score_wo_Q4and5 = sum(q_scores) / len(q_scores)
                            else:
                                overall_score_wo_Q4and5 = None
                            
                            # Append the record
                            data_records.append({
                                'file_name': gpt_score_file_name,
                                'model': model,
                                'category_id': category_id,
                                'iteration_number': iteration_number,  # Adjusted to start from 2
                                'question_index': idx,  # Assuming question index starts from 1
                                'GPT1': q1,
                                'GPT2': q2,
                                'GPT3': q3,
                                'GPT4': q4,
                                'GPT5': q5,
                                'GPT_overall_score': overall_score,
                                'GPT_overall_score_wo_Q4&5': overall_score_wo_Q4and5
                            })
                        else:
                            print(f"Could not find JSON in reply for {gpt_score_file_name}, question {idx+1}")
                    except Exception as e:
                        print(f"Error parsing JSON for {gpt_score_file_name}, question {idx+1}: {e}")
                        continue
    
    # Create the DataFrame
    df = pd.DataFrame(data_records)
    
    # Save the raw DataFrame to CSV
    df.to_csv('./rephrased_raw_GPT4-score.csv', index=False)

def mean_and_std_gpt_score():
#Compute means and standard deviations from the raw data
    
    # Read the raw data
    df = pd.read_csv('./rephrased_raw_GPT4-score.csv')
    
    # Use 'category_id' in groupings
    grouped = df.groupby(['model', 'category_id', 'question_index'])
    
    # Define columns to aggregate
    agg_columns = ['GPT1', 'GPT2', 'GPT3', 'GPT4', 'GPT5', 'GPT_overall_score', 'GPT_overall_score_wo_Q4&5']
    agg_columns_no_overall  = ['GPT1', 'GPT2', 'GPT3', 'GPT4', 'GPT5']
    agg_columns_no_overall_noQ4_5 = ['GPT1', 'GPT2', 'GPT3']
    
    # Compute mean and standard deviation over the iterations for each question
    mean_df = grouped[agg_columns].mean().reset_index()
    std_df = grouped[agg_columns_no_overall].std().reset_index()
    std_df_noQ4_5 = grouped[agg_columns_no_overall_noQ4_5].std().reset_index()
    
    #Compute overall std
    std_df['GPT_overall_score_std'] = std_df[agg_columns_no_overall].mean(axis=1)
    std_df['GPT_overall_score_wo_Q4&5_std'] = std_df_noQ4_5[agg_columns_no_overall_noQ4_5].mean(axis=1)
    
    # Compute the number of iterations per group
    iteration_counts = grouped['iteration_number'].count().reset_index(name='num_iterations')
    
    # Rename columns for clarity
    mean_df = mean_df.rename(columns={col: col + '_mean' for col in agg_columns})
    std_df = std_df.rename(columns={col: col + '_std' for col in agg_columns_no_overall})
    
    
    # Merge mean, std, and iteration_counts DataFrames
    result_df = iteration_counts.merge(mean_df, on=['model', 'category_id', 'question_index'])
    result_df = result_df.merge(std_df, on=['model', 'category_id', 'question_index'])
    
    # Validate the number of iterations
    expected_iterations = 3
    incomplete_groups = result_df[result_df['num_iterations'] < expected_iterations]
    if not incomplete_groups.empty:
        print("Warning: Some groups have fewer than expected iterations.")
        print(incomplete_groups[['model', 'category_id', 'question_index', 'num_iterations']])
    
    # Save the result DataFrame to CSV
    result_df.to_csv('./rephrased_averaged_GPT4-score.csv', index=False)

def merge_final_df(f1_json_path):
        # Load data_records from the JSON file
    with open(f1_json_path, 'r') as f:
        data_records = json.load(f)

    df_f1 = pd.DataFrame(data_records)
    
    # Compute mean and standard deviation of precision, recall, and F1 scores over iterations
    grouped = df_f1.groupby(['nlp_model', 'model', 'category_id', 'question_index'])
    df_stats = grouped[['precision', 'recall', 'f1_score']].agg(['mean', 'std']).reset_index()
    
    # Flatten MultiIndex columns
    df_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in df_stats.columns.values]
    df_stats.to_csv('./rephrased-F1_lexical_semantics_score.csv', index=False)
    
    # Load gpt-scores
    df_existing = pd.read_csv('./rephrased_averaged_GPT4-score.csv')
    
    df_stats['question_index'] = df_stats['question_index'].astype(int)
    df_existing['question_index'] = df_existing['question_index'].astype(int)
    df_stats['category_id'] = df_stats['category_id'].astype(str)
    df_existing['category_id'] = df_existing['category_id'].astype(str)
    
    # Merge DataFrames
    df_merged = pd.merge(df_existing, df_stats, on=['model', 'category_id', 'question_index'], how='left')

    # Save merged DataFrame
    df_merged.to_csv('./rephrased-evaluation_results_merged.csv', index=False)

    
if __name__ == "__main__":
    gpt4_api_key = "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"
    gpt4_base_url = "http://148.187.108.173:8080"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_answers_files_path = script_dir
    model_list = ["rephrased_true_answers"]

    #Don't run without running f1_score.py
    f1_json_path = './rephrased-f1_results.json'
    
    if os.path.isfile(f1_json_path): 

        #Add GPT scoring prompt to rephrased true answers
        print(f"model_answers_files_path {model_answers_files_path}")
        prompt_model_answers(input_answer_dir=model_answers_files_path, model_list=model_list)
    
        #get GPT scores
        get_GPT_scores_rephrased(model_list=model_list,gpt4_api_key=gpt4_api_key, model_answers_files_path=model_answers_files_path)
        
        #obtain GPT-scores dataframe from json
        json_to_df(model_list=model_list, model_answers_files_path=model_answers_files_path)
        
        #obtain .csv with mean and std over iterations
        mean_and_std_gpt_score()
        
        merge_final_df(f1_json_path=f1_json_path) #obtain final dataset

    else:
        print(f"File {f1_json_path} does not exist, please generate F1 scores before running this script")
