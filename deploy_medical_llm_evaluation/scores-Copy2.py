import os
import json
import openai
from openai import OpenAI
import pandas as pd
import re
import csv
import numpy as np

def get_GPT_scores(model_list,gpt4_api_key, model_answers_files_path):
    client = OpenAI(api_key=gpt4_api_key)
    
    for model in model_list:
        for category_id in [str(num) for num in range(1, 7)]:
            for iteration_number in range(1, 4):
                prompted_file_name = f"prompted_{model}_answers_category_{category_id}.{iteration_number}_HIV_EQ.json"
                prompted_files_path_model= os.path.join(model_answers_files_path, f'prompted_model_answers/{model}/')
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
                gpt_score_path = os.path.join(model_answers_files_path, f'results-GPT-score/{model}/')
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
                gpt_score_path_model = os.path.join(model_answers_files_path, f'results-GPT-score/{model}/')
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
    df.to_csv('./evaluation_results/raw_GPT4-score.csv', index=False)

def mean_and_std_gpt_score():
#Compute means and standard deviations from the raw data
    
    # Read the raw data
    df = pd.read_csv('./evaluation_results/raw_GPT4-score.csv')
    
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
    result_df.to_csv('./evaluation_results/overall_iterations_GPT4-score.csv', index=False)

def merge_final_df(f1_json_path):
    # Load data_records from the JSON file
    with open(f1_json_path, 'r') as f:
        data_records = json.load(f)

    df_f1 = pd.DataFrame(data_records)
    
    # Compute mean and standard deviation of all F1-related scores over iterations
    # Include synonyms and lemmatized synonyms columns in aggregation
    metrics = [
        'precision', 'recall', 'f1_score',
        'synonyms_precision', 'synonyms_recall', 'synonyms_f1',
        'synonyms_lemmatized_precision', 'synonyms_lemmatized_recall', 'synonyms_lemmatized_f1'
    ]
    
    grouped = df_f1.groupby(['nlp_model', 'model', 'category_id', 'question_index'])
    df_stats = grouped[metrics].agg(['mean', 'std']).reset_index()

    # Flatten MultiIndex columns
    df_stats.columns = [
        '_'.join(col).strip('_') if isinstance(col, tuple) else col 
        for col in df_stats.columns.values
    ]

    # Save the lexical semantics score CSV
    df_stats.to_csv('./evaluation_results/F1_lexical_semantics_score.csv', index=False)

    # Load gpt-scores
    df_existing = pd.read_csv('./evaluation_results/overall_iterations_GPT4-score.csv')
    
    df_stats['question_index'] = df_stats['question_index'].astype(int)
    df_existing['question_index'] = df_existing['question_index'].astype(int)
    df_stats['category_id'] = df_stats['category_id'].astype(str)
    df_existing['category_id'] = df_existing['category_id'].astype(str)
    
    # Merge DataFrames
    df_merged = pd.merge(df_existing, df_stats, on=['model', 'category_id', 'question_index'], how='left')

    # Check if the merged CSV file already exists
    merged_csv_path = './evaluation_results/evaluation_results_merged.csv'
    if os.path.exists(merged_csv_path):
        # Load existing CSV
        df_existing_merged = pd.read_csv(merged_csv_path)
        
        # Check if columns match
        if set(df_existing_merged.columns) == set(df_merged.columns):
            # Concatenate the new data with the existing file
            df_merged = pd.concat([df_existing_merged, df_merged], ignore_index=True)
        else:
            print("Column mismatch. Creating a new CSV file.")
    
    # Save merged DataFrame
    df_merged.to_csv(merged_csv_path, index=False)

    
if __name__ == "__main__":
    gpt4_api_key = "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"
    gpt4_base_url = "http://148.187.108.173:8080"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_answers_files_path = os.path.join(script_dir, 'model_answers/')
    model_list = ["Claude"]

    #Don't run without running f1_score.py
    f1_json_path = './evaluation_results/f1_results.json'
    
    if os.path.isfile(f1_json_path): 
        #get GPT scores
        get_GPT_scores(model_list=model_list,gpt4_api_key=gpt4_api_key, model_answers_files_path=model_answers_files_path)
        
        #obtain GPT-scores dataframe from json
        json_to_df(model_list=model_list, model_answers_files_path=model_answers_files_path)
        
        #obtain .csv with mean and std over iterations
        mean_and_std_gpt_score()
        
        merge_final_df(f1_json_path=f1_json_path) #obtain final dataset
    else:
        print(f"File {f1_json_path} does not exist, please generate F1 scores before running this script")
