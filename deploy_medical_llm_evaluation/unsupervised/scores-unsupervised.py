import os
import json
import openai
from openai import OpenAI
import pandas as pd
import re
import csv
import numpy as np

def get_GPT_scores(model_list, gpt4_api_key, model_answers_files_path):
    client = OpenAI(api_key=gpt4_api_key)
    for model in model_list:
        if model in ["Llama", "Meditron"]:
            subfolders = [f"{model}_api", f"{model}_cluster"]
        else:
            subfolders = [model]

        for subfolder in subfolders:
            prompted_files_path_model = os.path.join(
                model_answers_files_path, 'prompted_unsupervised_model_answers', subfolder
            )
            
            
            if not os.path.isdir(prompted_files_path_model):
                print(f"Missing directory: {prompted_files_path_model}")
                continue

            for category_id in [str(num) for num in range(1, 7)]:
                for iteration_number in range(1, 6):
                    prompted_file_name = (
                        f"unsupervised_prompted_{model}_answers_category_{category_id}."
                        f"{iteration_number}_HIV_EQ.json"
                    )
                    prompted_file_path = os.path.join(prompted_files_path_model, prompted_file_name)
                    
                    if not os.path.exists(prompted_file_path):
                        print(f"Missing prompted file: {prompted_file_path}")
                        continue
                    
                    with open(prompted_file_path, 'r') as f:
                        prompts_list = json.load(f)
                    
                    responses_list = []
                    for prompt_data in prompts_list:
                        prompt_text = prompt_data['prompt']
                        res = client.chat.completions.create(
                            model="gpt-4o-2024-08-06",
                            messages=[
                                {"role": "user", "content": prompt_text}
                            ],
                            stream=False,
                            temperature=0,
                        )
                        assistant_reply = res.choices[0].message.content.strip()
                        response_data = {"response": assistant_reply}
                        responses_list.append(response_data)
                    
                    gpt_score_path = os.path.join(
                        model_answers_files_path, 'results-unsupervised-GPT-score', subfolder
                    )
                    os.makedirs(gpt_score_path, exist_ok=True)
                    
                    gpt_score_file_name = (
                        f"unsupervised-GPT-score_{model}_answers_category_{category_id}."
                        f"{iteration_number}_HIV_EQ.json"
                    )
                    gpt_score_file_path = os.path.join(gpt_score_path, gpt_score_file_name)
                    
                    with open(gpt_score_file_path, 'w') as f:
                        json.dump(responses_list, f, indent=2)
                    
                    print(f"Saved Unsupervised GPT-4 scores to: {gpt_score_file_path}")

                    
def clean_gpt_json_response(response_text):
    # Strip code block markers
    cleaned = re.sub(r'^```json\s*|\s*```$', '', response_text.strip(), flags=re.MULTILINE)
    # Replace raw newlines within string values
    cleaned = re.sub(r'(?<!\\)\n', r'\\n', cleaned)
    return json.loads(cleaned)

def json_to_df(model_list, model_answers_files_path):
    data_records = []
    
    for model in model_list:
        # Determine the subfolder names for GPT-score files
        if model in ["Llama", "Meditron"]:
            score_subfolders = [f"{model}_api", f"{model}_cluster"]
            # For the raw answers, prepend "raw/" to the same names:
            answer_subfolders = [f"raw/{model}_api", f"raw/{model}_cluster"]
        else:
            score_subfolders = [model]
            answer_subfolders = [f"raw/{model}"]
        
        # Iterate over the corresponding subfolders in both directories
        # We assume the order of score_subfolders and answer_subfolders is aligned.
        for score_subfolder, answer_subfolder in zip(score_subfolders, answer_subfolders):
            # Path for GPT-score files
            gpt_score_path_model = os.path.join(
                model_answers_files_path, 'results-unsupervised-GPT-score', score_subfolder
            )
            if not os.path.isdir(gpt_score_path_model):
                continue
            
            for category_id in [str(num) for num in range(1, 7)]:
                # For GPT-scores, iterate over iterations 1 to 3
                for iteration_number in range(1, 6):
                    gpt_score_file_name = (
                        f"unsupervised-GPT-score_{model}_answers_category_{category_id}."
                        f"{iteration_number}_HIV_EQ.json"
                    )
                    gpt_score_file_path = os.path.join(gpt_score_path_model, gpt_score_file_name)
                    
                    if not os.path.exists(gpt_score_file_path):
                        continue
                    
                    # Load the GPT-score responses list.
                    try:
                        with open(gpt_score_file_path, 'r') as f:
                            responses_list = json.load(f)
                    except Exception as e:
                        print(f"Error loading GPT-score file {gpt_score_file_path}: {e}")
                        continue

                    # Now, also try to load the corresponding answer file.
                    answer_file_name = f"{model}_answers_category_{category_id}.{iteration_number}_HIV_EQ.json"
                    answer_file_path = os.path.join(model_answers_files_path, answer_subfolder, answer_file_name)
                    if os.path.exists(answer_file_path):
                        try:
                            with open(answer_file_path, 'r') as f:
                                answers_list = json.load(f)
                        except Exception as e:
                            print(f"Error loading answer file {answer_file_path}: {e}")
                            answers_list = None
                    else:
                        # If the answer file does not exist, we set answers_list to None
                        answers_list = None
                        print(f"Answer file not found: {answer_file_path}. Proceeding without answer info.")
                    
                    # Iterate over each question response in the GPT-score file.
                    for idx, response_data in enumerate(responses_list):
                        assistant_reply = response_data.get('response', '')
                        try:
                            
                            json_text = assistant_reply.strip()
                            # Attempt to extract the JSON block from the reply.
                            json_match = re.search(r'\{.*\}', json_text, re.DOTALL)
                            if json_match:
                                json_str = json_match.group()
                                scores = json.loads(json_str)
                                
                                # Get scores for each question.
                                q1 = float(scores.get('question 1', 0))
                                q2 = float(scores.get('question 2', 0))
                                q3 = float(scores.get('question 3', 0))
                                q4 = float(scores.get('question 4', 0))
                                q5 = float(scores.get('question 5', 0))
                                overall_score = float(scores.get('overall score', 0))
                                overall_score_wo_Q4and5 = np.mean([q1, q2, q3])
                            else:
                                print(f"No JSON found in GPT reply for {gpt_score_file_path}, Q#{idx+1}")
                                q1 = q2 = q3 = q4 = q5 = overall_score = overall_score_wo_Q4and5 = 0
                        except Exception as e:
                            print(f"Error parsing JSON in GPT reply for {gpt_score_file_path}, Q#{idx+1}: {e}")
                            continue
                        
                        # Try to get the additional info from the corresponding answer file.
                        if answers_list is not None and idx < len(answers_list):
                            answer_item = answers_list[idx]
                            question_text = answer_item.get('question', None)
                            model_answer = answer_item.get('answer', None)
                        else:
                            question_text = model_answer = gold_answer = None
                        
                        # Append the record to our list.
                        data_records.append({
                            'file_name': gpt_score_file_name,
                            'model': model,
                            'subfolder': score_subfolder,
                            'category_id': category_id,
                            'iteration_number': iteration_number,
                            'question_index': idx,
                            'question': question_text,
                            'model_answer': model_answer,
                            'GPT1': q1,
                            'GPT2': q2,
                            'GPT3': q3,
                            'GPT4': q4,
                            'GPT5': q5,
                            'GPT_overall_score': overall_score,
                            'GPT_overall_score_wo_Q4&5': overall_score_wo_Q4and5
                        })
    # Create a DataFrame and save to CSV.
    df = pd.DataFrame(data_records)
    os.makedirs('./evaluation_results', exist_ok=True)
    csv_path = './evaluation_results/unsupervised_raw_GPT4-score.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved GPT-4 scores with answer details to CSV: {csv_path}")


def mean_and_std_gpt_score_over_iterations():
    """
    Compute means and standard deviations for GPT-4 scores over the 3 iterations,
    grouping by (model, subfolder, category_id, question_index).
    Saves results to 'averaged_over_iterations_GPT4-score.csv'.
    """
    df = pd.read_csv('./evaluation_results/unsupervised_raw_GPT4-score.csv')
    
    # Group by (model, subfolder, category_id, question_index)
    grouped = df.groupby(['model', 'subfolder', 'category_id', 'question_index'])
    
    # Columns to aggregate
    agg_columns = ['GPT1', 'GPT2', 'GPT3', 'GPT4', 'GPT5', 'GPT_overall_score', 'GPT_overall_score_wo_Q4&5']

    # Compute mean and std over the iterations
    mean_df = grouped[agg_columns].mean().reset_index()
    std_df = grouped[agg_columns].std().reset_index()
    
    # Compute the number of iterations per group
    iteration_counts = grouped['iteration_number'].count().reset_index(name='num_iterations')
    
    # Rename columns for clarity
    mean_df = mean_df.rename(columns={col: col + '_mean' for col in agg_columns})
    std_df = std_df.rename(columns={col: col + '_std' for col in agg_columns})
    
    # Merge mean, std, and iteration_counts
    result_df = iteration_counts.merge(mean_df, on=['model','subfolder','category_id','question_index'])
    result_df = result_df.merge(std_df, on=['model','subfolder','category_id','question_index'])
    
    # Validate the number of iterations
    expected_iterations = 5
    incomplete_groups = result_df[result_df['num_iterations'] < expected_iterations]
    if not incomplete_groups.empty:
        print("Warning: Some groups have fewer than expected iterations.")
        print(incomplete_groups[['subfolder', 'category_id', 'question_index', 'num_iterations']])
    
    # Save the result to CSV
    result_df.to_csv('./evaluation_results/unsupervised/unsupervised_averaged_over_iterations_GPT4-score.csv', index=False)
    print("Saved aggregated GPT-4 scores over iterations: ./evaluation_results/unsupervised/unsupervised_averaged_over_iterations_GPT4-score.csv")


def merge_final_df_over_iterations(f1_json_path):
    """
    1) Loads the F1 data (with synonyms expansions) from f1_json_path.
    2) Aggregates (mean, std) across the 3 iterations per question_index.
    3) Merges with GPT-4 scores aggregated by question_index in
       'averaged_over_iterations_GPT4-score.csv'.
    4) Saves the final CSV with all columns (GPT & F1) as 'evaluation_results_merged_over_iterations.csv'.
    """
    with open(f1_json_path, 'r') as f:
        data_records = json.load(f)

    df_f1 = pd.DataFrame(data_records)
    
    # ------------------------------------------------------------------
    # Add all your synonyms-based columns here:
    # ------------------------------------------------------------------
    metrics = [
        'precision', 'recall', 'f1_score',
        'synonyms_precision_dict', 'synonyms_recall_dict', 'synonyms_f1_dict',
        'synonyms_precision_snomed', 'synonyms_recall_snomed', 'synonyms_f1_snomed',
        'synonyms_precision_wn', 'synonyms_recall_wn', 'synonyms_f1_wn',
        'synonyms_lemmatized_precision_dict', 'synonyms_lemmatized_recall_dict', 'synonyms_lemmatized_f1_dict',
        'synonyms_lemmatized_precision_snomed', 'synonyms_lemmatized_recall_snomed', 'synonyms_lemmatized_f1_snomed',
        'synonyms_lemmatized_precision_wn', 'synonyms_lemmatized_recall_wn', 'synonyms_lemmatized_f1_wn'
    ]

    # Group by these columns
    grouped = df_f1.groupby(['nlp_model', 'model', 'model_subfolder','category_id', 'question_index'])
    df_stats = grouped[metrics].agg(['mean', 'std']).reset_index()

    # Flatten the multi-level columns (e.g., ("precision", "mean") -> "precision_mean")
    df_stats.columns = [
        '_'.join(col).strip('_') if isinstance(col, tuple) else col
        for col in df_stats.columns.values
    ]
    
    # Rename column for consistent merging with GPT
    df_stats.rename(columns={'model_subfolder': 'subfolder'}, inplace=True)

    # Save the lexical semantics score CSV
    df_stats.to_csv('./evaluation_results/F1_lexical_semantics_score_over_iterations.csv', index=False)

    # Load GPT scores aggregated over iterations
    df_existing = pd.read_csv('./evaluation_results/unsupervised/unsupervised_averaged_over_iterations_GPT4-score.csv')
    
    # Ensure consistent dtypes for merging
    df_stats['question_index'] = df_stats['question_index'].astype(int)
    df_existing['question_index'] = df_existing['question_index'].astype(int)
    df_stats['category_id'] = df_stats['category_id'].astype(str)
    df_existing['category_id'] = df_existing['category_id'].astype(str)

    # Merge
    df_merged = pd.merge(
        df_existing, df_stats,
        on=['model','subfolder', 'category_id', 'question_index'],
        how='left'
    )

    merged_csv_path = './evaluation_results/unsupervised/unsupervised_evaluation_results_merged_over_iterations.csv'
    df_merged.to_csv(merged_csv_path, index=False)
    print(f"Saved merged CSV with iteration-level data: {merged_csv_path}")


def mean_and_std_gpt_score_over_questions():
    """
    1) Averages GPT scores over all questions within each category & iteration
       => saves to 'averaged_over_questions_GPT4-score.csv'
    2) Then computes mean + std across iterations for each category
       => also saved in that same file.
    """
    df_path = './evaluation_results/raw_GPT4-score.csv'
    if not os.path.exists(df_path):
        print(f"No file found at {df_path}")
        return
    
    df = pd.read_csv(df_path)
    
    # ----------------------------------------------------
    # STEP 1: AVERAGE GPT SCORES OVER QUESTIONS
    # Group by (model, subfolder, category_id, iteration_number)
    # ----------------------------------------------------
    group_cols_1 = ['model', 'subfolder', 'category_id', 'iteration_number']
    agg_columns = [
        'GPT1', 'GPT2', 'GPT3', 'GPT4', 'GPT5',
        'GPT_overall_score', 'GPT_overall_score_wo_Q4&5'
    ]
    
    df_questions_mean = (
        df.groupby(group_cols_1)[agg_columns]
        .mean()
        .reset_index()
    )
    
    # ----------------------------------------------------
    # STEP 2: MEAN + STD OVER ITERATIONS
    # Now group by (model, subfolder, category_id) to average across iteration_number
    # ----------------------------------------------------
    group_cols_2 = ['model', 'subfolder', 'category_id']
    
    df_iter_mean = (
        df_questions_mean
        .groupby(group_cols_2)[agg_columns]
        .mean()
        .reset_index()
    )
    df_iter_mean = df_iter_mean.rename(
        columns={col: col + '_mean' for col in agg_columns}
    )
    
    df_iter_std = (
        df_questions_mean
        .groupby(group_cols_2)[agg_columns]
        .std()
        .reset_index()
    )
    df_iter_std = df_iter_std.rename(
        columns={col: col + '_std' for col in agg_columns}
    )
    
    # Merge the mean + std
    df_iterations = pd.merge(df_iter_mean, df_iter_std, on=group_cols_2, how='left')
    
    # Save final CSV
    mean_over_iter_csv = './evaluation_results/unsupervised/unsupervised_averaged_over_questions_GPT4-score.csv'
    df_iterations.to_csv(mean_over_iter_csv, index=False)
    print(f"Saved GPT scores over questions & iterations: {mean_over_iter_csv}")


def merge_final_df_over_questions(f1_json_path):
    """
    1) Loads the detailed F1 JSON from f1_json_path.
    2) Averages over questions => Then mean + std over iterations.
    3) Merges with the newly created 'averaged_over_questions_GPT4-score.csv'.
    """
    if not os.path.exists(f1_json_path):
        print(f"No file found at {f1_json_path}")
        return
    
    # Load data_records from the JSON file
    with open(f1_json_path, 'r') as f:
        data_records = json.load(f)

    df_f1 = pd.DataFrame(data_records)
    
    # ------------------------------------------------------------------
    # Include all synonyms-based columns:
    # ------------------------------------------------------------------
    metrics = [
        'precision', 'recall', 'f1_score',
        'synonyms_precision_dict', 'synonyms_recall_dict', 'synonyms_f1_dict',
        'synonyms_precision_snomed', 'synonyms_recall_snomed', 'synonyms_f1_snomed',
        'synonyms_precision_wn', 'synonyms_recall_wn', 'synonyms_f1_wn',
        'synonyms_lemmatized_precision_dict', 'synonyms_lemmatized_recall_dict', 'synonyms_lemmatized_f1_dict',
        'synonyms_lemmatized_precision_snomed', 'synonyms_lemmatized_recall_snomed', 'synonyms_lemmatized_f1_snomed',
        'synonyms_lemmatized_precision_wn', 'synonyms_lemmatized_recall_wn', 'synonyms_lemmatized_f1_wn'
    ]
    
    # ----------------------------------------------------
    # STEP 1: AVERAGE OVER QUESTIONS (i.e. question_index)
    # ----------------------------------------------------
    # Group by (nlp_model, model, model_subfolder, category_id, iteration_number)
    df_f1_questions_mean = (
        df_f1
        .groupby(['nlp_model','model','model_subfolder','category_id','iteration_number'])[metrics]
        .mean()
        .reset_index()
    )
    
    # ----------------------------------------------------
    # STEP 2: MEAN + STD OVER ITERATIONS
    # (nlp_model, model, model_subfolder, category_id)
    # ----------------------------------------------------
    df_f1_iter_mean = (
        df_f1_questions_mean
        .groupby(['nlp_model','model','model_subfolder','category_id'])[metrics]
        .mean()
        .reset_index()
    )
    # rename columns to e.g. "precision_iter_mean"
    df_f1_iter_mean = df_f1_iter_mean.rename(
        columns={col: col + '_iter_mean' for col in metrics}
    )
    
    df_f1_iter_std = (
        df_f1_questions_mean
        .groupby(['nlp_model','model','model_subfolder','category_id'])[metrics]
        .std()
        .reset_index()
    )
    df_f1_iter_std = df_f1_iter_std.rename(
        columns={col: col + '_iter_std' for col in metrics}
    )
    
    # Rename folder for consistent merge with GPT:
    df_f1_iter_mean.rename(columns={'model_subfolder': 'subfolder'}, inplace=True)
    df_f1_iter_std.rename(columns={'model_subfolder': 'subfolder'}, inplace=True)
    
    # Merge mean + std
    df_f1_iterations = pd.merge(
        df_f1_iter_mean, df_f1_iter_std,
        on=['nlp_model','model','subfolder','category_id'],
        how='left'
    )
    
    # Save iteration-level F1
    df_f1_iterations.to_csv('./evaluation_results/F1_overall_results_mean_over_questions.csv', index=False)
    print("Saved synonyms-based F1 scores (means & std over questions & iterations).")
    
    # ----------------------------------------------------
    # Combine with GPT-4 question-based, iteration-averaged results
    # ----------------------------------------------------
    df_gpt_iter = pd.read_csv('./evaluation_results/unsupervised/unsupervised_averaged_over_questions_GPT4-score.csv')
    
    df_f1_iterations['category_id'] = df_f1_iterations['category_id'].astype(str)
    df_gpt_iter['category_id'] = df_gpt_iter['category_id'].astype(str)
    
    df_merged = pd.merge(
        df_f1_iterations, df_gpt_iter,
        on=['model','subfolder','category_id'],
        how='left'
    )
    
    merged_csv_path = './evaluation_results/unsupervised/unsupervised_evaluation_results_merged_mean_over_questions.csv'
    df_merged.to_csv(merged_csv_path, index=False)
    print(f"Saved merged dataset (F1 & GPT) over questions to: {merged_csv_path}")


if __name__ == "__main__":
    gpt4_api_key = ""  #Add your token for the OpenAI API
    gpt4_base_url = "http://148.187.108.173:8080"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_answers_files_path = os.path.join(script_dir, 'model_answers')
    
    #model_list = ["Llama", "Meditron", "NVLM", "Med42", "Claude", "Llama-8B", "Llama-1B", "Gemini_2.5Pro", "Gemma-3-27B"]
    #model_list = ["Gemini_2.5Pro", "Gemma-3-27B"]
    model_list = ["Gemma-3-27B", "MedGemma-3-27B"]
    get_GPT_scores(model_list=model_list, gpt4_api_key=gpt4_api_key, model_answers_files_path=model_answers_files_path)
    f1_json_path = './evaluation_results/f1_results.json'
    
    if os.path.isfile(f1_json_path): 
        
        model_list = ["Llama", "Meditron", "NVLM", "Med42", "Claude", "Llama-8B", "Llama-1B", "Gemini_2.5Pro", "Gemma-3-27B", "MedGemma-3-27B"]
        json_to_df(model_list=model_list, model_answers_files_path=model_answers_files_path)
        
        # 1. GPT means & std over iterations
        mean_and_std_gpt_score_over_iterations()
        # 2. Merge iteration-level F1 with GPT
        #merge_final_df_over_iterations(f1_json_path=f1_json_path)
        
        # 3. GPT means & std over questions
        mean_and_std_gpt_score_over_questions()
        # 4. Merge question-level F1 with GPT
        #merge_final_df_over_questions(f1_json_path=f1_json_path)
        
    else:
        print(f"File {f1_json_path} does not exist, please generate F1 scores before running this script.")