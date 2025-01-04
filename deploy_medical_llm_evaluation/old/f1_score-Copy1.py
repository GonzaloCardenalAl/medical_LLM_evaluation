# Import necessary libraries
import os
import re
import json
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import spacy
import json

# Define the cleaning function
def clean_text(text):
    # Replace newline characters with a space and remove asterisks
    text = text.replace('\n', ' ').replace('*', '')
    # Remove any extra spaces that may have resulted
    text = ' '.join(text.split())
    return text

def parsing_and_computing_f1(input_answer_dir,model_list, rephrased = False):
    # List of scispaCy models to use
    nlp_models = {
        'en_core_sci_lg': spacy.load('en_core_sci_lg'),
    #    'en_core_sci_scibert': spacy.load('en_core_sci_scibert')
    }
    
    # Initialize data collection
    data_records = []
    category_ids = [str(num) for num in range(1, 7)]  
    iteration_numbers = [1, 2, 3]
    
    # Process each file with each scispaCy model
    for nlp_name, nlp in nlp_models.items():
        print(f"Processing with model: {nlp_name}")
        for model in model_list:
            for category_id in category_ids:
                for iteration_number in iteration_numbers:
                    # Construct file names
                    answer_file_name = f"{model}_answers_category_{category_id}.{iteration_number}_HIV_EQ.json"
                    input_answer_model = os.path.join(input_answer_dir, f"raw/{model}/")
                    file_path = os.path.join(input_answer_model, answer_file_name)
                    
                    # Check if the file exists
                    if not os.path.exists(file_path):
                        print(f"File not found: {file_path}")
                        continue
                    
                    # Load JSON data
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Process each answer pair
                    for idx, item in enumerate(data):
                        true_answer = item.get('true_answer', '')
                        answer = item.get('answer', '')
                        question_index = idx
    
                        # Clean the 'true_answer' and 'answer' strings
                        true_answer_clean = clean_text(true_answer)
                        answer_clean = clean_text(answer)
    
                        # Extract entities using scispaCy
                        doc_true = nlp(true_answer_clean)
                        doc_answer = nlp(answer_clean)
    
                        # Proceed with entity extraction
                        entities_true = set(ent.text.lower() for ent in doc_true.ents)
                        entities_answer = set(ent.text.lower() for ent in doc_answer.ents)
    
                        # Compute precision, recall, and F1
                        if entities_answer:
                            intersection = entities_true & entities_answer
                            precision = len(intersection) / len(entities_answer) if len(entities_answer) > 0 else 0.0
                            recall = len(intersection) / len(entities_true) if len(entities_true) > 0 else 0.0
                            if precision + recall > 0:
                                f1 = 2 * (precision * recall) / (precision + recall)
                            else:
                                f1 = 0.0
                        else:
                            precision = 0.0
                            recall = 0.0
                            f1 = 0.0
    
                        # Append results to data_records
                        data_records.append({
                            'nlp_model': nlp_name,
                            'model': model,
                            'category_id': category_id,
                            'iteration_number': iteration_number,
                            'question_index': question_index,
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1
                        })
    
    # Save data_records to a JSON file
    with open('./evaluation_results/f1_results.json', 'w') as f:
        json.dump(data_records, f, indent=4)

if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_answers_files_path = os.path.join(script_dir, '/model_answers/')
    model_list = ["Claude"]
    
    # Run the F1 score calculation
    parsing_and_computing_f1(model_answers_files_path, model_list)