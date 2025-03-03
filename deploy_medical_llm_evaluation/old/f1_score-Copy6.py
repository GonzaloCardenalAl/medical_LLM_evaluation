import os
import re
import json
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import spacy
import json

# Additional imports for synonyms and lemmatization
from owlready2 import *
from owlready2.pymedtermino2 import *
from owlready2.pymedtermino2.umls import *
import stanza
import requests

# Configuration for UMLS
UMLS_API_KEY = '165ce1dc-cc57-4526-8e52-b7c4f7df2085'
DOWNLOAD_URL = 'https://download.nlm.nih.gov/umls/kss/2024AB/umls-2024AB-metathesaurus-full.zip'
DESTINATION_PATH = '/cluster/scratch/gcardenal/LLM_models/pymedtermino_sql/umls-2024AB-metathesaurus-full.zip'

def download_umls_file(api_key, download_url, destination_path):
    # Append API key to the URL
    url_with_api_key = f"https://uts-ws.nlm.nih.gov/download?url={download_url}&apiKey={api_key}"
    
    # Make the GET request
    response = requests.get(url_with_api_key, stream=True)
    if response.status_code == 200:
        # Save the file to the specified path
        with open(destination_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"File downloaded successfully: {destination_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
        print(f"Response: {response.text}")

# Load stanza pipeline once for efficiency
stanza.download('en', package='mimic', processors='tokenize,pos,lemma')
biomed_nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma', package='mimic')

# Define the path to the PyMedTermino database
pym_db_path = "/cluster/scratch/gcardenal/LLM_models/pymedtermino_sql/pym.sqlite3_batch"

# Check if the PyMedTermino SQLite database already exists
if not os.path.exists(pym_db_path):
    print("PyMedTermino SQLite database not found. Will attempt to create it.")
    # Check if the UMLS Metathesaurus ZIP file exists
    if not os.path.exists(DESTINATION_PATH):
        print(f"{DESTINATION_PATH} does not exist. Downloading...")
        download_umls_file(UMLS_API_KEY, DOWNLOAD_URL, DESTINATION_PATH)
    else:
        print(f"UMLS Metathesaurus file {DESTINATION_PATH} already exists. Skipping download.")
    
    # Create and import UMLS data
    default_world.set_backend(filename=pym_db_path)
    import_umls(DESTINATION_PATH, terminologies=["ICD10", "SNOMEDCT_US", "CUI"])
    default_world.save()
else:
    # If it exists, just set the backend without importing
    print("PyMedTermino SQLite database already exists. Skipping import.")
    default_world.set_backend(filename=pym_db_path)

# Define the cleaning function
def clean_text(text):
    # Replace newline characters with a space and remove asterisks
    text = text.replace('\n', ' ').replace('*', '')
    # Remove any extra spaces that may have resulted
    text = ' '.join(text.split())
    return text

file_path_to_dic =  "./synonyms_dictionary.json"
with open(file_path_to_dic, 'r') as file:
        synonyms_dict = json.load(file)
     
def get_synonyms(word, synonyms_dict):
    """
    Get a set of synonyms from the dictionary created for a given word.
    """
    if word in synonyms_dict:
        return synonyms_dict[word] 
    else:
        return []

# Function to compute f1 with synonyms
def compute_f1_with_synonyms(entities_true, entities_answer, synonyms_dict):
    """
    Compute precision, recall, and F1 score using synonym-based intersection.
    
    Args:
        entities_true (set): Set of entities from the true sentence.
        entities_answer (set): Set of entities from the predicted sentence.
        
    Returns:
        precision, recall, f1: Computed scores.
    """
    def expand_with_synonyms(entities, synonyms_dict):
        expanded = {}
        for entity in entities:
            synonyms = get_synonyms(entity,synonyms_dict)  
            # Include the original entity as well, for direct matches
            expanded[entity] = set(synonyms + [entity])
        return expanded

    # Expand entities with synonyms
    true_synonyms = expand_with_synonyms(entities_true, synonyms_dict)
    answer_synonyms = expand_with_synonyms(entities_answer, synonyms_dict)
    
    # Compute intersection based on synonym matches
    intersection_count = 0
    used_answer_entities = set()
    for true_entity, true_synonyms_set in true_synonyms.items():
        matched = False
        for answer_entity, answer_synonyms_set in answer_synonyms.items():
            if answer_entity not in used_answer_entities:
                # If the sets overlap, that's a match
                if not true_synonyms_set.isdisjoint(answer_synonyms_set):
                    intersection_count += 1
                    used_answer_entities.add(answer_entity)
                    matched = True
                    break

    # Calculate precision, recall, and F1 score
    precision = intersection_count / len(entities_answer) if len(entities_answer) > 0 else 0.0
    recall = intersection_count / len(entities_true) if len(entities_true) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

# Lemmatization function
def lemmatize_entities(entities):
    """
    Lemmatize a set of entity strings using Stanza's en_biomedical pipeline.
    
    Args:
        entities (set): A set of entity strings to be lemmatized.
    Returns:
        set: A set of lemmatized entity strings.
    """
    lemmatized_entities = set()
    for entity in entities:
        doc = biomed_nlp(entity)
        lemmatized_entity = " ".join([word.lemma for sentence in doc.sentences for word in sentence.words])
        lemmatized_entities.add(lemmatized_entity)
    return lemmatized_entities

def parsing_and_computing_f1(input_answer_dir, model_list, rephrased=False):
    # List of scispaCy models to use
    nlp_models = {
        'en_core_sci_lg': spacy.load('en_core_sci_lg'),
    }
    file_path_to_dic =  "./synonyms_dictionary.json"
    with open(file_path_to_dic, 'r') as file:
        synonyms_dict = json.load(file)
    
    # Initialize data collection
    data_records = []
    category_ids = [str(num) for num in range(1, 7)]  
    iteration_numbers = [1, 2, 3]
    
    # Define which subfolders to check per model
    for nlp_name, nlp in nlp_models.items():
        print(f"Processing with spacy model: {nlp_name}")
        for model in model_list:
            if model in ["Llama", "Meditron"]:
                subfolders = [f"{model}_api", f"{model}_cluster"]
            else:
                subfolders = [model]
            
            for subfolder in subfolders:
                print(f"  Checking subfolder: {subfolder}")
                for category_id in category_ids:
                    for iteration_number in iteration_numbers:
                        answer_file_name = f"{model}_answers_category_{category_id}.{iteration_number}_HIV_EQ.json"
                        input_answer_model = os.path.join(input_answer_dir, "raw", subfolder)
                        file_path = os.path.join(input_answer_model, answer_file_name)
                        
                        if not os.path.exists(file_path):
                            continue
                        
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        
                        for idx, item in enumerate(data):
                            true_answer = item.get('true_answer', '')
                            answer = item.get('answer', '')
                            question_index = idx
                            
                            true_answer_clean = clean_text(true_answer)
                            answer_clean = clean_text(answer)
                            
                            doc_true = nlp(true_answer_clean)
                            doc_answer = nlp(answer_clean)
                            
                            entities_true = set(ent.text.lower() for ent in doc_true.ents)
                            entities_answer = set(ent.text.lower() for ent in doc_answer.ents)
                            
                            if entities_answer:
                                intersection = entities_true & entities_answer
                                precision = len(intersection) / len(entities_answer) if len(entities_answer) > 0 else 0.0
                                recall = len(intersection) / len(entities_true) if len(entities_true) > 0 else 0.0
                                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                            else:
                                precision = recall = f1 = 0.0
                            
                            print(f"Computing F1 with synonyms for: {answer_file_name}")
                            synonyms_precision, synonyms_recall, synonyms_f1 = compute_f1_with_synonyms(
                                entities_true, entities_answer, synonyms_dict
                            )
                            
                            print(f"Computing F1 with synonyms and lemmatization for: {answer_file_name}")
                            entities_true_lemmatized = lemmatize_entities(entities_true)
                            entities_answer_lemmatized = lemmatize_entities(entities_answer)
                            synonyms_lemma_precision, synonyms_lemma_recall, synonyms_lemma_f1 = compute_f1_with_synonyms(
                                entities_true_lemmatized, entities_answer_lemmatized, synonyms_dict
                            )
                            
                            data_records.append({
                                'nlp_model': nlp_name,
                                'model': model,
                                'model_subfolder': subfolder,
                                'category_id': category_id,
                                'iteration_number': iteration_number,
                                'question_index': question_index,
                                'precision': precision,
                                'recall': recall,
                                'f1_score': f1,
                                'synonyms_precision': synonyms_precision,
                                'synonyms_recall': synonyms_recall,
                                'synonyms_f1': synonyms_f1,
                                'synonyms_lemmatized_precision': synonyms_lemma_precision,
                                'synonyms_lemmatized_recall': synonyms_lemma_recall,
                                'synonyms_lemmatized_f1': synonyms_lemma_f1
                            })

    os.makedirs('./evaluation_results', exist_ok=True)
    with open('./evaluation_results/f1_results.json', 'w') as f:
        json.dump(data_records, f, indent=4)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_answers_files_path = os.path.join(script_dir, 'model_answers')
    model_list = ["NVLM", "Med42", "Claude", "Llama", "Meditron"]
    
    parsing_and_computing_f1(model_answers_files_path, model_list)