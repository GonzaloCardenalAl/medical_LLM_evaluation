import os
import re
import json
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import spacy
import json
import stanza

# If not already downloaded, download stanza model
stanza.download('en', package='mimic', processors='tokenize,pos,lemma')
biomed_nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma', package='mimic')

# Define the cleaning function
def clean_text(text):
    # Replace newline characters with a space and remove asterisks
    text = text.replace('\n', ' ').replace('*', '')
    # Remove any extra spaces that may have resulted
    text = ' '.join(text.split())
    return text

#########################################
# Synonyms and Lemmatization Functions
#########################################

from owlready2 import *
from owlready2.pymedtermino2 import *
from owlready2.pymedtermino2.umls import *

# Ensure PyMedTermino ontology is loaded

def get_synonyms(entity):
    """
    Retrieve synonyms for a given entity using PyMedTermino2.
    Args:
        entity (str): The clinical entity for which synonyms are needed.
    Returns:
        list: A list of synonyms for the entity.
    """
    try:
        PYM = get_ontology("http://PYM/").load()
        SNOMEDCT_US = PYM["SNOMEDCT_US"]
        concept = SNOMEDCT_US.search(entity)  # Search for the entity in SNOMED CT
        if concept:
            concept = concept[0]  # Take the first match
            synonyms = [str(term) for term in concept.label]
            return synonyms
        else:
            return []
    except Exception as e:
        print(f"Error retrieving synonyms for {entity}: {e}")
        return []

def compute_f1_with_synonyms(entities_true, entities_answer):
    """
    Compute precision, recall, and F1 score using synonym-based intersection.
    """
    def expand_with_synonyms(entities):
        expanded = {}
        for entity in entities:
            synonyms = get_synonyms(entity)  # Retrieve synonyms using PyMedTermino2
            # Include the original entity as well, for direct matches
            expanded[entity] = set(synonyms + [entity])
        return expanded

    # Expand entities with synonyms
    true_synonyms = expand_with_synonyms(entities_true)
    answer_synonyms = expand_with_synonyms(entities_answer)
    
    # Compute intersection based on synonym matches
    intersection_count = 0
    used_answer_entities = set()
    for true_entity, true_synonyms_set in true_synonyms.items():
        for answer_entity, answer_synonyms_set in answer_synonyms.items():
            if answer_entity not in used_answer_entities and not true_synonyms_set.isdisjoint(answer_synonyms_set):
                intersection_count += 1
                used_answer_entities.add(answer_entity)
                break

    # Calculate precision, recall, and F1 score
    precision = intersection_count / len(entities_answer) if len(entities_answer) > 0 else 0.0
    recall = intersection_count / len(entities_true) if len(entities_true) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def lemmatize_entities(entities):
    """
    Lemmatize a set of entity strings using Stanza's en_biomedical pipeline.
    """
    lemmatized_entities = set()
    for entity in entities:
        doc = biomed_nlp(entity)
        lemmatized_entity = " ".join([word.lemma for sentence in doc.sentences for word in sentence.words])
        lemmatized_entities.add(lemmatized_entity)
    return lemmatized_entities

#########################################
# Modified parsing_and_computing_f1 function
#########################################

def parsing_and_computing_f1(input_answer_dir, model_list):
    # List of scispaCy models to use
    try:
        nlp = spacy.load('en_core_sci_lg')
    except OSError:
        print("en_core_sci_lg model not found. Please install it before running.")
        return
    
    nlp_models = {
        'en_core_sci_lg': nlp
    }
    
    # Initialize data collection
    data_records = []
    category_ids = [str(num) for num in range(1, 7)]  # Categories 1 to 4
    iteration_numbers = [1, 2, 3]
    
    # Process each file with each scispaCy model
    for nlp_name, nlp_instance in nlp_models.items():
        print(f"Processing with model: {nlp_name}")
        for model in model_list:
            for category_id in category_ids:
                for iteration_number in iteration_numbers:
                    # Construct file names
                    answer_file_name = f"GPT-rephrase_true_answer_answers_category_{category_id}.{iteration_number}_HIV_EQ.json"
                    input_answer_model = os.path.join(input_answer_dir, f"rephrased_true_answers/")
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
                        # Load the response field as JSON
                        try:
                            response_json = json.loads(item["response"].strip("```json\n"))
                        except json.JSONDecodeError as e:
                            print(f"JSON decode error for file {answer_file_name}, question {idx}: {e}")
                            continue
                        
                        true_answer = response_json.get('true_answer', '')
                        answer = response_json.get('true_answer_rephrased', '')
                        question_index = idx
                    
                        # Clean the 'true_answer' and 'answer' strings
                        true_answer_clean = clean_text(true_answer)
                        answer_clean = clean_text(answer)
    
                        # Extract entities using scispaCy
                        doc_true = nlp_instance(true_answer_clean)
                        doc_answer = nlp_instance(answer_clean)
    
                        # Proceed with entity extraction
                        entities_true = set(ent.text.lower() for ent in doc_true.ents)
                        entities_answer = set(ent.text.lower() for ent in doc_answer.ents)
    
                        # Compute original exact match F1
                        if entities_answer:
                            intersection = entities_true & entities_answer
                            precision = len(intersection) / len(entities_answer) if len(entities_answer) > 0 else 0.0
                            recall = len(intersection) / len(entities_true) if len(entities_true) > 0 else 0.0
                            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                        else:
                            precision = 0.0
                            recall = 0.0
                            f1 = 0.0

                        # --- F1 with synonyms ---
                        synonyms_precision, synonyms_recall, synonyms_f1 = compute_f1_with_synonyms(entities_true, entities_answer)
    
                        # --- F1 with synonyms + lemmatization ---
                        entities_true_lemmatized = lemmatize_entities(entities_true)
                        entities_answer_lemmatized = lemmatize_entities(entities_answer)
                        synonyms_lemma_precision, synonyms_lemma_recall, synonyms_lemma_f1 = compute_f1_with_synonyms(entities_true_lemmatized, entities_answer_lemmatized)
    
                        # Append results to data_records
                        data_records.append({
                            'nlp_model': nlp_name,
                            'model': model,
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
    
    # Save data_records to a JSON file
    with open('./rephrased-f1_results.json', 'w') as f:
        json.dump(data_records, f, indent=4)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_list = ["rephrased_true_answers"]
    # Run the F1 score calculation
    pym_db_path = "/cluster/scratch/gcardenal/LLM_models/pymedtermino_sql/pym.sqlite3"
    default_world.set_backend(filename=pym_db_path)
    parsing_and_computing_f1(input_answer_dir=script_dir, model_list=model_list)