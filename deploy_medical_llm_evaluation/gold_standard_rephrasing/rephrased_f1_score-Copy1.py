import os
import re
import json
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import spacy
import json
import stanza
from nltk.corpus import wordnet

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

# Define the cleaning function
def clean_text(text):
    # Replace newline characters with a space and remove asterisks
    text = text.replace('\n', ' ').replace('*', '')
    # Remove any extra spaces that may have resulted
    text = ' '.join(text.split())
    return text
     
#def get_synonyms(word, synonyms_dict):
#    """
#    Get a set of synonyms from the dictionary created for a given word.
#    """
#    if word in synonyms_dict:
#        return synonyms_dict[word] 
 #   else:
 #       return []

def get_synonyms(word, synonyms_dict):
    """
    Retrieve synonyms for a given entity from:
      1) The synonyms_dict (JSON-based or custom dictionary)
      2) WordNet
      3) PyMedTermino2 (SNOMED CT or another ontology)

    Args:
        word (str): The word or entity to get synonyms for.
        synonyms_dict (dict): A dictionary mapping words to a list of synonyms.

    Returns:
        list: A list of unique synonyms from all sources.
    """
    synonyms = set()

    # 1) Synonyms from your custom dictionary
    if word in synonyms_dict:
        synonyms.update(synonyms_dict[word])

    # 2) Synonyms from PyMedTermino2
    try:
        PYM = get_ontology("http://PYM/").load()
        SNOMEDCT_US = PYM["SNOMEDCT_US"]  # Adjust to your actual PyMedTermino reference
        concept_matches = SNOMEDCT_US.search(word)
        if concept_matches:
            # Take the first match or iterate over all if you prefer
            concept = concept_matches[0]
            for term in concept.label:
                synonyms.add(str(term))
    except Exception as e:
        print(f"Error retrieving synonyms from PyMedTermino2 for '{word}': {e}")

    # 3) Synonyms from WordNet
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())

    return list(synonyms)

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

def parsing_and_computing_f1(input_answer_dir, model_list):
    # List of scispaCy models to use
    nlp_models = {
        'en_core_sci_lg': spacy.load('en_core_sci_lg'),
    }
    file_path_to_dic =  ".././synonyms_dictionary.json"
    with open(file_path_to_dic, 'r') as file:
        synonyms_dict = json.load(file)
    
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
                        synonyms_precision, synonyms_recall, synonyms_f1 = compute_f1_with_synonyms(entities_true, entities_answer, synonyms_dict)
    
                        # --- F1 with synonyms + lemmatization ---
                        entities_true_lemmatized = lemmatize_entities(entities_true)
                        entities_answer_lemmatized = lemmatize_entities(entities_answer)
                        synonyms_lemma_precision, synonyms_lemma_recall, synonyms_lemma_f1 = compute_f1_with_synonyms(entities_true_lemmatized, entities_answer_lemmatized, synonyms_dict)
    
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