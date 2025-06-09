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

# For WordNet synonyms
import nltk
from nltk.corpus import wordnet

# Make sure you have downloaded WordNet data:
# nltk.download('wordnet')

# Configuration for UMLS
UMLS_API_KEY = '165ce1dc-cc57-4526-8e52-b7c4f7df2085'
DOWNLOAD_URL = 'https://download.nlm.nih.gov/umls/kss/2024AB/umls-2024AB-metathesaurus-full.zip'
DESTINATION_PATH = '/cluster/scratch/gcardenal/LLM_models/pymedtermino_sql/umls-2024AB-metathesaurus-full.zip'

def download_umls_file(api_key, download_url, destination_path):
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)

    # Append API key to the URL
    url_with_api_key = f"https://uts-ws.nlm.nih.gov/download?url={download_url}&apiKey={api_key}"
    
    # Make the GET request
    response = requests.get(url_with_api_key, stream=True)
    if response.status_code == 200:
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

####################################################
# Synonym retrieval functions
####################################################
file_path_to_dic = "./synonyms_dictionary.json"
with open(file_path_to_dic, 'r') as file:
    synonyms_dict = json.load(file)

def get_synonyms_dictionary(word, synonyms_dict):
    """
    Get a list of synonyms from the custom dictionary for a given word.
    """
    if word in synonyms_dict:
        return synonyms_dict[word]
    else:
        return []

def get_synonyms_snomedct_us(entity):
    """
    Retrieve synonyms for a given entity using PyMedTermino2's SNOMEDCT_US.
    """
    try:
        PYM = get_ontology("http://PYM/").load()
        SNOMEDCT_US = PYM["SNOMEDCT_US"]
        concept = SNOMEDCT_US.search(entity)  # Search for the entity in SNOMED CT
        if concept:
            # Take the first match
            concept = concept[0]
            synonyms = [str(term) for term in concept.label]
            return synonyms
        else:
            return []
    except Exception as e:
        print(f"Error retrieving SNOMED synonyms for {entity}: {e}")
        return []

def get_synonyms_wordnet(word):
    """
    Get a list of synonyms for a given word from WordNet.
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

####################################################
# Cleaning and lemmatizing
####################################################
def clean_text(text):
    # Replace newline characters with a space and remove asterisks
    text = text.replace('\n', ' ').replace('*', '')
    # Remove any extra spaces that may have resulted
    text = ' '.join(text.split())
    return text

def lemmatize_entities(entities):
    """
    Lemmatize a set of entity strings using Stanza's biomedical pipeline.
    """
    lemmatized_entities = set()
    for entity in entities:
        doc = biomed_nlp(entity)
        lemmatized_entity = " ".join(word.lemma for sent in doc.sentences for word in sent.words)
        lemmatized_entities.add(lemmatized_entity)
    return lemmatized_entities

####################################################
# Expansion with all synonyms (three sources)
####################################################
def expand_with_all_synonyms(entities, synonyms_dict):
    """
    Returns three different expansions of the given set of entities:
      1. Using your local synonyms dictionary
      2. Using SNOMEDCT_US
      3. Using WordNet
    Each expansion is a dict: entity -> set_of_synonyms_including_entity_itself
    """
    expansions_dict = {}
    expansions_snomed = {}
    expansions_wordnet = {}

    for entity in entities:
        # 1) local dictionary
        dict_syns = get_synonyms_dictionary(entity, synonyms_dict)
        expansions_dict[entity] = set(dict_syns + [entity])

        # 2) SNOMED
        snomed_syns = get_synonyms_snomedct_us(entity)
        expansions_snomed[entity] = set(snomed_syns + [entity])

        # 3) WordNet
        wn_syns = get_synonyms_wordnet(entity)
        expansions_wordnet[entity] = set(wn_syns + [entity])
    
    return expansions_dict, expansions_snomed, expansions_wordnet

####################################################
# Computing F1 given expansions
####################################################
def compute_f1_from_expanded(entities_true, entities_answer, true_expanded, answer_expanded):
    """
    Given two dicts (true_expanded, answer_expanded) where:
       - true_expanded[entity_true]  = set of synonyms (including the entity itself)
       - answer_expanded[entity_ans] = set of synonyms (including the entity itself)
    Compute precision, recall, and F1 if we consider a match to occur when
    the sets of synonyms intersect.
    """
    if not entities_true and not entities_answer:
        return 1.0, 1.0, 1.0  # both empty => perfect match in this interpretation
    if not entities_true:
        # If there's no ground truth but we predicted something, precision=0, recall=1
        return 0.0, 1.0, 0.0
    if not entities_answer:
        # If we have ground truth but predicted nothing, recall=0
        return 1.0, 0.0, 0.0

    intersection_count = 0
    used_answer_entities = set()

    for true_entity, true_synonyms_set in true_expanded.items():
        for ans_entity, ans_synonyms_set in answer_expanded.items():
            if ans_entity not in used_answer_entities:
                # If the sets overlap => treat as a match
                if not true_synonyms_set.isdisjoint(ans_synonyms_set):
                    intersection_count += 1
                    used_answer_entities.add(ans_entity)
                    break

    precision = intersection_count / len(entities_answer) if len(entities_answer) > 0 else 0.0
    recall = intersection_count / len(entities_true) if len(entities_true) > 0 else 0.0
    if (precision + recall) > 0:
        f1 = 2.0 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1

####################################################
# Main parsing and computing F1
####################################################
def parsing_and_computing_f1(input_answer_dir, model_list, rephrased=False):
    # Load spacy model(s)
    nlp_models = {
        'en_core_sci_lg': spacy.load('en_core_sci_lg'),
    }

    # For final JSON records
    data_records = []

    category_ids = [str(num) for num in range(1, 7)]
    iteration_numbers = [1, 2, 3, 4, 5]

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

                            #-----------------------
                            # Basic string-match F1
                            #-----------------------
                            if entities_answer:
                                intersection = entities_true & entities_answer
                                precision = (len(intersection) / len(entities_answer)
                                             if len(entities_answer) > 0 else 0.0)
                                recall = (len(intersection) / len(entities_true)
                                          if len(entities_true) > 0 else 0.0)
                                f1 = (2 * (precision * recall) / (precision + recall)
                                      if (precision + recall) > 0 else 0.0)
                            else:
                                precision = recall = f1 = 0.0

                            #-----------------------------------
                            # Expand synonyms (three sources)
                            #-----------------------------------
                            (true_dict_expanded,
                             true_snomed_expanded,
                             true_wordnet_expanded) = expand_with_all_synonyms(entities_true, synonyms_dict)
                            (ans_dict_expanded,
                             ans_snomed_expanded,
                             ans_wordnet_expanded) = expand_with_all_synonyms(entities_answer, synonyms_dict)

                            # Dictionary-based synonyms F1
                            dict_precision, dict_recall, dict_f1 = compute_f1_from_expanded(
                                entities_true, entities_answer,
                                true_dict_expanded, ans_dict_expanded
                            )

                            # SNOMED-based synonyms F1
                            snomed_precision, snomed_recall, snomed_f1 = compute_f1_from_expanded(
                                entities_true, entities_answer,
                                true_snomed_expanded, ans_snomed_expanded
                            )

                            # WordNet-based synonyms F1
                            wn_precision, wn_recall, wn_f1 = compute_f1_from_expanded(
                                entities_true, entities_answer,
                                true_wordnet_expanded, ans_wordnet_expanded
                            )

                            #-----------------------------------
                            # Lemmatized expansions (three sources)
                            #-----------------------------------
                            entities_true_lemmatized = lemmatize_entities(entities_true)
                            entities_answer_lemmatized = lemmatize_entities(entities_answer)

                            (true_dict_expanded_lemma,
                             true_snomed_expanded_lemma,
                             true_wordnet_expanded_lemma) = expand_with_all_synonyms(entities_true_lemmatized, synonyms_dict)
                            (ans_dict_expanded_lemma,
                             ans_snomed_expanded_lemma,
                             ans_wordnet_expanded_lemma) = expand_with_all_synonyms(entities_answer_lemmatized, synonyms_dict)

                            # Dictionary-based synonyms + lemma F1
                            dict_lemma_precision, dict_lemma_recall, dict_lemma_f1 = compute_f1_from_expanded(
                                entities_true_lemmatized, entities_answer_lemmatized,
                                true_dict_expanded_lemma, ans_dict_expanded_lemma
                            )

                            # SNOMED-based synonyms + lemma F1
                            snomed_lemma_precision, snomed_lemma_recall, snomed_lemma_f1 = compute_f1_from_expanded(
                                entities_true_lemmatized, entities_answer_lemmatized,
                                true_snomed_expanded_lemma, ans_snomed_expanded_lemma
                            )

                            # WordNet-based synonyms + lemma F1
                            wn_lemma_precision, wn_lemma_recall, wn_lemma_f1 = compute_f1_from_expanded(
                                entities_true_lemmatized, entities_answer_lemmatized,
                                true_wordnet_expanded_lemma, ans_wordnet_expanded_lemma
                            )

                            #---------------------------------------------------
                            # Collect all the results in one record
                            #---------------------------------------------------
                            data_records.append({
                                'nlp_model': nlp_name,
                                'model': model,
                                'model_subfolder': subfolder,
                                'category_id': category_id,
                                'iteration_number': iteration_number,
                                'question_index': question_index,
                                # Basic entity-overlap results
                                'precision': precision,
                                'recall': recall,
                                'f1_score': f1,

                                # Dictionary-based synonyms
                                'synonyms_precision_dict': dict_precision,
                                'synonyms_recall_dict': dict_recall,
                                'synonyms_f1_dict': dict_f1,

                                # SNOMED-based synonyms
                                'synonyms_precision_snomed': snomed_precision,
                                'synonyms_recall_snomed': snomed_recall,
                                'synonyms_f1_snomed': snomed_f1,

                                # WordNet-based synonyms
                                'synonyms_precision_wn': wn_precision,
                                'synonyms_recall_wn': wn_recall,
                                'synonyms_f1_wn': wn_f1,

                                # Dictionary-based synonyms + lemma
                                'synonyms_lemmatized_precision_dict': dict_lemma_precision,
                                'synonyms_lemmatized_recall_dict': dict_lemma_recall,
                                'synonyms_lemmatized_f1_dict': dict_lemma_f1,

                                # SNOMED-based synonyms + lemma
                                'synonyms_lemmatized_precision_snomed': snomed_lemma_precision,
                                'synonyms_lemmatized_recall_snomed': snomed_lemma_recall,
                                'synonyms_lemmatized_f1_snomed': snomed_lemma_f1,

                                # WordNet-based synonyms + lemma
                                'synonyms_lemmatized_precision_wn': wn_lemma_precision,
                                'synonyms_lemmatized_recall_wn': wn_lemma_recall,
                                'synonyms_lemmatized_f1_wn': wn_lemma_f1
                            })

    # Save all the results to JSON
    os.makedirs('./evaluation_results', exist_ok=True)
    with open('./evaluation_results/f1_results.json', 'w') as f:
        json.dump(data_records, f, indent=4)

####################################################
# Main script entry
####################################################
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_answers_files_path = os.path.join(script_dir, 'model_answers')
    model_list = ["Gemma-3-27B","NVLM", "Med42", "Claude", "Llama", "Meditron", "Llama-8B", "Llama-1B", "Gemini_2.5Pro",  "MedGemma-3-27B"]
    
    parsing_and_computing_f1(model_answers_files_path, model_list)