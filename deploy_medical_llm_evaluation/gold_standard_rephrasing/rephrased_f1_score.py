import os
import re
import json
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import spacy
import stanza

import nltk
from nltk.corpus import wordnet

from owlready2 import *
from owlready2.pymedtermino2 import *
from owlready2.pymedtermino2.umls import *

# If not already downloaded, download stanza model
stanza.download('en', package='mimic', processors='tokenize,pos,lemma')
biomed_nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma', package='mimic')


########################################################################
# Cleaning
########################################################################
def clean_text(text):
    # Replace newline characters with a space and remove asterisks
    text = text.replace('\n', ' ').replace('*', '')
    # Remove any extra spaces that may have resulted
    text = ' '.join(text.split())
    return text


########################################################################
# Synonym retrieval: Dictionary, SNOMED, WordNet
########################################################################
def get_synonyms_dictionary(word, synonyms_dict):
    """Get a list of synonyms from the local dictionary for a given word."""
    if word in synonyms_dict:
        return synonyms_dict[word]
    else:
        return []

def get_synonyms_snomedct_us(word):
    """
    Retrieve synonyms for a given word using PyMedTermino2 (SNOMED CT).
    """
    synonyms = []
    try:
        PYM = get_ontology("http://PYM/").load()
        SNOMEDCT_US = PYM["SNOMEDCT_US"]
        concept_matches = SNOMEDCT_US.search(word)
        if concept_matches:
            # take the first match or iterate over all if you prefer
            concept = concept_matches[0]
            for term in concept.label:
                synonyms.append(str(term))
    except Exception as e:
        print(f"Error retrieving synonyms from SNOMED for '{word}': {e}")
    return synonyms

def get_synonyms_wordnet(word):
    """Get a list of synonyms from WordNet for a given word."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)


########################################################################
# Expanding Entities with All Synonyms (Dictionary, SNOMED, WordNet)
########################################################################
def expand_with_all_synonyms(entities, synonyms_dict):
    """
    Returns three expansions (dicts) of the given entities:
     1) Dictionary-based
     2) SNOMED-based
     3) WordNet-based

    expansions_dict[entity] -> set of synonyms (including entity)
    expansions_snomed[entity] -> set of synonyms (including entity)
    expansions_wordnet[entity] -> set of synonyms (including entity)
    """
    expansions_dict = {}
    expansions_snomed = {}
    expansions_wordnet = {}

    for entity in entities:
        # dictionary
        dict_syns = get_synonyms_dictionary(entity, synonyms_dict)
        expansions_dict[entity] = set(dict_syns + [entity])

        # snomed
        snomed_syns = get_synonyms_snomedct_us(entity)
        expansions_snomed[entity] = set(snomed_syns + [entity])

        # wordnet
        wn_syns = get_synonyms_wordnet(entity)
        expansions_wordnet[entity] = set(wn_syns + [entity])

    return expansions_dict, expansions_snomed, expansions_wordnet


########################################################################
# Compute F1 from Already Expanded Sets
########################################################################
def compute_f1_from_expanded(entities_true, entities_answer, expanded_true, expanded_answer):
    """
    Given expanded_true[true_entity] = set_of_syns, 
          expanded_answer[answer_entity] = set_of_syns,
    compute the number of matches and then precision/recall/F1.
    """
    if not entities_true and not entities_answer:
        return 1.0, 1.0, 1.0  # both empty => perfect
    if not entities_true:
        # no ground truth, but we predicted something => recall=1, precision=0
        return 0.0, 1.0, 0.0
    if not entities_answer:
        # have ground truth, but predicted nothing => recall=0
        return 1.0, 0.0, 0.0

    intersection_count = 0
    used_answer_entities = set()
    for true_ent, syns_true in expanded_true.items():
        for ans_ent, syns_ans in expanded_answer.items():
            if ans_ent not in used_answer_entities:
                # Check if synonyms sets intersect => match
                if not syns_true.isdisjoint(syns_ans):
                    intersection_count += 1
                    used_answer_entities.add(ans_ent)
                    break

    precision = intersection_count / len(entities_answer) if len(entities_answer) > 0 else 0.0
    recall = intersection_count / len(entities_true) if len(entities_true) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    return precision, recall, f1


########################################################################
# Lemmatization function
########################################################################
def lemmatize_entities(entities):
    """
    Lemmatize a set of entity strings using Stanza's en_biomedical pipeline.
    """
    lemmatized_entities = set()
    for entity in entities:
        doc = biomed_nlp(entity)
        lem = " ".join(word.lemma for sent in doc.sentences for word in sent.words)
        lemmatized_entities.add(lem)
    return lemmatized_entities


########################################################################
# Main Parsing & F1 Computation
########################################################################
def parsing_and_computing_f1(input_answer_dir, model_list):
    # You only have "en_core_sci_lg" here, but you can add more if desired
    nlp_models = {
        'en_core_sci_lg': spacy.load('en_core_sci_lg'),
    }

    # Load your custom synonyms dictionary
    file_path_to_dic = ".././synonyms_dictionary.json"
    with open(file_path_to_dic, 'r') as file:
        synonyms_dict = json.load(file)
    
    data_records = []
    category_ids = [str(num) for num in range(1, 7)]
    iteration_numbers = [1, 2, 3, 4, 5]
    
    # Process each file with each scispaCy model
    for nlp_name, nlp_instance in nlp_models.items():
        print(f"Processing with model: {nlp_name}")
        for model in model_list:
            for category_id in category_ids:
                for iteration_number in iteration_numbers:
                    # Construct file name
                    answer_file_name = (
                        f"GPT-rephrase_true_answer_answers_category_{category_id}."
                        f"{iteration_number}_HIV_EQ.json"
                    )
                    input_answer_model = os.path.join(
                        input_answer_dir, "rephrased_true_answers"
                    )
                    file_path = os.path.join(input_answer_model, answer_file_name)
                    
                    if not os.path.exists(file_path):
                        print(f"File not found: {file_path}")
                        continue
                    
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                    # Process each answer pair
                    for idx, item in enumerate(data):
                        # The 'response' field contains the JSON with rephrased answers
                        try:
                            # Remove leading/trailing content around the JSON block
                            response_json = json.loads(item["response"].strip("```json\n"))
                        except json.JSONDecodeError as e:
                            print(f"JSON decode error for file {answer_file_name}, question {idx}: {e}")
                            continue
                        
                        # Extract the text from JSON
                        true_answer = response_json.get('true_answer', '')
                        answer = response_json.get('true_answer_rephrased', '')
                        question_index = idx
                    
                        # Clean
                        true_answer_clean = clean_text(true_answer)
                        answer_clean = clean_text(answer)
    
                        # Use scispaCy to get entities
                        doc_true = nlp_instance(true_answer_clean)
                        doc_answer = nlp_instance(answer_clean)
    
                        entities_true = set(ent.text.lower() for ent in doc_true.ents)
                        entities_answer = set(ent.text.lower() for ent in doc_answer.ents)
    
                        # 1) Basic exact-match F1
                        if entities_answer:
                            intersection = entities_true & entities_answer
                            precision = len(intersection) / len(entities_answer) if len(entities_answer) > 0 else 0.0
                            recall = len(intersection) / len(entities_true) if len(entities_true) > 0 else 0.0
                            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                        else:
                            precision = recall = f1 = 0.0

                        # 2) Synonyms expansions (dictionary, SNOMED, WordNet) for original entities
                        (true_dict_expanded,
                         true_snomed_expanded,
                         true_wn_expanded) = expand_with_all_synonyms(entities_true, synonyms_dict)
                        
                        (ans_dict_expanded,
                         ans_snomed_expanded,
                         ans_wn_expanded) = expand_with_all_synonyms(entities_answer, synonyms_dict)
                        
                        # 2a) Dictionary expansions
                        dict_prec, dict_rec, dict_f1 = compute_f1_from_expanded(
                            entities_true, entities_answer,
                            true_dict_expanded, ans_dict_expanded
                        )
                        # 2b) SNOMED expansions
                        snomed_prec, snomed_rec, snomed_f1 = compute_f1_from_expanded(
                            entities_true, entities_answer,
                            true_snomed_expanded, ans_snomed_expanded
                        )
                        # 2c) WordNet expansions
                        wn_prec, wn_rec, wn_f1 = compute_f1_from_expanded(
                            entities_true, entities_answer,
                            true_wn_expanded, ans_wn_expanded
                        )
    
                        # 3) Lemmatized expansions
                        entities_true_lemma = lemmatize_entities(entities_true)
                        entities_answer_lemma = lemmatize_entities(entities_answer)
                        
                        (true_dict_expanded_lemma,
                         true_snomed_expanded_lemma,
                         true_wn_expanded_lemma) = expand_with_all_synonyms(entities_true_lemma, synonyms_dict)
                        
                        (ans_dict_expanded_lemma,
                         ans_snomed_expanded_lemma,
                         ans_wn_expanded_lemma) = expand_with_all_synonyms(entities_answer_lemma, synonyms_dict)
                        
                        # 3a) Dictionary expansions, lemma
                        dict_lemma_prec, dict_lemma_rec, dict_lemma_f1 = compute_f1_from_expanded(
                            entities_true_lemma, entities_answer_lemma,
                            true_dict_expanded_lemma, ans_dict_expanded_lemma
                        )
                        # 3b) SNOMED expansions, lemma
                        snomed_lemma_prec, snomed_lemma_rec, snomed_lemma_f1 = compute_f1_from_expanded(
                            entities_true_lemma, entities_answer_lemma,
                            true_snomed_expanded_lemma, ans_snomed_expanded_lemma
                        )
                        # 3c) WordNet expansions, lemma
                        wn_lemma_prec, wn_lemma_rec, wn_lemma_f1 = compute_f1_from_expanded(
                            entities_true_lemma, entities_answer_lemma,
                            true_wn_expanded_lemma, ans_wn_expanded_lemma
                        )

                        # Final record
                        data_records.append({
                            'nlp_model': nlp_name,
                            'model': model,
                            'category_id': category_id,
                            'iteration_number': iteration_number,
                            'question_index': question_index,

                            # Basic exact matches
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1,

                            # Dictionary expansions
                            'synonyms_precision_dict': dict_prec,
                            'synonyms_recall_dict': dict_rec,
                            'synonyms_f1_dict': dict_f1,
                            # SNOMED expansions
                            'synonyms_precision_snomed': snomed_prec,
                            'synonyms_recall_snomed': snomed_rec,
                            'synonyms_f1_snomed': snomed_f1,
                            # WordNet expansions
                            'synonyms_precision_wn': wn_prec,
                            'synonyms_recall_wn': wn_rec,
                            'synonyms_f1_wn': wn_f1,

                            # Dictionary expansions + lemma
                            'synonyms_lemmatized_precision_dict': dict_lemma_prec,
                            'synonyms_lemmatized_recall_dict': dict_lemma_rec,
                            'synonyms_lemmatized_f1_dict': dict_lemma_f1,
                            # SNOMED expansions + lemma
                            'synonyms_lemmatized_precision_snomed': snomed_lemma_prec,
                            'synonyms_lemmatized_recall_snomed': snomed_lemma_rec,
                            'synonyms_lemmatized_f1_snomed': snomed_lemma_f1,
                            # WordNet expansions + lemma
                            'synonyms_lemmatized_precision_wn': wn_lemma_prec,
                            'synonyms_lemmatized_recall_wn': wn_lemma_rec,
                            'synonyms_lemmatized_f1_wn': wn_lemma_f1,
                        })
    
    # Save to JSON
    out_path = './rephrased-f1_results.json'
    with open(out_path, 'w') as f:
        json.dump(data_records, f, indent=4)
    print(f"Saved results to {out_path}")


########################################################################
# Run
########################################################################
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_list = ["rephrased_true_answers"]

    # Set PyMedTermino's SQLite DB if needed
    pym_db_path = "/cluster/scratch/gcardenal/LLM_models/pymedtermino_sql/pym.sqlite3_batch"
    default_world.set_backend(filename=pym_db_path)

    parsing_and_computing_f1(input_answer_dir=script_dir, model_list=model_list)