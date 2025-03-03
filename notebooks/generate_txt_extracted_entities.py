#This notebook, generates a dictionary with synonyms from the true answers
#Due to library conflicts, the code can not be run in a single conda environment. Therefore, the code is divided in two chunks to be run in separate environments.
import os
import json
import stanza
import spacy

stanza.download('en', package='mimic', processors='tokenize,pos,lemma')
biomed_nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma', package='mimic')


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

def clean_text(text):
    # Replace newline characters with a space and remove asterisks
    text = text.replace('\n', ' ').replace('*', '')
    # Remove any extra spaces that may have resulted
    text = ' '.join(text.split())
    return text

category_ids = [str(num) for num in range(1, 6)] 

all_entities_true = set()

for category_id in category_ids:
    file_path = f".././deploy_medical_llm_evaluation/questions_files/HIV_evaluation_questionare_category_{category_id}.json"
    file_path = f"/cluster/home/gcardenal/HIV/deploy_medical_LLM_evaluation/deploy_medical_llm_evaluation/questions_files/HIV_evaluation_questionare_category_{category_id}.json"
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    nlp = spacy.load('en_core_sci_lg')
    
    # Process each answer pair
    for idx, item in enumerate(data):
        # Load the response field as JSON
        try:
            answer = item.get('true_answer', '')
        except json.JSONDecodeError as e:
            print(f"JSON decode error for file {answer_file_name}, question {idx}: {e}")
            continue
            
       # answer = response_json.get('true_answer', '')
        answer_clean = clean_text(answer)
        doc_answer = nlp(answer_clean)
        
        # Proceed with entity extraction
        entities_true = set(ent.text.lower() for ent in doc_answer.ents)

        entities_true_lemmatized = lemmatize_entities(entities_true)
        all_entities_true.update(entities_true_lemmatized)

entities_list = sorted(all_entities_true)
output_file_path = "./extracted_entities.txt"


# Save the entities list to the file
with open(output_file_path, 'w') as file:
    for entity in entities_list:
        file.write(f"'{entity}', ")

print(f"Entities list has been saved to {output_file_path}")