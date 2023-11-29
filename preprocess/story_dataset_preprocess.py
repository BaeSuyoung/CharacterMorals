import pandas as pd
import json
import os 
import io
import re
import tqdm
from zipfile import ZipFile
from collections import Counter

import spacy
NER=spacy.load('en_core_web_sm')

from pycorenlp import StanfordCoreNLP

is_preprocess=False
is_coref=False
is_char_replace=True

Seg_unit=2

DATA_PATH='data/IMDB_movie_details.json.zip'
OUTPUT_PATH="data/preprocessed/movie_synopsis_segments_n2.jsonl"

####################### Functions #######################

def load_jsonl(input_path):
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(input_path, datasets):
    with open(input_path, 'w') as outfile:
        json.dump(datasets, outfile)

def token_len(documents):
    overview_len=[]
    for i in range(len(documents)):
        overview_len.append(len(documents[i].split(" ")))
    print("average token length: ", sum(overview_len)/len(overview_len))
    print("min token length: ", min(overview_len), "\nmax token length: ", max(overview_len))
    print(overview_len[:5])
    return overview_len

def preprocess(documents):
    documents=documents.str.replace(r'\([^()]*\)', '', regex=True)
    documents=documents.str.replace("Mr. ", "")
    documents=documents.str.replace("Mrs. ", "")
    documents=documents.str.replace("Dr. ", "")
    documents=documents.str.replace('--', ' ')
    documents=documents.str.replace('-', ' ')
    documents=documents.str.replace('[', ' ')
    documents=documents.str.replace(']', ' ')
    documents=documents.str.replace('*', '')
    documents=documents.str.replace(':', '.')
    documents=documents.str.replace(';', '.')
    documents=documents.str.replace('_', '')
    documents=documents.str.replace('\n', ' ')
    documents=documents.str.replace('  ', ' ')

    return documents

def resolve_coreferences(text):
    
    final_output=""
    nlp = StanfordCoreNLP('http://localhost:9000')

    # len split
    segment_texts=[]
    text_list=text.split(".")
    for i in range(0, len(text_list), 50):
        merged_element = ".".join(map(str, text_list[i:i + 50]))
        segment_texts.append(merged_element)

    for seg in segment_texts:
    
        output = nlp.annotate(str(seg), properties={
            'annotators': 'coref',
            'outputFormat': 'json'
        })

        sentences=[]
        for out_sents in output['sentences']:
            sentence=[]
            for out_sent in out_sents['tokens']:
                sentence.append(out_sent['word'])
            sentences.append(sentence)

        #print(sentences)
        for cluster in output['corefs'].values():
            main_mention = cluster[0]['text']

            #print(main_mention)
            
            #print(cluster)
            for mention in cluster[1:]:
                sent_num=mention['sentNum']-1 #1

                start_idx = mention['startIndex'] -1
                end_idx = mention['endIndex'] -1
                
                #print(start_idx, end_idx, main_mention)
                #print("before: ", sentences[sent_num])
                sentences[sent_num][start_idx]=main_mention

                if start_idx != (end_idx-1) : 
                    for i in range(start_idx+1, end_idx):
                        sentences[sent_num][i]=''
                #print("after", sentences[sent_num])
        
        sentences =" ".join([" ".join(x) for x in sentences])
        final_output+=sentences+" "
    #print(final_output)
    return final_output

def character_replacement(txt):
    doc = NER(txt)
    char_set=set()
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            char_set.add(ent.text)
    
    for char in list(char_set):
        
        txt=txt.replace(char, "["+char.replace(" ", "_")+"]")

    return txt

def find_pattern_in_text(text):
    pattern = r'\[[a-zA-Z_]+\]'
    matches = re.findall(pattern, text)
    matches=[x[1:-1] for x in matches]
    return matches


def merge_segments_with_characters(segment_list, character_list):
    merged_segments = []
    merged_characters=[]
    current_segment = ""
    current_character = None
    for segment, character in zip(segment_list, character_list):
        if current_character is None:
            current_segment = segment
            current_character = character
        elif current_character == character:
            current_segment += " " + segment
        else:
            merged_segments.append(current_segment)
            merged_characters.append(current_character)
            current_segment = segment
            current_character = character

    if current_segment:
        merged_segments.append(current_segment)
        merged_characters.append(current_character)

    assert len(merged_segments) == len(merged_characters)

    return merged_segments, merged_characters



####################### Preprocessing #######################

with ZipFile(DATA_PATH, 'r') as zipObj:
    zipObj.extractall('data/')


movie_dataset=pd.read_json("data/IMDB_movie_details.json", lines=True)

plot_synopsis=list(movie_dataset["plot_synopsis"])

# token length
plot_synopsis_len=token_len(plot_synopsis)
movie_dataset["plot_synopsis_len"]=plot_synopsis_len

# movid_id, genre, ploy_synopsis, plot_synopsis_len
movie_dataset=movie_dataset[["movie_id", "genre", "plot_synopsis", "plot_synopsis_len"]]

# plot_synopsis_len 10 이하 제거
drop_idxs=movie_dataset[movie_dataset["plot_synopsis_len"]<10].index
print("number of drops: ", len(drop_idxs))
movie_dataset=movie_dataset.drop(drop_idxs)
movie_dataset=movie_dataset.reset_index(drop=True)

# plot_synopsis 전처리
if is_preprocess:
    movie_dataset["plot_synopsis"]=preprocess(movie_dataset["plot_synopsis"])

# coreference
if is_coref:
    movie_dataset["plot_synopsis_coref"]=movie_dataset['plot_synopsis'].apply(resolve_coreferences)

# character replacement
if is_char_replace:
    movie_dataset['plot_synopsis_cvt'] = movie_dataset['plot_synopsis_coref'].apply(character_replacement)


####################### Segmentation #######################

output_dict=[]
for i in range(len(movie_dataset)):
    segment_texts=[]
    segment_char=[]
    movie_id=movie_dataset['movie_id'][i]
    movie_genre=movie_dataset["genre"][i]
    text_list=movie_dataset["plot_synopsis_cvt"][i].split(".")
    
    for i in range(0, len(text_list), Seg_unit):
        merged_element = ".".join(map(str, text_list[i:i + Seg_unit]))
        segment_texts.append(merged_element)
    
    segment_texts=[x for x in segment_texts if len(x)>10]
    for i in range(len(segment_texts)):
        segment_char.append(find_pattern_in_text(segment_texts[i]))
    
    assert len(segment_texts)==len(segment_char)

    # main character
    character_occurrence=[]
    for i in range(len(segment_char)):
        character_occurrence.extend(segment_char[i])
        segment_char[i]=list(set(segment_char[i]))

    main_character=Counter(character_occurrence).most_common(5)
    
    output_dict.append({
        "movie_id": movie_id,
        "genre": movie_genre,
        "segments": segment_texts,
        "segment_char": segment_char,
        "main_character": main_character
    })

final_dict=[]
for i in range(len(output_dict)):
    segment_char=output_dict[i]["segment_char"]
    segments=output_dict[i]["segments"]
    main_char=output_dict[i]["main_character"]

    merged_segments, merged_characters = merge_segments_with_characters(segments, segment_char)
    
    final_dict.append({
        "movie_id": output_dict[i]["movie_id"],
        "genre": output_dict[i]["genre"],
        "segments": merged_segments,
        "segment_char": merged_characters,
        "main_char": main_char
    })

save_jsonl(OUTPUT_PATH, final_dict)
