# Ethical Dynamics in Fictional Characters: A Framework for Monitoring Moral Actions in Narrative Texts  

## Abstract  

## Implementation  
### 1. Dataset  
- movie dataset : [Download Link](https://drive.google.com/drive/folders/1rfEtKgLVnjhGAgxKPWuguCsc6eHI3vUh) 에서 *IMDB_movie_details.json.zip* 파일 다운로드. `data/` 폴더에 넣기.
- Moral Action detector training dataset : [Download Link](https://drive.google.com/drive/folders/1rfEtKgLVnjhGAgxKPWuguCsc6eHI3vUh) 에서 *moral_stories_train.tsv*, *moral_stories_valid.tsv*, *moral_stories_test.tsv* 파일 다운로드. `data/moral_dataset/` 폴더에 넣기.

### 2. Dataset Preprocessing  
```
cd preprocess
python story_dataset_preprocess.py
```
- *data/preprocessed/movie_synopsis_segments_n2.jsonl* 파일 생성
- 전처리 완료 파일 : [Download Link](https://drive.google.com/drive/folders/1rfEtKgLVnjhGAgxKPWuguCsc6eHI3vUh) 에서 *movie_synopsis_segments_n2.jsonl* 다운로드해서 `data/preprocessed/` 폴더에 넣기.
