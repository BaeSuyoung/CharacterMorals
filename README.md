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
- *movie_synopsis_segments_n2.jsonl* 파일 생성
- 전처리 완료 파일 : [Download Link](https://drive.google.com/drive/folders/1rfEtKgLVnjhGAgxKPWuguCsc6eHI3vUh) 에서 *movie_synopsis_segments_n2.jsonl* 다운로드해서 `data/preprocessed/` 폴더에 넣기.
- *movie_synopsis_segments_n2.jsonl* index:
    * movie_id
    * genre
    * segments: merged segment  
    * segment_char: merged segment character set  
    * main_char: movie main characters  

### 3. Story Transformation  
```
cd preprocess
python transform.py
```
- `gpt-3.5 turbo` 사용해서 segment 를 main character's action, situation, intention, consequence 로 변환.
- *inference_n2.tsv* 파일 생성
- 전처리 완료 파일 : [Download Link](https://drive.google.com/drive/folders/1rfEtKgLVnjhGAgxKPWuguCsc6eHI3vUh) 에서 *inference_n2.tsv* 다운로드해서 `data/moral_stories/` 폴더에 넣기.
- *inference_n2.tsv* index:
    * ID
    * norm
    * situation
    * intention
    * action
    * consequence
    * label: 0으로 초기화  
    * movie id
    * original: original segment
    * genre
    * main_char : segment main character
 
 
 ### 4. Moral Action Detector Training  
 ```
./train_cls.sh
```

### 5. Segment Character's Action Prediction  
 ```
./inference_cls.sh
```

