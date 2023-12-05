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
- 4가지 입력 형식(A, SA, SIA, SICA) 2개 모델(bert, roberta)에 대해 실험 진행하고 가장 성능이 높은거 선택
- 실험 결과:
<table>
  <tr>
    <th>Model</th>
    <th>Task Name</th>
    <th>Train Loss</th>
    <th>Valid Acc</th>
    <th>Valid F1</th>
    <th>Test Acc</th>
    <th>Test F1</th>
  </tr>
  <tr>
    <td>BERT</td>
    <td>A</td>
    <td>0.04</td>
    <td> 0.81</td>
    <td>0.81</td>
    <td>0.81</td>
   <td>0.81</td>
  </tr>
  <tr>
    <td>BERT</td>
    <td>SA</td>
    <td>0.04</td>
    <td> 0.83</td>
    <td>0.83</td>
    <td>0.82</td>
   <td>0.82</td>
  </tr>
  <tr>
    <td>BERT</td>
    <td>SIA</td>
    <td>0.05</td>
    <td> 0.83</td>
    <td>0.83</td>
    <td>0.82</td>
   <td>0.83</td>
  </tr>
 <tr>
    <td>BERT</td>
    <td>SICA</td>
    <td>0.01</td>
    <td> 0.97</td>
    <td>0.97</td>
    <td>0.98</td>
   <td>0.98</td>
  </tr>
   <tr>
    <td>ROBERTA</td>
    <td>A</td>
    <td>0.08</td>
    <td> 0.86</td>
    <td>0.86</td>
    <td>0.85</td>
   <td>0.85</td>
  </tr>
  <tr>
    <td>ROBERTA</td>
    <td>SA</td>
    <td>0.07</td>
    <td> 0.85</td>
    <td>0.85</td>
    <td>0.85</td>
   <td>0.85</td>
  </tr>
  <tr>
    <td>ROBERTA</td>
    <td>SIA</td>
    <td>0.07</td>
    <td> 0.86</td>
    <td>0.87</td>
    <td>0.86</td>
   <td>0.86</td>
  </tr>
 <tr>
    <td>*ROBERTA*</td>
    <td>*SICA*</td>
    <td>*0.01*</td>
    <td> *0.98*</td>
    <td>*0.98*</td>
    <td>*0.98*</td>
   <td>*0.98*</td>
  </tr>
</table>



- 가장 성능이 좋은 학습 MAD 모델 : [Download Link](https://drive.google.com/drive/folders/1rfEtKgLVnjhGAgxKPWuguCsc6eHI3vUh) 에서 *checkpoint-SICA-7200.zip* 다운로드해서 `output/A/roberta/` 폴더에 넣고 압축 풀기.

### 5. Segment Character's Action Prediction  
 ```
./inference_cls.sh
```
- inference_n2 각 segment 에 대해 결과 예측해줌.
- 예측결과 : [Download Link](https://drive.google.com/drive/folders/1rfEtKgLVnjhGAgxKPWuguCsc6eHI3vUh) 에서 *inference_roberta_S_I_C_A_inf_4800_n2.lst* 다운로드해서 `inference_n2.tsv` label column 에 넣기

### 6. Evaluation  
