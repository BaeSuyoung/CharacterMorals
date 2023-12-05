# Ethical Dynamics in Fictional Characters: A Framework for Monitoring Moral Actions in Narrative Texts  

## Abstract  
- AI system 으로 이야기 캐릭터의 윤리적 성격을 분석하는 방법론
- 이야기의 자동 분석은 흥미로운 이야기를 선정하는 데 도움을 줄 수 있는 도구 역할을 한다.
- 매일 수많은 이야기가 새로 창작되고, 스튜디오와 엔터테인먼트 전문가들은 이야기 케스팅과 마케팅 시 모든 이야기를 직접 읽기에는 시간과 비용이 많이 들기 때문에 자동 이야기 분석 도구를 통해 빠르게 흥미로운 이야기를 선별하는 것은 유용하다.
- 흥미로운 이야기의 중요한 요소로 캐릭터는 중요한 역할을 한다. 캐릭터들의 행동이 독자들에게 호감을 가는 지, 확실한 악역, 확실한 영웅이 있는지에 대한 분석이 필요한데 이런 등장인물의 역할 구분은 결국 윤리적 성격과 관련이 있다.
- 등장인물의 윤리성을 자동으로 판단할 수 있는 AI system 은 어려운 문제다.
- 윤리성은 인간이 판단할 수 있는 복잡한 성질로 사람들은 사회적 기준과 일치하는 행동과 옳고 그름의 행위를 윤리적인 행위라고 판단할 수 있다.
- 또한 윤리성은 같은 행동이라도 주인공이 처한 상황에 따라 윤리적 판단이 달라질 수 있다. 따라서 행동 자체보다는 행동에 대한 주변 상황을 함께 고려해 윤리성을 판단할 수 있어야 한다.
- 본 프로젝트에서는 AI 모델을 사용해서 문맥을 반영해 이야기 등장인물의 윤리성을 예측할 수 있는 방법론을 제안한다.  
![Framework](https://github.com/BaeSuyoung/CharacterMorals/blob/main/framework.png)

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
    <td>ROBERTA</td>
    <td>SICA</td>
    <td>0.01</td>
    <td>0.98</td>
    <td>0.98</td>
    <td>0.98</td>
   <td>0.98</td>
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
<table>
  <tr>
    <th></th>
    <th>ChatGPT</th>
    <th>Ours(A)</th>
    <th>Ours(SA)</th>
    <th>Ours(SIA)</th>
    <th>Ours(SICA)</th>
  </tr>
  <tr>
    <td>Accuracy</td>
    <td>82.56</td>
    <td>89.53</td>
    <td>93.02</td>
    <td>96.51</td>
    <td>89.53</td>
  </tr>
</table>

![Human Evaluation Result](https://github.com/BaeSuyoung/CharacterMorals/blob/main/Human%20Evaluation%20Result.png)
