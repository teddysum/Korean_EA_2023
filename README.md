# 감정 분석 (2023) Baseline
본 리포지토리는 2023 국립국어원 인공 지능 언어 능력 평가 중 감정 분석(Emotion Analysis)의 베이스라인 모델 및 해당 모델의 재현을 위한 소스 코드를 포함하고 있습니다.
### Baseline
|Model|Micro-F1|Accuracy|
|:---:|---|---|
|klue/roberta-base|0.85749|0.79156|

## Directory Structue
```
resource
└── data

# Executable python script
run
├── infernece.py
└── train.py

# Python dependency file
requirements.txt
```

## Data Format
```
{
    "id": "nikluge-2023-ea-dev-000001",
    "input": {
        "form": "하,,,,내일 옥상다이브 하기 전에 표 구하길 기도해주세요",
        "target": {
            "form": "표",
            "begin": 20,
            "end": 21
        }
    },
    "output": {
        "joy": "False",
        "anticipation": "True",
        "trust": "False",
        "surprise": "False",
        "disgust": "False",
        "fear": "False",
        "anger": "False",
        "sadness": "False"
    }
}
```


## Enviroments
Docker Image
```
docker pull nvcr.io/nvidia/pytorch:22.08-py3 
```

Docker Run Script
```
docker run -dit --gpus all --shm-size=8G --name baseline_ea nvcr.io/nvidia/pytorch:22.08-py3
```

Install Python Dependency
```
pip install -r requirements.txt
```

## How to Run
### Train
```
python -m run train \
    --output-dir outputs/sa \
    --seed 42 --epoch 10 \
    --learning-rate 2e-5 --weight-decay 0.01 \
    --batch-size 64 --valid-batch-size 64
```

### Inference
```
python -m run inference \
    --model-ckpt-path /workspace/Korean_EA_2023/outputs/sa/checkpoint-1212 \
    --output-path test_output.jsonl \
    --batch-size 64 \
    --device cuda:0
```

### Reference
국립국어원 모두의말뭉치 (https://corpus.korean.go.kr/)  
transformers (https://github.com/huggingface/transformers)  
KLUE (https://github.com/KLUE-benchmark/KLUE)
