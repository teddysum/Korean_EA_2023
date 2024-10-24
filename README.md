# 감정 분석 Baseline
본 레포지토리는 '2024년 국립국어원 인공지능의 한국어 능력 평가' 상시 과제 중 '감정 분석'에 대한 베이스라인 모델의 학습과 평가를 재현하기 위한 코드를 포함하고 있습니다.

### Baseline
|Model|Micro-F1|
|:---:|---|
|klue/roberta-base|0.850|

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
    --output-dir outputs/ \
    --seed 42 --epoch 10 \
    --learning-rate 2e-5 --weight-decay 0.01 \
    --batch-size 64 --valid-batch-size 64
```

### Inference
```
python -m run inference \
    --model-ckpt-path /workspace/Korean_EA_2023/outputs/<your-model-ckpt-path> \
    --output-path test_output.jsonl \
    --batch-size 64 \
    --device cuda:0
```

### Reference
국립국어원 인공지능 (AI)말평 (https://kli.korean.go.kr/benchmark)  
transformers (https://github.com/huggingface/transformers)  
KLUE (https://github.com/KLUE-benchmark/KLUE)
