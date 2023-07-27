# Table-to-Text Baseline
본 리포지토리는 2023 국립국어원 인공 지능 언어 능력 평가 중 Table-to-Text의 베이스라인 모델 및 해당 모델의 재현을 위한 소스 코드를 포함하고 있습니다.  
### Baseline
|Model|Micro-F1|Accuracy|
|:---:|---|---|
|klue/roberta|0.85749|0.79156|

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
docker run -dit --gpus all --shm-size=8G --name baseline_sa nvcr.io/nvidia/pytorch:23.05-py3
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
    --seed 42 --epoch 10 --gpus 2 \
    --learning-rate 2e-4 --weight-decay 0.01 \
    --batch-size=16 --valid-batch-size=16 \
    --wandb-project sa
```

### Inference
```
python -m run inference \
    --model-ckpt-path outputs/sa/<your-model-ckpt-path> \
    --output-path test_output.jsonl \
    --batch-size=64 \
    --device cuda:0
```
