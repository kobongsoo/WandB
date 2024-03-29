# WandB
<img src="https://img.shields.io/badge/Pytorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=white"/><img src="https://img.shields.io/badge/Python-3766AB?style=flat-square&logo=Python&logoColor=white"/></a>

## WandB 테스트 예제
WandB(Weights & Biases)란 더 나은 모델을 빨리 만들 수 있도록, 
loss, Accuracy, HyperPrameter등 을 손쉽게 테스트 하고 시각적으로 확인 할 수 있는, 머신러닝 Experiment tracking tool이다.

### 1. WandB 사전준비
1) [wandb사이트](https://wandb.ai/home)에서 회원 가입 한다.
2) [Setting](https://wandb.ai/settings) 에 들어가서, **API Keys** 를 복사해 둔다.

![image](https://user-images.githubusercontent.com/93692701/163515921-f3ef9abd-a156-40ce-b9a5-52f8a409634f.png)

3) wand 패키지를 설치한다.
```
!pip install wandb -qqq
```

### 2. Pytorch를 이용한 WandB 사용 방법
#### 1) WandB 로그인
- 이때 복사해둔 **API Key 입력**해야 함
- API Key 입력 후 **True** 나오면 로그인 성공
```
import wandb
wandb.login()
```

#### 2) WandB 초기화
- hyper parameter 설정 하고, project명 설정
```
wandb.init(
        project='bert-ft-multiclassification',
        config={
            "epochs": 3,
            "batch_size": 32,
            "learning_rate": 2e-5,
        })

config = wandb.config
```
#### 3) WandB 로그 기록
- 훈련 for 루프에 손실 구하는 코드에 아래 예시 처럼 Loss와 Accuracy 등을 로그에 기록함
 ```
 wandb.log({"Loss": total_loss/p_itr,
             "Accuracy": total_correct/total_len
             })
 ```
4) WandB 종료
- 훈련이 다 끝나면 WandB 종료
```
wnadb.finish()
```

### 3. Pytorch를 이용한 WandB HyperParameter Sweeps 

#### 1) WandB 로그인
- 이때 복사해둔 **API Key 입력**해야 함
- API Key 입력 후 **True** 나오면 로그인 성공
```
import wandb
wandb.login()
```

#### 2) sweep_config 설정
- **name 과 로그(wandb.log)에서 사용하는 명칭은 동일해야 함(val_accuracy)**
```
sweep_config ={ 
        'method': 'random', #grid, random
        'metric':{
            'name': 'val_accuracy', #loss
            'goal': 'maximize'      # loss일때는 minimize로 설정해야함
        },
        'parameters':{
            'learning_rate':{
                'distribution': 'uniform',
                'min' : 0,
                'max': 0.1
            },
            'batch_size':{
                'values' : [8, 16, 32]
            },
            'epochs':{
                'values' : [2, 3, 4]
            }
        }
}

sweep_id = wandb.sweep(sweep_config, project="bert-wb-test")
```
#### 3) train 함수 구현(WandB 초기화)
- WandB sweep 실행할 train 함수 정의
```
def train(config=None):
    # wandb 초기화
    with wandb.init(config=config):
        config = wandb.config
        
        # 데이터 로더 생성
        train_loader, eval_loader = build_dataset(config.batch_size)
        
        # 모델 로딩
        tmodel = load_model()
        
        # 훈련 시작 
        Train_epoch(config, tmodel, train_loader, eval_loader)
```
#### 4) WandB 로그 기록
- 훈련 for 루프에 손실 구하는 코드에 아래 예시 처럼 Loss와 Accuracy 등을 로그에 기록함
 ```
 wandb.log({"val_accuracy": total_test_correct / total_test_len})
 ```

#### 5) WandB seep 실행
```
wandb.agent(sweep_id, train, count=3)
```
### 4. WandB 확인 
- [wandb사이트](https://wandb.ai/home) 로그인
- 해당 Projects 선택
![image](https://user-images.githubusercontent.com/93692701/163515515-b7db7ef4-8bc7-4a58-aeae-5ba22bcc623d.png)

- **HyperParameter Sweeps 인 경우에는 Parallel coordinates 그래프를 추가하면 lr, batch_size, epochs등의 관계를 확인할 수 있다.**
- **[+Add Panel]** 버튼 클릭하여 **Parallel cooridinates** 선택해서 추가하면됨
![image](https://user-images.githubusercontent.com/93692701/163515617-5dc85c67-6032-449c-aa8a-11e33e2a0696.png)

### Huggingface Trainer
- Huggingface Tranier 로 훈련하는 경우 기본이 WandB 사용으로 설정되어 있다.
- 따라서 Tranier 훈련하는 경우 WandB를 Disable 하는 방법은 다음과 같다(2중 하나만 적용하면됨)

```
import os
os.environ["WANDB_DISABLED"] = "true"
```
```
# None disables all integrations
args = TrainingArguments(report_to=None, ...)
```


