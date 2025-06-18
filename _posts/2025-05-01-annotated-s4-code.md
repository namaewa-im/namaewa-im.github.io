---
title: "[Annotated S4] 코드 실행 방법 및 코드 리뷰"
date: 2025-05-01 21:00:00 +0900
math: true
categories: [code]
description: "S4 코드 리뷰"
tags: [SSM, S4, Annotated-S4, Docker, JAX,]
---

> 본 글은 Annotated-S4([링크](https://srush.github.io/annotated-s4/))의 코드 실행 방법과 핵심 파일을 설명하는 글입니다.

---

[Annotated S4](https://github.com/srush/annotated-s4)는 Structured State Space 모델(S4)을 구현한 JAX/Flax 기반 코드입니다.  
이 글에서는 해당 코드를 로컬 또는 Docker 환경에서 실행하기 위한 방법과, 주요 구성 파일인 `s4.py`, `train.py`에 대한 구조 및 역할을 정리합니다.

---

## 코드 다운로드 및 설치

### 사전 준비

- **Docker 사용**:
  - [pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel](https://hub.docker.com/r/pytorch/pytorch/) 기반 이미지 추천
  - GPU 사용 가능 환경 필요 (e.g. NVIDIA Container Toolkit)
  - 컨테이너 실행 시: `/workspace` 디렉토리 생성, `--gpus all` 플래그 사용

- **Local (Conda 환경) 사용**:
  ```bash
  conda create -n s4_env python=3.10
  conda activate s4_env
	```

### 🔽 코드 클론 및 디렉토리 이동
Annotated-S4 저장소를 클론한 후, 작업 디렉토리로 이동합니다:
```bash
git clone https://github.com/srush/annotated-s4.git
cd annotated-s4
```

### 의존성 설치
공식 [requirements.txt](https://github.com/srush/annotated-s4/blob/main/requirements-gpu.txt)

> 제가 구성한 [requirements.txt](/assets/data/s4/annotated-s4.txt)도 제공하겠습니다. 하나의 예시이므로 실제 사용 시 환경에 맞게 수정이 필요할 수도 있습니다. python 3.7.11

---

#### 실행 예시
MNIST Classification
```bash
python -m s4.train dataset=mnist layer=s4 train.epochs=100 train.bsz=128 model.d_model=128 model.layer.N=64
CIFAR Classification
```

```bash
python -m s4.train dataset=cifar-classification layer=s4d train.epochs=100 train.bsz=50 model.n_layers=6 model.d_model=512 model.dropout=0.25 train.lr=5e-3 train.weight_decay=0.01 train.lr_schedule=true seed=1 +model.layer.scaling=linear
```
### 📂 LRA Dataset Mount

> LRA dataset 중 Annotated-s4가 기본으로 제공하는 것은 MNIST Classification, CIFAR-10 Classification 입니다. listops, aan, pathfinder를 실행하기 위해서는 [LRA 벤치마크 데이터셋을 다운](https://namaewa-im.github.io/posts/lra/#how-to-download-dataset)받아야합니다.

LRA 벤치마크 데이터셋을 로컬 또는 Docker 환경에 마운트합니다:

#### 다운받을 로컬 위치
~/Downloads/lra_release/lra_release

#### Docker 실행 코드
```bash
docker run -it --gpus all -v ~/Downloads/lra_release/lra_release:/workspace/lra_release <image>:<tag> bash
```
데이터셋이 저장되는 컨테이너 내부 경로는 /workspace/lra_release입니다.

#### 실행 예시
```bash
python -m train dataset=listops-classification +dataset.data_dir=/workspace/lra_release/listops-1000
``` 

#### imdb
imdb의 arrow 데이터를 가지고 있다면 /annotated-s4/s4/data.py의 create_imdb_classification_dataset()을 다음과 같이 외부 데이터를 경로에서 데이터를 읽어오도록 수정할 수 있습니다:
```python
def create_imdb_classification_dataset(bsz=128):
    # Constants, the default max length is 4096
    APPEND_BOS = False
    APPEND_EOS = True
    LOAD_WORDER = 20
    MIN_FREQ = 15

    SEQ_LENGTH, N_CLASSES, IN_DIM = 2048, 2, 135

    # 외부 데이터 경로 설정
    data_path = "/workspace/s4/data/imdb/imdb/plain_text/0.0.0/e6281661ce1c48d982bc483cf8a173c1bbeb5d31/"
    
    # 외부 데이터 로드
    train_data = load_dataset("arrow", data_files=f"{data_path}imdb-train.arrow")
    test_data = load_dataset("arrow", data_files=f"{data_path}imdb-test.arrow")
    
    # DatasetDict 생성
    dataset = DatasetDict(
        train=train_data["train"],
        test=test_data["train"]  # arrow 파일은 기본적으로 "train" 스플릿을 가짐
    )

    # train 데이터의 10%를 validation으로 분할
    train_val = dataset["train"].train_test_split(test_size=0.1, seed=42)
    dataset = DatasetDict(
        train=train_val["train"],
        val=train_val["test"],
        test=dataset["test"]
    )

    l_max = SEQ_LENGTH - int(APPEND_BOS) - int(APPEND_EOS)

    # step one, byte level tokenization
    dataset = dataset.map(
        lambda example: {"tokens": list(example["text"])[:l_max]},
        remove_columns=["text"],
        keep_in_memory=True,
        load_from_cache_file=False,
        num_proc=max(LOAD_WORDER, 1),
    )

    # print("byte characters for first example:", dataset['train']['tokens'][0])

    # step two, build vocabulary based on the byte characters, each character appear at least MIN_FREQ times
    vocab = torchtext.vocab.build_vocab_from_iterator(
        dataset["train"]["tokens"],
        min_freq=MIN_FREQ,
        specials=(
            ["<pad>", "<unk>"]
            + (["<bos>"] if APPEND_BOS else [])
            + (["<eos>"] if APPEND_EOS else [])
        ),
    )

    # step three, numericalize the tokens
    vocab.set_default_index(vocab["<unk>"])

    dataset = dataset.map(
        lambda example: {
            "input_ids": vocab(
                (["<bos>"] if APPEND_BOS else [])
                + example["tokens"]
                + (["<eos>"] if APPEND_EOS else [])
            )
        },
        remove_columns=["tokens"],
        keep_in_memory=True,
        load_from_cache_file=False,
        num_proc=max(LOAD_WORDER, 1),
    )

    # print("numericalize result for first example:", dataset['train']['input_ids'][0])

    dataset["train"].set_format(type="torch", columns=["input_ids", "label"])
    dataset["test"].set_format(type="torch", columns=["input_ids", "label"])

    def imdb_collate(batch):
        batchfy_input_ids = [data["input_ids"] for data in batch]
        batchfy_labels = torch.cat(
            [data["label"].unsqueeze(0) for data in batch], dim=0
        )
        batchfy_input_ids = torch.nn.utils.rnn.pad_sequence(
            batchfy_input_ids + [torch.zeros(SEQ_LENGTH)],
            padding_value=vocab["<pad>"],
            batch_first=True,
        )
        batchfy_input_ids = torch.nn.functional.one_hot(
            batchfy_input_ids[:-1], IN_DIM
        )
        return batchfy_input_ids, batchfy_labels

    trainloader = torch.utils.data.DataLoader(
        dataset["train"], batch_size=bsz, shuffle=True, collate_fn=imdb_collate
    )

    testloader = torch.utils.data.DataLoader(
        dataset["test"], batch_size=bsz, shuffle=True, collate_fn=imdb_collate
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM
```
만약 이런 에러가 뜬다면
```bash
TypeError: An invalid dataloader was returned from SequenceLightningModule.val_dataloader(). Found None.
```
dataset config에서 val_split: 0.1로 설정합니다.

#### Pathfinder, Path-X, AAN
> Pathfinder, Path-X, AAN는 annotated-s4/s4/data.py에서 따로 제공하고 있지 않기 때문에 해당 데이터셋을 사용하기 위해서는 create_{pathfinder, pathx, aan}_dataset()을 직접 만들어야합니다. 

다음은 create_pathfinder_dataset()의 예시입니다. annotated-s4/s4/data.py 아래에 다음과 같이 작성합니다:

```python
def create_pathfinder_dataset(bsz=128, n_workers=2):
    import os
    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader, random_split
    from torchvision import transforms
    from PIL import Image
    from pathlib import Path

    blacklist = {"/workspace/lra_release/pathfinder32/curv_baseline/imgs/0/sample_172.png"} # empty data

    base_path = "/workspace/lra_release/pathfinder32/curv_baseline"
    metadata_path = os.path.join(base_path, "metadata")
    imgs_base_path = os.path.join(base_path, "imgs")

    print(f"Loading Pathfinder32 from {base_path}")

    # Step 1: 메타데이터 파싱
    samples = []
    for fname in sorted(os.listdir(metadata_path), key=lambda f: int(f.split('.')[0])):
        with open(os.path.join(metadata_path, fname), "r") as f:
            for line in f:
                parts = line.strip().split()
                img_dir = parts[0]      # e.g. imgs/0
                img_file = parts[1]     # e.g. sample_0.png
                label = int(parts[3])   # 0 or 1
                img_path = os.path.join(base_path, img_dir, img_file)

                # 블랙리스트 체크
                if img_path in blacklist:
                    continue
                
                samples.append((img_path, label))

    # [Debug] 클래스 분포 확인
    print("\n[Debug] Class Distribution:")
    unique_labels, counts = np.unique([s[1] for s in samples], return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Class {label}: {count} samples ({count/len(samples)*100:.2f}%)")

    # Step 2: Dataset 정의
    class PathfinderDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples
            self.transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),                  # (1, 32, 32)
                transforms.Lambda(lambda x: x.view(-1, 1))  # Flatten to (1024, 1)
            ])

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            img_path, label = self.samples[idx]
            img = Image.open(img_path).convert("L")
            img = self.transform(img)
            return img, torch.tensor(label, dtype=torch.long)

    # Step 3: train/val/test split
    full_dataset = PathfinderDataset(samples)
    
    # [Debug] 데이터셋 샘플 확인
    print("\n[Debug] Dataset Sample Check:")
    sample_img, sample_label = full_dataset[0]
    print(f"Sample image shape: {sample_img.shape}")
    print(f"Sample label: {sample_label}")
    print(f"Sample image min/max: {sample_img.min()}, {sample_img.max()}")
    print(f"Sample image mean/std: {sample_img.mean()}, {sample_img.std()}")
    
    val_len = int(0.1 * len(full_dataset))
    test_len = int(0.1 * len(full_dataset))
    train_len = len(full_dataset) - val_len - test_len
    train_set, val_set, test_set = random_split(full_dataset, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(42))

    # [Debug] 데이터셋 크기 확인
    print("\n[Debug] Dataset Sizes:")
    print(f"Train set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")
    print(f"Test set size: {len(test_set)}")

    # Step 4: DataLoader 구성
    trainloader = DataLoader(train_set, batch_size=bsz, shuffle=True, num_workers=n_workers, pin_memory=True)
    testloader = DataLoader(test_set, batch_size=bsz, shuffle=False, num_workers=n_workers, pin_memory=True)

    # [Debug] DataLoader 샘플 확인
    print("\n[Debug] DataLoader Sample Check:")
    train_batch = next(iter(trainloader))
    print(f"Train batch images shape: {train_batch[0].shape}")
    print(f"Train batch labels shape: {train_batch[1].shape}")
    print(f"Train batch labels: {train_batch[1][:5]}")  # 첫 5개 라벨 출력

    SEQ_LENGTH = 32 * 32
    IN_DIM = 1
    N_CLASSES = 2

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


Datasets = {
    "mnist": create_mnist_dataset,
    "quickdraw": create_quickdraw_dataset,
    "fsdd": create_fsdd_dataset,
    "sc": create_sc_dataset,
    "sin": create_sin_x_dataset,
    "sin_noise": create_sin_ax_b_dataset,
    "mnist-classification": create_mnist_classification_dataset,
    "fsdd-classification": create_fsdd_classification_dataset,
    "cifar-classification": create_cifar_classification_dataset,
    "imdb-classification": create_imdb_classification_dataset,
    "listops-classification": create_listops_classification_dataset,
    "pathfinder": create_pathfinder_dataset, # 데이터셋 추가
}

```

다음을 실행하여 코드가 동작하는 지 확인합니다:
```bash
python -m train dataset=pathfinder
``` 


### 결과 시각화
wandb를 사용해 학습 과정을 시각화할 수 있습니다. wandb 아이디를 만든 다음 설치 및 터미널 창에서의 로그인 및 init을 실행합니다:
```bash
pip install wandb
wandb login
wandb init
```
그 후 다음 코드를 실행합니다:
```bash
python -m s4.train dataset=listops-classification wandb.mode=online wandb.project=annotated-s4
```
train.py 실행 시 로그인한 계정에 자동으로 로깅되어 실시간으로 그래프를 확인할 수 있습니다:

#### ListOps

```bash
python -m s4.train dataset=listops-classification layer=s4 model.layer.N=64 model.n_layers=6 model.d_model=128 model.prenorm=False model.dropout=0.0 train.lr=0.01 train.bsz=100 train.epochs=50 train.weight_decay=0.01 wandb.mode=online wandb.project=Annotated-s4-listops train.lr_schedule=true
```
![listops](/assets/img/annotated-s4/listops1.webp)

#### IMDb
```bash
python -m s4.train dataset=imdb-classification layer=s4 model.d_model=64 model.n_layers=4 model.layer.N=64  model.prenorm=True model.dropout=0.0 train.lr=1e-3 train.bsz=50 train.epochs=20 train.weight_decay=0.01 train.lr_schedule=true seed=1 wandb.mode=online wandb.project=Annotated-s4-imdb
```
![imdb](/assets/img/annotated-s4/imdb1.webp)

#### Cifar-10
```bash
python -m s4.train dataset=cifar-classification layer=s4 model.d_model=512 model.n_layers=6 model.layer.N=64 model.prenorm=False model.dropout=0.25 train.lr=5e-3 train.bsz=50 train.epochs=200 train.weight_decay=0.01 train.lr_schedule=true seed=1 wandb.mode=online wandb.project=Annotated-s4-cifar10
```
![cifar10](/assets/img/annotated-s4/cifar1.webp)

#### Pathfinder
```bash
python -m s4.train dataset=imdb-classification layer=s4 model.layer.N=64 model.n_layers=4 model.d_model=64 model.prenorm=True model.dropout=0.0 train.lr=5e-3 train.bsz=50 train.epochs=100 train.weight_decay=0.01 train.lr_schedule=true seed=1 wandb.mode=online wandb.project=Annotated-s4-imdb
```
![pathfinder](/assets/img/annotated-s4/pathfinder1.webp)


## 코드 구조 리뷰
Annotated-S4 코드에서 가장 중요한 두 파일인 [s4.py](https://github.com/srush/annotated-s4/blob/main/s4/s4.py)와 [train.py](https://github.com/srush/annotated-s4/blob/main/s4/train.py)의 구조와 작동 방식을 살펴봅니다.

### s4.py

s4.py는 3개의 주요 층위로 구성됩니다.

#### 1. 이론 구현 레벨
SSM의 기본 연산

> **random_SSM(rng, N):** A, B, C 생성  
**discretize(A, B, C, step):** Ab, Bb, C  
**scan_SSM(Ab, Bb, Cb, u, x0):** recurrence 처리  
**run_SSM(A, B, C, u):** 전체 시퀀스 처리  

##### 1-1. random_SSM
```
input: rng, N
return: A, B, C
```
##### 1-2. discretize
```
input: A, B, C, step
return: Ab, Bb, C
```
##### 1-3. scan_SSM
```
input: Ab, Bb, Cb, u, x0
return: jax.lax.scan(step, x0, u)
```
###### 1-3-1. step
```
input: x_k_1, u_k
return x_k, y_k
```
##### 1-4. run_SSM
```
input: A, B, C, u
return: y
```

#### 2. 신호 처리 레벨 
주파수 영역에서의 K 계산

> **K_conv(Ab, Bb, Cb, L):** 커널 생성  
**causal_convolution(u, K):** 컨볼루션 계산  
**K_gen_DPLR(...), kernel_DPLR(...):** DPLR 기반 커널 생성  
**discrete_DPLR(...):** 커널 이산화  

##### 2-1. K_conv
```
input: Ab, Bb, Cb, L
return: K
```
##### 2-2. causal_convolution
```
input: u, K
return: y 
```
##### 2-3. K_gen_DPLR
```
input: Lamdba, P, Q, B, C, step
return: gen
```
###### 2-3-1. cauchy_dot
```
input: a, g, Lambda
out: gen
```
##### 2-4. kernel_DPLR
```
input: Lambda, P, Q, B, C, step, L
return: y.real
```

##### 2-5. discrete_DPLR
```
input: Lambda, P, Q, B, C, step, L
return: Ab, Bb, Cb.conj()
```

#### 3. Neural Network 레벨
S4를 Flax 모델로 통합

> **S4Layer:** training mode/decode mode 모드 지원  
**cloneLayer:** H개의 S4Layer 복제 지원  
**SequenceBlock:** S4Layer + Dropout + Dense를 묶은 블록  
**StackedModel:** 여러 레이어를 쌓고, encoder/decoder 구성  
**BatchStackedModel:** 배치사이즈 B개로의 복제 지원

##### 3-1. SSMLayer
```
setup: A, B, C, D, log_step, ssm, k, x_k_1
call: if not decode: causal_convolution(u,k)+D*u else: scan_SSM으로 y_s.real+D*u
```
##### 3-2. cloneLayer
SSMLayer나 S4Layer를 layer dim H개로 복제 (L) -> (H,L)

##### 3-3. SequenceBlock
```
setup: seq, layer_cls, norm, out, glu, out2, drop, ...
call: norm-> seq->drop-> glu-> skip+dorp-> norm
```
##### 3-4. StackedModel
```
setup: encoder (if embedding: Embedding else: Dense), decoder, layers = (SequenceBlock())
call: classfication if embedding: embedding else: decode
encoder-> layer-> decoder-> log_softmax
```
##### 3-5. BatchStakedModel
StackedModel을 배치사이즈 B개로 복제 (H,L) -> (B, H, L)

##### 3-6. S4Layer
```
setup: Lambda_re, Lambda_im, Lambda, P, B, C, D, step
if not decode: K else: discrete_DPLR
call: if not decode: causal_convolution else: scan_SSM
```
### train.py

#### train.py 구조
모델 학습과 평가를 위한 루틴이 포함되어 있습니다.

> **create_train_state()**: 모델 초기화 및 옵티마이저 구성  
**train_epoch()**: 1 epoch 학습 루프 수행  
**validate()**: 검증 루프 수행  
**train_step() / eval_step()**: JIT 기반 1 step 학습/검증 처리  
**example_train()**: 전체 학습 과정 조립 및 수행  

##### 1. create_train_state: 모델 파라미터 초기화 및 옵티마이저 설정
```
input: rng, model_cls, trainloader, lr, lr_layer, lr_schedule, weight_decay, total_steps=1
return: flax.training.train_state.TrainState(model.apply, params, tx)
```
##### 2. train_epoch: 에폭 단위 학습
```
input: state, rng, model, trainloader, classification=False
return: state, batch_losses, batch_accuracies
```
##### 3. validate: 에폭 단위 검증
```
input: params, model, testloader, classification=False
return: loss, acc
```
##### 4. train_step: @partial(jax.jit, static_argnums=(4,5))
```
input: state, rng, batch_inputs, batch_labels, model, classification=False 
return: state, loss, acc
```
##### 5. eval_step: @partial(jax.jit, static_argnums=(3,4))
```
input: batch_input, batch_labels, params, model, classification=False
return: loss, acc
```
##### 6. example_train: 
```
input: dataset, layer, seed, model_cfg, train_cfg
return: None
```

##### example_train의 입출력/코드 흐름

**입력:** Hydra로부터 받은 cfg: DictConfig  
**출력:** None(모델 학습 및 wandb logging, checkpoint 저장)

[init]  
    - torch/jax random seed 설정  
    - 데이터셋 로딩 (Datasets[dataset])  
    - 모델 클래스 결정 (Models[layer])  
    - 모델 설정 값 수정 (model.layer.l_max 등)  
    - 모델 partial 생성 (BatchStackedModel(layer_cls, ...))  

[Model init]  
    - create_train_state()로 파라미터 초기화 + 옵티마이저 설정

[epoch loop]  
for epoch:  
    - train_epoch() 호출 → train_step 반복  
    - validate() 호출 → eval_step 반복  
    - wandb log, checkpoint 저장, 샘플링 수행  

#### 전체 코드 흐름
config.yaml -> main() -> example_train() -> create_train_state() -> train_epoch() -> validate()
