# Reliable-AI_Assignment4
 
신경망 검증(Neural Network Verification) 도구인 **Marabou**와 **alpha-beta-CROWN**을 이용해, 동일한 MNIST 분류기(MLP)에 대한 **적대적 견고성(adversarial robustness)** 을 검증하고 두 도구의 결과·속도를 비교하는 과제입니다.
 
## 개요
 
- **대상 모델**: 784 → 32 (ReLU) → 10 구조의 단순 MLP (`MLP` 클래스, `fc1` → `ReLU` → `fc2`)
- **데이터셋**: MNIST
- **검증 속성**: 입력에 L-∞ 노름 기준 epsilon 크기의 perturbation(섭동)을 가했을 때, 모델이 원래 예측 클래스를 계속 유지하는지(= 다른 클래스로 오분류되는 반례가 존재하지 않는지)
- **비교 대상 검증기**
  - [Marabou](https://github.com/NeuralNetworkVerification/Marabou) — SMT 기반 완전(complete) 검증기 (`verify.marabou.py`)
  - [alpha-beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN) — Branch-and-Bound 기반 검증기, git submodule로 포함 (`config.yaml`)
자세한 실험 배경과 결과 분석은 [`report.pdf`](./report.pdf)를 참고하세요.
 
## 디렉토리 구조
 
```
.
├── alpha-beta-CROWN/      # git submodule (Verified-Intelligence/alpha-beta-CROWN)
├── config.yaml            # alpha-beta-CROWN 검증 설정
├── verify.marabou.py      # Marabou를 이용한 검증 스크립트 (PyTorch → ONNX 변환 포함)
├── test.py                # MLP 모델 정의 / 학습 / build_model (alpha-beta-CROWN의 Customized 모델 로더용)
├── mlp_mnist.pth           # 학습된 PyTorch 모델 가중치
├── mlp_mnist.onnx           # ONNX로 변환된 모델 (verify.marabou.py 실행 시 생성)
├── mlp_mnist.onnx.data       # ONNX 외부 가중치 데이터
├── requirements.txt        # Python 의존성 목록
└── report.pdf              # 실험 결과 보고서
```
 
## 환경 설정
 
### 1. 저장소 클론 (submodule 포함)
 
```bash
git clone --recurse-submodules https://github.com/Doa-ddaram/Reliable-AI_Assignment4.git
cd Reliable-AI_Assignment4
```
 
이미 일반 clone을 했다면:
 
```bash
git submodule update --init --recursive
```
 
### 2. 의존성 설치
 
```bash
pip install -r requirements.txt
```
 
> `torch`, `torchvision`은 `requirements.txt`에 직접 명시되어 있지 않지만, alpha-beta-CROWN 자체 의존성을 설치하는 과정에서 함께 설치됩니다.
> `gurobipy`는 Gurobi 라이선스가 있을 경우 alpha-beta-CROWN의 일부 솔버 가속에 사용됩니다(필수는 아님).
 
alpha-beta-CROWN 자체의 의존성은 submodule 디렉토리 안의 안내를 따로 참고하세요.
 
```bash
cd alpha-beta-CROWN
pip install -e .
cd ..
```
 
## 실행 방법
 
### Marabou로 검증하기
 
```bash
python verify.marabou.py
```
 
동작 순서:
1. `mlp_mnist.pth`를 불러와 `mlp_mnist.onnx`로 변환(export)
2. MNIST 테스트셋의 첫 번째 이미지에 대해, epsilon(`L∞` 기준, 기본값 0.01) 범위 내 모든 perturbation에서 **다른 클래스로 오분류될 수 있는지**를 클래스별로 Marabou에 질의
3. 하나라도 `sat`(반례 존재)이면 `UNSAFE`, 모든 클래스가 `unsat`이면 `SAFE`로 출력하고 검증 소요 시간을 함께 출력
epsilon 값을 바꾸려면 `verify.marabou.py` 맨 아래의 `verify_with_marabou(onnx_model_path, epsilon=0.01)` 호출부를 수정하세요.
 
### alpha-beta-CROWN으로 검증하기
 
```bash
cd alpha-beta-CROWN/complete_verifier
python abcrown.py --config ../../config.yaml
```
 
`config.yaml` 주요 설정:
 
| 항목 | 값 | 설명 |
|---|---|---|
| `model.path` | `./mlp_mnist.pth` | 검증 대상 가중치 |
| `data.dataset` | `MNIST` | 자동 다운로드됨 |
| `data.start` / `data.end` | `0` / `50` | 검증할 테스트 이미지 인덱스 범위 |
| `specification.epsilon` | `0.01` | L∞ 견고성 반경 |
| `bab.timeout` | `120` | 이미지당 Branch-and-Bound 타임아웃(초) |
 
> `model.name`은 `Customized("<모듈명>", "build_model")` 형식으로, alpha-beta-CROWN이 `build_model` 함수를 불러올 Python 모듈(파일)명을 가리킵니다. 현재 `build_model` 정의가 `test.py`에 있으므로 모듈명도 `"test"`로 맞춰져 있어야 합니다.
 
### 모델 학습 / 재현 (선택)
 
`mlp_mnist.pth`가 이미 포함되어 있어 바로 검증을 실행할 수 있지만, 처음부터 다시 학습하려면 `test.py`의 학습 로직을 실행하면 됩니다(10 epoch, Adam optimizer, lr=0.005).
 
## 실험 결과 요약
 
`report.pdf` 기준, 동일 MLP 모델(2-layer)에 대해 epsilon을 0.01 / 0.05로 바꿔가며 alpha-beta-CROWN으로 10개 샘플을 검증한 결과:
 
| epsilon | 테스트 데이터 수 | 검증 성공 | 방어 실패(반례 발견) | 평균 검증 시간 | 최대 검증 시간 |
|---|---|---|---|---|---|
| 0.01 | 10 | 10 | 0 | 0.563초 | 1.211초 |
| 0.05 | 10 | 0 | 10 | 0.081초 | 0.597초 |
 
- epsilon이 작을수록(0.01) 더 많은 샘플에서 견고성이 입증됨
- epsilon이 클 경우(0.05) 입력이 결정 경계를 쉽게 넘어가 모든 샘플에서 반례(PGD 공격 기반 조기 발견)가 나타나며, 오히려 검증 시간은 더 짧음
- 검증 대상 샘플 수를 50개로 늘렸을 때도 epsilon 0.01에서는 단 1개 샘플에서만 반례 발견
자세한 수치와 Marabou와의 비교 분석은 `report.pdf`를 참고하세요.
 
## 참고
 
- 비교 실험의 목적상(검증기 자체의 구조적 차이·연산 효율성 비교에 집중하기 위해) 데이터셋은 Fashion-MNIST가 아닌 표준 MNIST로 통일했습니다.
- Marabou 검증(`verify.marabou.py`)은 완전(complete) 검증 결과(`sat`/`unsat`)를 제공하며, alpha-beta-CROWN은 PGD 공격 기반 반례 탐색과 Branch-and-Bound 기반 완전 검증을 함께 사용합니다.
 