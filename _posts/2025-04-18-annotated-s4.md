---
title: "The Annotated S4 이해하기 ②: HiPPO 초기화부터 S4의 효율적인 커널 계산까지"
date: 2025-04-18 18:00:00 +0900
description: "본 글은 S4에 대한 이해를 돕기 위해 Annotated-S4를 바탕으로 작성된 글입니다."
math: true
image:
  path: /assets/img/s4/2025-04-17-thumb.jpg
  alt: "S4"
categories: [SSM, S4, HiPPO, Sequence Modeling]
tags: [SSM, S4, HiPPO, Annotated-S4, LRA]
---

> 본 글은 Annotated-S4([링크](https://srush.github.io/annotated-s4/))를 바탕으로 상태공간 모델을 활용한 딥러닝 모델 S4의 기초를 정리한 내용입니다.  
> 모든 내용은 [Annotated-S4 Github 문서](https://srush.github.io/annotated-s4/)와 [그에 대한 코드](https://github.com/srush/annotated-s4)에 기반하고 있습니다. 
> SSM에 대한 기본적인 이해가 있다고 가정합니다.

---

## S4 단순 구현의 두가지 문제
> [이전 글](https://namaewa-im.github.io/posts/annotated-s4-1/)에서 S4를 단순하게 구현했을 때 발생하는 두 가지 핵심 문제를 정의했습니다.

#### 문제 1. 랜덤 초기화된 SSM은 성능이 좋지 않다
랜덤한 파라미터로 초기화된 SSM은 학습 성능이 매우 불안정하며, 시퀀스가 길어질수록 gradient가 기하급수적으로 확장되거나 소멸하여 안정적인 학습이 어렵습니다.

#### 문제 2. 단순한 커널 계산은 느리고 메모리 비효율적이다
커널 $K$를 naive하게 계산할 경우 $A$에 대해 $L$번의 행렬 곱 연산이 필요하고, 이는 $\mathcal{O}(N^2 L)$의 연산량과 $\mathcal{O}(NL)$의 메모리를 요구합니다.

## S4 논문이 제안한 해결방법
### 문제 1의 해결 - Part 1b. HiPPO 메모리 사용
> Annotated-s4 문서의 Part 1b에 대한 설명입니다.
HiPPO(Hierarchical Orthogonal Polynomial Projection)는 연속시간에서 히스토리를 압축하는 수학적 이론입니다.
> 이론이라기보다는 제안된 프레임워크

입력 $u$의 과거 히스토리를 압축하여 상태 $x$에 보존되도록 하는 것이 목표입니다. $x$는 르장드르 다항식의 계수를 추적하며, 이를 통해 과거의 모든 값을 근사화합니다. 매 단계마다 이 계수를 업데이트함으로써, $x$는 **입력의 연속적인 기억(memory)**을 표현합니다.

이 HiPPO 메모리를 사용하여 SSM의 $A$ 행렬을 초기화했을때 다음과 같은 이점을 얻을 수 있었습니다.

- 학습 전에 $A$를 한 번 계산하는 것으로 랜덤 초기화 시 60% 수준이던 정확도가, HiPPO 초기화만으로 98%까지 향상되었습니다. 
- ODE를 직접 풀지 않고도 시퀀스 압축의 효과를 얻을 수 있습니다.

정리하자면, 이산화된 SSM을 초기화하기 위해 HiPPO라는 연속시간 시퀀스 압축 이론을 사용하여 학습 안정성과 성능 향상을 이룰 수 있습니다.

### 문제 2의 해결 - Part 2. S4 구현
> Annotated-s4 문서의 Part 2에 대한 설명입니다.

Part 2의 목표는 커널 $K$를 빠르고 안정적으로 계산하는 방법을 찾는 것입니다.

$A$를 다음과 같은 형태로 가정합니다.

$$A = \Lambda + PQ^*$$

$\Lambda$는 [diagonal matrix](https://en.wikipedia.org/wiki/Diagonal_matrix)이고 $P, Q$는 복소수인 Low-rank 항입니다. 이를 DPLR(Diagonal Plus Low Rank)라고 부릅니다.

$$\Lambda \in \mathbb{C}^{N \times N}$$
$$P \in \mathbb{C}^{N \times r}$$
$$Q \in \mathbb{C}^{N \times r}$$
$$B \in \mathbb{C}^{N \times 1}$$
$$C \in \mathbb{C}^{1 \times N}$$

> Q. 왜 $A$를 DPLR이라고 가정하나요?
> A. 이후에 나오는 수학적 기술들을 사용하기 위함입니다.
> Q. 왜 갑자기 행렬 공간을 $\mathbb{C}$로 확장하나요?
> A. 실수 행렬이라도 고유값이 복소수일 수 있습니다. 이 경우, 실수 공간에서는 대각화가 불가능한 경우가 많습니다. 하지만 복소수 공간에서는 모든 normal 행렬이 unitary하게 diagonalizable하기 때문에 효율적인 커널 계산을 위해 행렬을 복소수 공간으로 확장하는 것입니다.

DPLR S4는 다음 3단계를 통해 연산 속도 병목을 극복할 수 있습니다. 

**1. Truncated Generating Function 평가**
커널 $K$를 직접 계산하지 않고 truncated generating function을 [evaluate]()하는 방식으로 [specturm]()을 계산합니다.

**2. Cauchy 커널로 표현**
Diagonal matrix case는 다음과 같은 [Cauchy 커널](https://en.wikipedia.org/wiki/Cauchy_matrix)의 형태로 변환이 가능하다는 것을 이용합니다.

$$
C_{kj} = \frac{1}{\omega_j - \zeta_k}
$$

**3. Low-rank 보정**  
Low-rank term이 [Woodbury Identity](링크)를 통해 $\Lambda + P Q^*$가 $\Lambda^{-1}$와 rank-1 보정 항으로 표현될 수 있음을 이용합니다.  

Cauchy 커널로 표현하면 Cauchy matrix-vector 곱의 성질을 활용해 **truncated generating function $\hat{K}_L(z)$**를 빠르게 계산할 수 있으며, $\Lambda^{-1}$ 구조로 정리하면 [FFT](링크)를 이용해 전체 커널 계산을 $O(N \log N)$까지 줄일 수 있습니다.

#### Step 1. Truncated Generating Function
S4는 커널 계산을 시간 도메인에서 직접 수행하지 않고, **생성 함수(generating function)**를 도입하여 주파수 도메인에서 효율적으로 계산합니다.

여기서 생성함수란 필터 $K$를 시간 도메인이 아닌 z-도메인에서 표현하는 도구입니다. 특히, 단위원 위의 복소수 노드 $z = e^{2\pi i k / L}$ (roots of unity)에서 평가하면 **DFT (Discrete Fourier Transform)**을 계산하는 것과 동일합니다.

$$\tilde{K}(z) = \tilde{C}(zI-\tilde{A})^{-1}\bar{B}$$

위 수식은 truncated SSM generating function으로, $A$를 거듭제곱할 필요 없이 $K$를 주파수 영역에서 정의합니다. 다시 말해 시간영역에서 $A$의 거듭제곱을 주파수 영역에서 $A$를 한번 역행렬 계산하는 것으로 대체합니다.

$\tilde{K}(z)$를 FFT 주파수 노드 $z \in \Omega_L$에서 평가하고, 이를 IFFT (역 푸리에 변환)하면 $O(L\text{log}L)$에서 시간 도메인 커널 $K$를 복원할 수 있습니다.

$$
\Omega = \left\{ \exp\left(2\pi i \frac{k}{L} \right) : k \in [L] \right\}
$$

> 디테일한 내용은 논문 부록 C.3에 있습니다.

다음과 같이 행렬곱이 역연산으로 바뀐다는게 핵심입니다:

$$
\hat{K}_L(z) = \sum_{i=0}^{L-1} \bar{C} \bar{A}^i \bar{B} z^i = \bar{C} (I - \bar{A}^L z^L)(I - \bar{A} z)^{-1} \bar{B} = \tilde{C} (I - \bar{A} z)^{-1} \bar{B}
$$

$$
\tilde{C} = \bar{C} (I - \bar{A}^L z^L)
$$

> 단위원(node z=1)에서 평가할 것이기 때문에 표기가 생략될 수 있는 것입니다.

정리:
- 시간 도메인 → 주파수 도메인 변환: z-transform
- z를 단위원 위에서 평가: DFT와 동일한 연산
- IFFT로 역변환: 시간 도메인 커널 복원
- 복잡도 감소: ${O}(L^2) \Rightarrow {O}(L \log L)$
- 행렬 거듭제곱 회피: 연산량과 메모리 모두 절약

이 과정은 $A$ 행렬의 거듭제곱 없이도 커널 전체를 재구성할 수 있게 해주며, S4의 근본적인 병목을 제거하는 열쇠가 됩니다.

#### Step 2. Diagonal Case와 Cauchy Kernel

$$
\hat{K}_L(z) = \tilde{C} (I - \bar{A} z)^{-1} \bar{B}
$$

일단 여기까지 왔습니다. 여기서의 가장 큰 병목은 역행렬 계산입니다. 매번 일반적인 행렬에 대해 역행렬을 계산하는 것은 연산량이 매우 큽니다. 이를 해결하기 위해 $\bar{A}$를 diagonal matrix라고 가정해보겠습니다. 

> Q. 왜 Diagonal로 가정하나요?
> A. Diagonal matrix가 가장 단순한 형태의 행렬이기 때문입니다. Diagonal의 역행렬은 대각 원소의 역수이기 때문에 역행렬 계산이 매우 단순해집니다. 따라서 복잡한 행렬 연산 없이 벡터 연산으로 커널 계산이 단순화됩니다. 

위의 커널을 다음과 같이 재표현할 수 있습니다:

$$
\tilde{C}(I - \bar{A}z)^{-1} \bar{B} = \frac{2\Delta}{1 + z} \tilde{C} \left[ 2\frac{1 - z}{1 + z} - \Delta \bar{A} \right]^{-1} \bar{B}
$$

> 디테일한 내용은 논문 부록 C.3에 있습니다.

그리고 커널은 다음과 Cauchy 커널 형태로 표현할 수 있습니다:

$$
\hat{K}_{\Lambda}(z) = c(z) \sum_i \frac{\tilde{C}_i B_i}{g(z) - \Lambda_i} = c(z) \cdot k_{z, \Lambda}(\tilde{C}, \bar{B})
$$

$A$를 Diagonal로 가정했기 때문에 $(I - Az)^{-1}$의 계산이 각 성분에 대해 $\displaystyle \frac{1}{g(z) - \Lambda_i}$로 분해될 수 있고, 이는 곧 [Cauchy kernel](https://encyclopediaofmath.org/wiki/Cauchy_kernel)의 선형 결합 표현입니다.

#### Step 3. DPLR Case: Diagonal 조건 완화

앞서 **diagonal 행렬 $\bar{A}$**를 사용하여 역행을 간단히 만들 수 있었지만,
현실적으로 더 일반적인 구조로 조건을 완화해보겠습니다:

$$
\bar{A} = \Lambda - PQ^*
$$

$$ \Lambda \in \mathbb{C}^{N \times N} : \text{diagonal matrix} $$

$$P, Q \in \mathbb{C}^{N \times r} :\text{r-rank term} $$ 

Low-rank는 일반성을 잃지않고 r=1로 가정할 수 있습니다.

$$P, Q \in \mathbb{C}^{N \times 1} :\text{1-rank term} $$

DPLR 구조를 활용하면 역행렬 계산을 [Woodbury identity](https://en.wikipedia.org/wiki/Woodbury_matrix_identity)로 다음과 같이 변형할 수 있습니다:

$$
(\Lambda - P Q^*)^{-1} = \Lambda^{-1} + \Lambda^{-1} P \left(I - Q^* \Lambda^{-1} P \right)^{-1} Q^* \Lambda^{-1}
$$

즉, full matrix inverse 없이, 역행렬 연산을 diagonal 역수 + rank-1 연산만으로 처리가 가능하다.

> 논문 부록 C.3 Proposition 4 참고

기존에는 다음을 계산해야 했습니다:

$$
\hat{K}(z) = \tilde{C} (I - \bar{A} z)^{-1} \bar{B}
$$

DPLR에서는 이 연산을 아래와 같이 재구성할 수 있습니다:

$$
\hat{K}_{\Lambda}(z) = c(z) \sum_i \frac{\tilde{C}_i \bar{B}_i}{g(z) - \Lambda_i} = c(z) \cdot k_{z, \Lambda}(\tilde{C}, \bar{B})
$$

DPLR의 두번째 이점은 $A$의 역행렬을 계산할 필요없이 이산화된 SSM을 계산할 수 있다는 것입니다. 

> Q. 아니 Woodbury 항등식으로 역행렬 계산 없이 효율적인 커널 계산이 가능하게 했다면서 역행렬 계산이 필요없는 SSM 이산화는 또 뭔가요?
> A. SSM을 Bilinear transform할 때도 $A$의 역행렬이 등장합니다. 그 부분에서 역행렬을 계산하지 않아도 된다는 이점이 있다는 얘기입니다. 

$A$에 $\Lambda-PQ^*$를 대입하여 식을 정리하면 최종 형태는 다음과 같습니다:

$$
x_k = \bar{A} x_{k-1} + \bar{B}u_k 
\\
= A_1 A_0 x_{k-1} + 2 A_1 B u_k
$$

$$
y_k = C x_k
$$

> 디테일한 내용은 논문 부록 C.2에 있습니다.

## HiPPO를 DPLR로 전환
HiPPO는 Normal Plus Low-Rank인 NPLR입니다. 그렇다고 DPLR에서의 논의를 적용할 수 없는 것은 아닙니다. [Normal](https://en.wikipedia.org/wiki/Normal_matrix)은 [unitary](https://en.wikipedia.org/wiki/Unitary_matrix)로 대각화가 가능한 행렬입니다. 따라서 SSM 관점에서 NPLR은 DPLR과 본질적으로 동일합니다. 

NPLR SSM:
$$
A = V \Lambda V^* - P Q^\top = V \left( \Lambda - V^* P (V^* Q)^* \right) V^*
$$

$$V \in \mathbb{C}^{N \times N}: \text{unitary matrix}$$

$$\Lambda \in \mathbb{C}^{N \times N}: \text{diagonal matrix}$$

$$P, Q \in \mathbb{R}^{N \times r}: \text{low-rank term}$$

이번에도 일반성을 잃지 않고 r=1로 둘 수 있습니다. 

$$P, Q \in \mathbb{R}^{N \times 1}: \text{1-rank term}$$

이 NPLR을 DPLR로 변환하는 과정은 다음과 같습니다: 
1. HiPPO 행렬에 rank-1 term을 더한 normal 행렬로 만든다.
2. 이 normal 행렬을 unitary diagonalization하여 $\Lambda, V$를 추출한다.
3. Low-rank항 $P$도 같은 basis로 회전해 DPLR 구조로 만든다.

> 디테일한 내용은 논문 부록 B와 C에 있습니다. 
> 자세한 내용은 심화편에서 다루겠습니다.

#### Advantage
1. 복소수 고윳값 $\Lambda$의 실수/허수 부분을 따로 제어할 수 있어 학습 안정성에 유리합니다.
2. $A$ 행렬을 Hermitian(에르미트) 행렬로 바꾸면 더 빠르고 정확한 diagonalization이 가능합니다. (하지만 논문 시점에서는 JAX가 에르미트 행렬에 대한 대각화를 지원하지는 않는다고 함)

추가적인 구조 제약으로 $P=Q$를 설정하면 SSM의 안정성이 더욱 향상되고 실전 학습 성능이 개선되는 것을 후속 연구에서 보였습니다. [링크](https://en.wikipedia.org/wiki/Unitary_matrix)

> - 이제 우리는 S4로 MNIST, QuickDraw, Spoken Digits를 학습할 수 있습니다.
> - 실습 코드는 [Part 3](https://srush.github.io/annotated-s4/)를 참고하세요.

## Conclusion
수학적인 세부 전개보다는 전체 구조와 흐름에 초점을 맞춰,
S4 커널 계산이 어떤 아이디어와 구조 위에 설계되었는지를 이해하는 데에 집중했습니다.

Diagonal -> DPLR -> NPLR로 이어지는 흐름과, Cauchy 커널 및 FFT 기반 계산 방식은 S4가 어떻게 효율성과 표현력을 동시에 확보했는지를 보여주는 핵심적인 아이디어입니다.

수식 전개같은 디테일한 내용은 심화편인 **"S4 Appendix로 보는 Algorithm"**에서 상세히 다뤄보도록 하겠습니다.

이번 글이 S4의 논리 흐름을 대략적으로 이해하는 데 도움이 되었기를 바랍니다.