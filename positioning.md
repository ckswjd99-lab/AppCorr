````markdown
# Progressive VLA Serving System

## 1. 핵심 아이디어

현재 AppCorr/Oudjat은 DINOv3 기반 offloaded ViT inference를 빠르게 만들기 위한 시스템이다. 모바일은 이미지를 Laplacian pyramid 기반의 low-resolution base와 residual patches로 나누어 전송하고, 서버는 base가 도착하자마자 approximate forward를 실행한 뒤 residual이 도착하면 중요한 token만 보정한다.

이 아이디어를 VLA serving으로 확장한다.

기존 VLA serving은 보통 다음과 같다.

```text
high-res observation upload
  -> visual encoder
  -> LLM/VLM prefill
  -> action generation
````

문제는 서버가 high-resolution visual observation을 모두 받기 전까지 prefill을 시작하지 못한다는 점이다. 로봇/피지컬 AI에서는 이 지연이 곧 stale observation 문제로 이어진다.

제안하는 시스템은 다음과 같다.

```text
low-res visual base upload
  -> early visual encoder
  -> early LLM/VLM prefill

residual upload
  -> visual / prefill / conditioning repair
  -> refined action
```

즉, 목표는 **visual upload latency와 VLA prefill을 overlap**하는 것이다.

---

## 2. 기존 Oudjat에서 가져올 것

AppCorr/Oudjat에서 이미 존재하는 요소:

* mobile -> server runtime
* TCP 기반 patch streaming
* transmission policy
* scheduling policy
* Task / Instruction 기반 서버 실행 구조
* DINOv3 executor
* CUDA event 기반 timing
* Laplacian pyramid 기반 progressive input
* approximate-then-correct inference 구조

VLA 버전에서는 DINOv3 executor를 VLA executor로 확장한다.

```text
Current:
  DINOv3 executor

New:
  OpenVLA executor
  pi0 executor
  pi0-FAST executor
```

핵심적으로 재사용할 abstraction은 다음이다.

```text
ITransmissionPolicy:
  visual base와 residual을 어떤 순서로 보낼지 결정

ISchedulingPolicy:
  base prefill, residual repair, recompute, decode 시점을 결정

Task / Instruction:
  서버에서 실행할 model operation을 명시
```

---

## 3. 타겟 모델

### OpenVLA

가장 중요한 primary target.

```text
image
  -> vision encoder
  -> projector
  -> LLM backbone prefill
  -> autoregressive action tokens
```

OpenVLA에서는 가장 깔끔하게 다음을 측정할 수 있다.

* time-to-start-prefill
* visual upload와 prefill overlap 효과
* low-res prefill quality
* residual repair 후 action quality 회복
* full recompute 대비 repair cost
* action agreement with full-resolution oracle

### RT-2

직접 실험보다는 signature reference로 사용한다.

RT-2는 VLM-as-policy 계열의 대표 모델이므로, visual context가 도착하기 전까지 VLM prefill/action-token generation이 시작되지 못한다는 문제를 설명하는 데 쓴다.

### pi0

pi0는 OpenVLA처럼 action token을 단순 AR decode하는 구조는 아니고, VLM backbone 뒤에 flow-matching action expert가 붙는다.

```text
image
  -> vision encoder
  -> VLM/LLM conditioning
  -> action expert
  -> flow matching
  -> continuous action chunk
```

따라서 pi0에서는 "KV repair"보다는 "visual/VLM conditioning을 더 일찍 시작하고 residual로 conditioning을 보정한다"는 식으로 해석한다.

### pi0-FAST

modern AR VLA stress test로 사용한다.

action tokenization이 더 최적화된 모델에서도 visual arrival bottleneck이 남는지 보여준다.

---

## 4. 시스템 설계

### 4.1 Progressive visual input

로봇은 high-resolution observation을 한 번에 보내지 않고 다음처럼 보낸다.

```text
x_full = x_base + r1 + r2 + ... + rk
```

* `x_base`: low-resolution visual base
* `r_i`: high-frequency residual
* optional: ROI-first residual, gripper 주변 residual, target object 주변 residual

### 4.2 Early VLA prefill

서버는 `x_base`가 도착하면 즉시 VLA prefill 또는 conditioning을 시작한다.

```text
C_base = Prefill(x_base, instruction, robot_state)
```

모델별 의미:

* OpenVLA: visual tokens / LLM prefill / draft action context
* pi0: VLM conditioning for action expert
* pi0-FAST: AR action-token context

### 4.3 Residual repair

Residual이 도착하면 기존 state를 보정한다.

```text
C_refined = Repair(C_base, r1, ..., ri)
```

가능한 구현 단계:

1. visual feature repair
2. visual token repair
3. selected layer recomputation
4. prefill/KV-like state repair
5. action hypothesis refinement

초기 구현은 Oudjat과 가장 가까운 **visual encoder feature repair**부터 시작하는 것이 현실적이다.

### 4.4 Scheduler

Scheduler는 매 시점 다음을 결정한다.

* base만으로 action hypothesis를 낼 것인가?
* residual을 더 기다릴 것인가?
* repair할 것인가 full recompute할 것인가?
* action deadline 전에 early return할 것인가?
* confidence가 충분하면 early exit할 것인가?

---

## 5. Evaluation

### 5.1 Models

Main quantitative model:

* OpenVLA

Broader coverage:

* RT-2: architectural reference
* pi0: flow-based VLA case
* pi0-FAST: modern AR VLA case

가장 예쁜 latency/repair 숫자는 OpenVLA에서 뽑고, pi0/pi0-FAST는 "visual input availability bottleneck은 구조가 달라도 존재한다"는 broader relevance를 보여주는 용도로 쓴다.

---

## 5.2 Tasks

### LIBERO

Main closed-loop benchmark.

사용할 suite:

* LIBERO-Spatial
* LIBERO-Object
* LIBERO-Goal
* LIBERO-10 또는 LIBERO-Long

Task category:

1. Coarse-layout tasks
   Low-res base만으로도 scene layout과 target 위치를 대략 알 수 있는 task.

2. Fine-detail tasks
   object identity, color, texture, small geometry, handle 등이 중요해서 residual repair가 필요한 task.

3. Long-horizon / stale-sensitive tasks
   서버 응답이 늦으면 action이 stale해지는 multi-step task.

### RoboCasa

Realistic visual complexity를 보강하기 위한 benchmark.

추천 task:

* cluttered object retrieval
* drawer/cabinet opening
* counter cleanup
* appliance interaction
* long-horizon kitchen tasks

RoboCasa는 low-res base와 residual의 역할 차이를 보여주기 좋다.

```text
low-res:
  scene layout, large object location

residual:
  handle, label, boundary, clutter detail
```

### BridgeData V2 / Open X-Embodiment replay

Trace-driven systems evaluation용.

각 recorded frame에 대해:

```text
full-res oracle action = VLA(full-res observation)
progressive action = VLA(low-res base + residual repair)
```

측정:

* action agreement
* action error
* time-to-oracle-equivalent-action
* bandwidth-to-useful-action
* stale observation gap

### Optional real robot demo

작게만 해도 좋다.

추천 task:

* colored cup among distractors
* small-label object picking
* object near obstacle
* drawer/cabinet handle manipulation

---

## 5.3 Baselines

1. Full-resolution wait
   Full high-res observation이 다 도착한 뒤 VLA 실행.

2. Low-resolution only
   Low-res base만으로 VLA 실행.

3. Progressive upload + full recompute
   Base로 draft를 만들고, full input 도착 후 처음부터 다시 실행.

4. Progressive upload + naive append
   Residual 정보를 단순 append.

5. Proposed progressive repair
   Base prefill을 residual로 보정.

6. Zero-delay full-resolution oracle
   Full-res input이 서버에 즉시 있다고 가정한 upper bound.

---

## 5.4 Metrics

Systems metrics:

* time-to-start-prefill
* time-to-first-action-hypothesis
* time-to-correct-action
* upload latency hidden
* GPU busy time during upload
* repair overhead
* prefill reuse ratio
* bandwidth-to-useful-action
* p50/p95/p99 latency

VLA / robotics metrics:

* task success rate
* action agreement with full-res oracle
* action L1/L2 error
* trajectory deviation
* correction rate
* harmful early action rate
* confidence calibration

---

## 6. Expected Results

Expected claims:

1. Progressive VLA serving reduces time-to-start-prefill by starting from low-res visual base.

2. It overlaps visual upload with VLA prefill, unlike conventional full-resolution wait.

3. Low-res only is fast but inaccurate on fine-detail tasks.

4. Residual repair recovers much of the full-resolution action quality.

5. Repair is cheaper and faster than full recomputation.

6. The benefit grows under low bandwidth, high RTT, or high jitter.

7. OpenVLA gives the cleanest result because its visual encoder, LLM prefill, and action decode stages are explicit.

8. pi0 and pi0-FAST broaden the story by showing that visual-context arrival delay matters beyond one OpenVLA implementation.

---

## 7. Paper Positioning

Do not frame this primarily as robot-server collaboration.

Frame it as:

```text
Progressive VLA Serving System
```

Core statement:

```text
Existing VLA serving assumes high-fidelity visual observations are available atomically before inference begins. This delays server-side visual encoding and multimodal prefill under realistic visual upload latency. We propose Progressive VLA Serving, which starts VLA prefill from a low-resolution visual base and repairs the prefill or conditioning state as residual visual evidence arrives.
```

Relation to Oudjat:

```text
Oudjat:
  progressive refinement for offloaded DINOv3 ViT inference

This work:
  progressive refinement for offloaded VLA serving
```

Main contribution shift:

```text
From:
  accelerating visual feature extraction

To:
  accelerating time-to-start-prefill and time-to-useful-action
  in server-side VLA serving
```

---

## 8. EuroSys Fit

This is a good EuroSys-style systems paper because it combines:

* AI/ML systems: VLA inference serving
* networked systems: visual upload latency and residual streaming
* distributed systems: robot/mobile client and GPU server
* cyber-physical systems: stale physical state and action deadline
* runtime systems: repair vs recompute scheduling

The key message:

```text
This is not image compression for robots.
This is a serving runtime that changes when server-side VLA computation can begin.
```

---

## 9. Preferred Title

```text
Progressive VLA Serving with Repairable Visual Prefill
```

One-sentence summary:

```text
Progressive VLA Serving extends Oudjat's progressive DINOv3 offloading to VLA inference: the robot sends a low-resolution visual base first, the server starts VLA prefill immediately, and residual visual evidence repairs the prefill or conditioning state so the system returns useful actions earlier than full-resolution offloading while approaching full-resolution action quality.
```

```
```
