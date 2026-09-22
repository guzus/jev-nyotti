# 저트래픽 공개 사이트용 파인튜닝·서빙 비교

조사 기준: 2026-09-22. 금액은 USD, 별도 표기한 경우만 CNY. 공식 문서·요금표 기준이며 실제 계정의 배포, 성능, 과금은 검증하지 않았다. 원본 매매 데이터도 아직 확보하지 않았다. 아래는 구매나 웹사이트 구현 완료 보고가 아닌 조사와 제안이다.

## 추천

**Qwen3.8-27B를 유지하면 Together에서 LoRA SFT 후, DeepInfra 또는 Novita에서 잠깐씩 구동하는 구성을 먼저 검토한다.** DeepInfra는 가격, Novita는 관리형 LoRA 기능이 매력적이다. 정확한 모델 지원이나 내보낸 체크포인트 수용에 문제가 있으면 자체 컨테이너를 제어할 수 있는 Modal을 검토한다. 범용 Hugging Face 업로드 지원만으로 정확한 Qwen 버전의 배포 성공을 단정할 수는 없다.

**모델 변경이 가능하면 W&B Training이 별도의 유력 후보다.** 지원되는 작은 Qwen을 학습하고 토큰 단위로 호출하면, 직접 GPU를 관리하지 않아도 된다. 다만 Qwen3.8-27B 기본 모델 API 제공과 그 모델의 커스텀 학습 지원은 다르다.

공개 사이트의 첫 형태는 **과거 차트에서 사용자가 롱·숏·관망을 고르고 AI와 비교하는 체험**으로 가정했다. 실제 자동매매나 매 방문자에게 실시간 장문 상담을 제공하는 서비스라면 예산과 평가 기준이 달라진다.

## 비교할 때 구분할 것

- **공유 API의 기본 모델 가격**은 내가 파인튜닝한 모델의 서빙 가격이 아니다.
- **토큰 과금**과 **GPU 가동시간 과금**은 다르다. 후자는 모델 로딩·요청 사이 대기·자동 종료까지의 시간도 비용에 들어갈 수 있다.
- **Scale-to-zero**는 요청이 없을 때 GPU가 0개가 된다는 의미다. 다음 요청이 즉시 응답한다는 의미가 아니다.
- 27B bf16 가중치는 단순 계산으로 약 54GB다. KV 캐시와 런타임을 더하면 80GB GPU가 초기 검토 대상이다. 이는 정확한 모델의 메모리 실측이나 장문·다중 요청 지원 보장이 아니다.

## GPU 시간으로 과금하는 후보

한 GPU의 공개 단가를 비교했다. 서로 다른 GPU·소프트웨어의 처리량은 다르므로 시간당 가격만으로 요청당 비용 순위를 정할 수 없다. 10시간·50시간은 **월 전체 청구 대상 GPU 가동시간**이라는 가정이다. 저장소·CPU·RAM·트래픽·세금 등의 추가 비용과 할인은 제외했다.

| 업체 / 대표 GPU | 시간당 | 월 10시간 | 월 50시간 | 커스텀 모델·운영 특성 |
| --- | ---: | ---: | ---: | --- |
| [DeepInfra / A100 80GB](https://deepinfra.com/pricing) | $0.89 | $8.90 | $44.50 | private HF 가중치, min_instances=0. 저렴한 관리형 후보 |
| [Chutes / RTX Pro 6000 96GB](https://chutes.ai/pricing) | $1.80 | $18.00 | $90.00 | private custom 배포, 배포당 $5.40 별도 |
| [Novita / H100 80GB](https://novita.ai/dedicated-endpoint) | $1.99 | $19.90 | $99.50 | private HF·LoRA, 최소 replica 0, 활성 replica 초 단위 과금 |
| [Cerebrium / A100 80GB](https://www.cerebrium.ai/pricing) | $2.0988 | $20.99 | $104.94 | 커스텀 앱·가중치, CPU/RAM 별도 |
| [Modal / A100 80GB](https://modal.com/pricing) | $2.4984 | $24.98 | $124.92 | 런타임·큐·캐시를 직접 제어, CPU/RAM 별도 |
| [Hugging Face Endpoints / AWS A100 80GB](https://huggingface.co/docs/inference-endpoints/pricing) | $2.50 | $25.00 | $125.00 | private repo 및 custom container, 분 단위 청구 |
| [Runpod Flex / A100 80GB](https://www.runpod.io/pricing) | $2.72 | $27.20 | $136.00 | custom worker/container, 큐 기반 호출 가능 |
| [Baseten / A100 80GB](https://www.baseten.co/pricing/) | 약 $4.00 | 약 $40.00 | 약 $200.00 | 관리형 배포, 기본 idle 대기시간 확인 필요 |
| [Replicate / A100 80GB](https://replicate.com/pricing) | $5.04 | $50.40 | $252.00 | private Cog 모델은 실행 외 시작·idle도 과금 |
| [Together / H100](https://www.together.ai/pricing) | $5.49 | $54.90 | $274.50 | 학습과 연결 편리, DMI 자동 wake 제약 |
| [Fireworks / H100 또는 H200](https://fireworks.ai/pricing) | $8.00 | $80.00 | $400.00 | dedicated LoRA도 scale-to-zero 가능 |

추가 후보:

| 업체 | 확인한 조건 | 이번 용도 판단 |
| --- | --- | --- |
| [Beam](https://www.beam.cloud/pricing) | H100 serverless $0.000972/s, 약 $3.50/h 표시에는 committed-spend 조건이 붙음 | 과거 블로그의 싼 GPU 단가나 GPU rental 요금을 그대로 쓰면 안 됨. 무약정 견적 확인 후 비교 |
| [fal](https://fal.ai/pricing) | H100 정가 $4.50/h, 'as low as $1.89'는 별도 조건 확인 필요 | [커스텀 serverless](https://fal.ai/serverless)는 가능하지만 미디어 중심 제품이며 이번 텍스트 모델의 우선 후보는 아님 |

Together의 H100 $3.99/h 프로모션은 2026-09-30까지 표시되어 장기 예산에는 정가를 사용했다. 어느 GPU 업체도 여기서 정확한 Qwen3.8-27B 튜닝 체크포인트를 실제로 배포한 것은 아니다.

### 호출이 적어도 비용·대기시간이 생기는 부분

- **DeepInfra:** [Custom LLM 문서](https://docs.deepinfra.com/private-models/custom-llms)는 private HF 모델과 min_instances=0을 지원한다. 현재 quantization 미지원, scale-up 때 GPU 확보는 보장하지 않는다. 먼저 80GB 단일 GPU에서 짧은 컨텍스트로 호환성을 확인할 대상이다.
- **Novita:** [공식 설명](https://novita.ai/dedicated-endpoint)은 0→N autoscaling, private HF, LoRA hot-swap을 제공한다. GPU가 0일 때 비용이 없지만, 켜진 채 기다리는 replica까지 무료라고 해석해서는 안 된다. 실제 cold-start 지연은 측정 대상이다.
- **Modal:** [cold-start 문서](https://modal.com/docs/guide/cold-start)의 종료 대기 설정이 비용을 좌우한다. min_containers=0으로 시작하고 캐시·weight loading·큐를 설계할 수 있다. 편리한 앱 개발 플랫폼이지만 학습 파일 업로드만으로 완료되는 SaaS와는 다르다.
- **Runpod:** [serverless billing](https://docs.runpod.io/serverless/pricing)은 시작·모델 초기화·처리·idle 구간을 포함한다. scale-to-zero와 짧은 idle 설정을 사용해도, 자주 깨우면 로딩 비용이 반복된다.
- **HF Endpoints:** [autoscaling 문서](https://huggingface.co/docs/inference-endpoints/guides/autoscaling)는 기본 1시간 후 scale-to-zero, 초기화 중 503, 대기용 X-Scale-Up-Timeout 헤더를 설명한다. 모델에 따라 시작에 몇 분이 걸릴 수 있다.
- **Cerebrium / Baseten:** 각각 [scaling 설정](https://docs.cerebrium.ai/cerebrium/scaling/batching-concurrency)과 [autoscaling 설정](https://docs.baseten.co/deployment/autoscaling/overview)에서 최소 replica와 종료 대기를 조정한다. Baseten 기본 종료 대기는 900초다.
- **Together:** [현재 DMI 설명](https://www.together.ai/blog/autoscaling-endpoints-for-llm-inference)에서는 min=max=0이 명시적 정지이고, 다음 요청으로 자동 재시작되지 않는다. 저트래픽 공개 사이트의 자동 절전 기능과 동일하게 보면 안 된다.
- **Fireworks:** [dedicated autoscaling](https://docs.fireworks.ai/deployments/autoscaling)은 0까지 축소 가능하다. 기본 idle 1시간, 최소 5분. 초기화 중 요청을 큐에 넣지 않고 503을 돌려주므로 재시도가 필요하다. 기존 24시간 상시 GPU 계산은 필수 유지비가 아니었다.

## 토큰 과금·학습 통합·기타 서비스

| 선택지 | 학습한 모델 서빙 여부 | 가격·제약과 판단 |
| --- | --- | --- |
| **W&B Training / OpenPipe** | [학습 artifact로 API 호출](https://docs.wandb.ai/serverless-training/use-trained-models) 지원 | [학습 가격표](https://site.wandb.ai/pricing/training/)의 Qwen3-14B 입력 $0.05/M, 출력 $0.22/M. 공개 preview 중 학습 무료, 저장료 별도. 작은 모델 허용 시 강력한 후보 |
| **Thinking Machines Tinker** | 학습 checkpoint sampling 가능 | Qwen3.8-27B 입력 $1.86/M, 출력 $5.595/M. [OpenAI 호환 API](https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/openai/)는 beta·내부 테스트 트래픽 용도. 공개 웹사이트 주력으로 추천하지 않음 |
| **DeepInfra LoRA API** | [지원되는 base model에 adapter 업로드](https://docs.deepinfra.com/private-models/lora) 경로 존재 | 정확한 Qwen3.8-27B 지원과 커스텀 adapter의 과금은 인증된 모델 목록/계정에서 추가 확인. 위 custom GPU 경로와 구분 |
| **Featherless** | [Scale 고객의 private HF 모델](https://featherless.ai/docs/models-model-compatibility) | full weights 필요, LoRA 자체 업로드와 다름. 해당 버전과 Scale 견적 미확인. 일반 구독료를 private hosting 가격으로 쓰지 않음 |
| **Inference.net** | [private HF·자체 학습 모델 배포](https://docs.inference.net/platform/deploy/overview) | 커스텀 GPU 경로는 있으나 이번 모델의 공개 견적·자동 절전 조건 미확인. 추가 견적 후보 |
| **Alibaba Model Studio / PAI** | [공식 학습 요금표](https://help.aliyun.com/en/model-studio/model-training-and-deployment-billing)에 Qwen3.8-27B | 학습 ¥0.05/1K tokens 표시. 국제 계정·지역별 모델 지원 및 튜닝 모델 서빙 요금 미확인; 중국 지역 가격을 글로벌 서비스 견적으로 쓰지 않음 |
| **Predibase / Rubrik** | 기존 커스텀 모델 플랫폼 | [Rubrik 인수 발표](https://www.rubrik.com/company/newsroom/press-releases/25/rubrik-to-acquire-predibase-to-accelerate-agentic-ai-adoption). 현행 [가격 페이지](https://predibase.com/pricing)에서 소규모 셀프서비스 조건을 확정하지 못함. 영업 문의형 후보 |
| **Cloudflare Workers AI LoRA** | [선택된 base model만 지원](https://developers.cloudflare.com/workers-ai/features/fine-tunes/loras/) | 임의 Qwen3.8-27B adapter 서빙 지원을 확인하지 못했으므로 이번 모델의 후보로 확정하지 않음 |
| **AWS SageMaker Async** | [커스텀 모델·GPU scale-to-zero·비동기 큐](https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference.html) | 가능하지만 소규모 체험 사이트에 비해 설정이 많음. 일반 Serverless Inference와 GPU 지원을 혼동하지 말 것 |
| **Google Vertex AI** | [GPU scale-to-zero preview](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/predictions/autoscaling) | 모델 배포 경로는 있으나 idle·cold-start·쿼터 관리가 필요. 기존 GCP 기반이 없다면 우선순위 낮음 |

W&B의 [지원 모델 문서](https://docs.wandb.ai/serverless-training/available-models)와 가격표의 모델 목록은 완전히 같지 않다. 가격표에는 Qwen3.6-27B도 보이지만, 이것을 Qwen3.8-27B 학습 가능 근거로 사용할 수 없다. 현재 기본 모델 API 제공 여부와 튜닝 가능 여부를 구분해야 한다.

W&B Qwen3-14B 예시: 요청당 입력 2,000·출력 200 tokens라면 **1만 요청에 $1.44**다. 플랫폼 구독·저장료·재시도는 제외한 추론 계산이며, 학습 artifact에도 해당 가격이 적용되는지 계정에서 확인해야 한다. 27B와 동급 품질·지연이라는 의미도 아니다.

Tinker의 동일 토큰 가정은 요청당 $0.004839, 1만 요청 $48.39다. [모델별 요금](https://tinker-docs.thinkingmachines.ai/tinker/models/)을 기준으로 했으며, thinking을 포함한 전체 생성 tokens가 증가하면 비용도 늘어난다. 학습·연구 API와 공개 앱의 운영 적합성은 별개다.

## 학습 SaaS 선택

전체 140만 행을 바로 학습하지 않고, 재구성한 의사결정 표본 1만 개로 먼저 실험한다. 한 주문의 분할 체결 여러 행은 독립적인 여러 판단이 아닐 수 있다.

| 공급자 | 30M 학습 tokens 예시 | 의미 |
| --- | ---: | --- |
| [Together](https://www.together.ai/pricing) | $31.50 | Qwen3.8-27B LoRA SFT $1.05/M, 최소 $4. 내보내기·서빙 호환성 별도 확인 |
| [Fireworks](https://fireworks.ai/pricing) | $90.00 | 16.1–80B LoRA tier $3/M. 정확한 모델 지원 확인 필요 |
| [Tinker](https://tinker-docs.thinkingmachines.ai/tinker/models/) | $123.09 | Qwen3.8-27B 64K, $4.103/M. 연구 루프 제어에 적합 |
| [W&B Training](https://site.wandb.ai/pricing/training/) | preview 중 학습 무료 | 지원 모델로 변경하는 경로. 추론·저장 비용까지 무료라는 뜻은 아님 |

30M = 표본 10,000 × 평균 1,000 tokens × 3 epochs. 검증·반복 실험·포맷 확장은 별도다. Unsloth는 직접 학습을 실행하는 도구이므로 GPU를 빌리는 운영 작업을 없애주는 관리형 SaaS와 다르다.

## 방문자 수와 추론 비용을 분리하는 사이트 제안

**제안 경험:** “나는 롱, AI는 관망. 결과는?” 과거 차트의 미래 구간을 가리고, 사용자가 판단한 뒤 AI 선택·실제 이후 가격 경로·원본에 있는 실제 체결을 구분해서 공개한다. 결과 카드를 공유하고 친구가 같은 문제를 풀게 한다. 이는 검증 전의 제품 가설이다.

1. 학습에 사용하지 않은 시점의 사례 수백 개를 먼저 선정한다. 모델도 판단 시점까지의 정보만 받는다.
2. 각 사례의 모델 결과를 한 번 생성해 저장한다. model revision, prompt version, 데이터 기준 시각을 함께 기록한다.
3. 방문자에게 동일한 저장 결과를 제공한다. 사용자 선택이 바뀌어도 AI 답을 다시 생성하지 않는다. 유리한 결과만 골라 보여주는 방식은 피한다.
4. 공유 링크와 카드도 같은 저장 결과를 사용한다. 방문 10만 회가 추론 10만 회가 되지 않는다.
5. 자유 질문은 선택 기능으로 두고 별도 큐·횟수 제한을 둔다. 시작은 replica 최대 1개와 짧은 컨텍스트, 총 생성 tokens 제한으로 잡는다.

현재 시장 코멘트를 추가하더라도 종목·시간대별 공통 결과를 생성해 캐시할 수 있다. 예를 들어 BTC 한 종목을 시간당 한 번 갱신하면 30일에 720회다. 화면에는 반드시 실제 데이터 시각과 생성 시각을 표시한다. 저장된 결과를 실시간 추론처럼 표시하지 않는다.

GPU 가동비는 `청구 가동시간 × GPU 개수 × 시간당 단가`다. 예를 들어 각 방문마다 2분 초기화가 필요한지, 50명이 한 번에 같은 캐시를 읽는지에 따라 같은 방문 수의 비용이 크게 달라진다. 실제 부하 테스트 전에는 방문자당 가격을 확정하지 않는다.

운영 초기 **학습 포함 $100–250 준비금**, 이후 **추론 월 $30–150 관리 목표**를 제안한다. 이는 요금 약속이 아니라 spending cap 설계용 가정이다. 저장소·프런트엔드·데이터 구매·세금·운영 작업은 별도이며, 상한에 도달하면 새 추론을 멈추고 기존 예제를 표시하도록 구현해야 한다. 알림만 설정했다고 지출이 차단되는 것은 아니다.

## “워뇨띠 매매를 학습했다”라는 표현의 근거

첨부 이미지는 원본 거래 파일 자체가 아니며, 공개 여부·진위·행 수·성과를 아직 독립적으로 확인하지 못했다. 현 단계에서 이미 학습했다고 홍보할 근거는 없다.

원본 출처와 사용 가능 범위를 확인하고 실제 학습을 마친 뒤에는 **“공개된 워뇨띠 매매기록을 바탕으로 학습한 실험 모델”** 같은 문구를 검토할 수 있다. 출처·대상 기간·실제 사용한 표본 수·미학습 평가 기간·모델 버전을 별도 설명에 담는다. 본인 참여·공식 서비스·판단 재현·수익 재현은 각각 별도 근거가 필요한 주장이다.

매매기록은 체결 결과이지 판단 당시의 모든 정보나 생각이 아니다. 당시 시장·포지션·미체결 주문·거래하지 않은 구간을 얼마나 재구성했는지가 중요하다. 모델의 해설은 생성된 추론이며 당사자가 남긴 실제 해설과 구분한다. 데이터·평가 준비 순서는 [실험 계획](experiment.md)을 따른다.

## 다음 결정에 필요한 검증

1. 데이터 소량 확보 및 출처·스키마 확인. 사진 속 전체 파일을 확보했다는 가정은 금지한다.
2. Qwen3.8-27B 유지 여부 결정. 8–14B 비교 실험도 함께 두되 크기만으로 거래 성능을 판단하지 않는다.
3. 공급자 계정에서 정확한 모델 ID, 학습 export 형식, 서빙 허용 runtime, 최종 견적 확인.
4. 선택한 업체 한 곳에서 작은 호환성 실험: base revision·adapter·chat template 일치, 한국어 출력, JSON action, 메모리, cold/warm 지연, 실제 청구 시간 기록.
5. 시간 순서로 분리한 평가에서 untuned 모델·간단한 수치 모델·tuned 모델 비교. 수수료·슬리피지 포함 평가는 체결 모방 정확도와 별도 보고.
6. 체험 사이트 공개 전 전체 비용 상한과 캐시 동작 검증. 이번 조사에서는 유료 작업·학습·배포를 실행하지 않았다.
