**Bayesian Change Point Model을 활용한 사회적 이슈 키워드 언급량 변곡점 탐색**

**1. 모델 설계 및 비교 (Statistical Modeling)**

• **베이지안 추론:** 국회 본회의 회의록 데이터를 바탕으로 [참사, 국가책임, 추모, 재난] 등 특정 키워드의 언급 빈도가 급격히 변하는 시점(Change Point)을 탐색.

• **사전분포 최적화:** Lambda에 대해 지수분포와 감마분포를 각각 적용한 모델을 설계하여 사전 지식 반영의 유연성 비교 분석 .

• **다중 변곡점 모델링:** 변곡점의 개수(1~3개)에 따른 6가지 유형의 모델을 구축하여 사회적 사건과 키워드 급증 시점 간의 연관성 검증 .



**2. 모델 검증 및 진단 (MCMC Diagnostics)**

• **수렴 진단:** **MCMC(Markov Chain Monte Carlo)** 샘플링을 수행하고, R-hat 지표와 ESS(Effective Sample Size)를 통해 모델의 수렴 여부 및 샘플 효율성을 정밀 진단 .

• **불확실성 정량화:** 사후신뢰구간(**HDI**)과 **MCSE**(Monte Carlo Standard Error)를 산출하여 추정된 변곡점의 통계적 신뢰도 확보 .


역할: 개인 프로젝트
