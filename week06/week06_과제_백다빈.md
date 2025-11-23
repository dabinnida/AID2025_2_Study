# Week 06 딥러닝 훈련 기술 정리

## 1. 배치 정규화 (Batch Normalization)
**역할**  
배치 단위로 층의 출력을 정규화하여 평균과 분산을 일정하게 유지한다.  
Internal Covariate Shift 문제를 줄여 훈련 안정성과 속도를 향상시킨다.

**사용법 (Keras)**  
```python
keras.layers.BatchNormalization()
```

---

## 2. 학습률 스케줄링 (Learning Rate Scheduling)
**역할**  
훈련 중 학습률을 동적으로 조정하여 초반에는 빠르게 학습하고, 후반에는 미세 조정이 가능하도록 한다.

**사용법**  
```python
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    0.01, 1000, 0.9
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
```

---

## 3. 데이터 증강 (Data Augmentation)
**역할**  
원본 이미지를 변형하여 데이터 다양성을 증가시키고 과대적합을 줄인다.

**사용법**  
```python
keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1)
])
```

---

## 4. 가중치 초기화 (Weight Initialization)
**역할**  
신경망 가중치의 초깃값을 설정하는 방식으로 학습 안정성과 속도에 큰 영향을 준다.

**대표 초기화 방법**  
- Glorot/Xavier 초깃값  
- He 초기화(ReLU 계열)

**사용법**  
```python
keras.layers.Dense(
    100, activation='relu',
    kernel_initializer='he_normal'
)
```

---

## 5. 드롭아웃 (Dropout)
**역할**  
훈련 중 일부 뉴런을 비활성화하여 과대적합을 효과적으로 방지한다.

**사용법**  
```python
keras.layers.Dropout(0.3)
```

---

## 6. 조기 종료 (Early Stopping)
**역할**  
검증 손실이 더 이상 좋아지지 않을 때 훈련을 자동으로 중단하여 과대적합과 시간 낭비를 방지한다.

**사용법**  
```python
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
```

---

## 7. 모델 체크포인트 (ModelCheckpoint)
**역할**  
훈련 중 가장 성능이 좋은 모델을 자동으로 저장한다.

**사용법**  
```python
checkpoint = keras.callbacks.ModelCheckpoint(
    'best-model.h5',
    save_best_only=True,
    monitor='val_loss'
)
```

---

## 8. 옵티마이저 선택 (Optimizer Selection)
**역할**  
가중치 업데이트 방식이며 성능과 훈련 속도에 큰 영향을 준다.

**대표 알고리즘**  
SGD, Momentum, Nesterov, Adagrad, RMSprop, Adam 등

**사용**  
```python
optimizer = keras.optimizers.Adam()
```

---

## 9. L1, L2 정규화 (Regularization)
**역할**  
가중치 크기가 커지는 것을 방지하여 과대적합을 줄인다.

**사용법**  
```python
keras.layers.Dense(
    100,
    kernel_regularizer=keras.regularizers.l2(0.001)
)
```
