# GoAI - 19x19 바둑 AI 엔진

AlphaZero 스타일(규칙엔진 + 정책/가치망 + MCTS + 셀프플레이)로 구성된
19x19 전용 바둑 AI. 사람과 대국하며 넘기기/기권을 스스로 판단하고,
사용자가 "아니다"라고 교정하면 그 피드백을 즉시 반영 + 영구 저장합니다.

## 구성
- `goai/board.py` : 19x19 규칙 엔진 (착수/따냄/패/자충수금지/슈퍼코/중국식 계가)
- `goai/sgf.py` : SGF 읽기/쓰기
- `goai/timecontrol.py` : 제한시간 1분 + 초읽기 30초x3
- `goai/features.py`, `goai/model.py` : 신경망 입력/경량 ResNet (정책+가치)
- `goai/mcts.py` : PUCT 몬테카를로 트리 탐색
- `goai/judge.py` : 넘기기/기권/떡수 판단 + 근거 로깅 + 사용자 교정("아니다") 저장
- `goai/engine.py` : `GameSession` - 위 모든 요소를 통합한 대국 세션
- `goai/dataset.py` : `기보/` 폴더의 SGF들을 하나의 학습 데이터셋으로 병합
- `goai/trainer.py` : 지도학습 스크립트
- `goai/selfplay.py` : 셀프플레이로 자동 대국 생성 (`기보/selfplay/`에 SGF 저장)
- `기보/` : SGF 기보 저장 폴더 (사용자가 기보를 넣는 곳)
- `data/` : 데이터셋(.npz), 모델 체크포인트(.pt), 사용자 교정 로그(corrections.jsonl)

## 사용 흐름
```bash
# 1) 기보/ 폴더에 .sgf 파일들을 넣는다

# 2) 하나의 데이터셋으로 병합
python3 -m goai.dataset

# 3) 지도학습
python3 -m goai.trainer --epochs 5

# 4) 셀프플레이로 추가 기보 생성 (기보/selfplay/ 에 저장됨)
python3 -m goai.selfplay --games 10

# (반복) 1~4를 돌며 점점 강해짐

# 5) 사람 vs AI 대국 (파이썬에서)
python3 -c "
from goai.engine import GameSession
sess = GameSession()
sess.human_move((3,3))       # 사람이 (3,3)에 착수
decision = sess.ai_move()    # AI가 수/패스/기권 결정
print(decision.reason)       # AI의 판단 근거
# 판단이 부당하면:
sess.reject_last_ai_decision("그건 떡수입니다")  # 교정 -> 저장 + 즉시 재결정
print(sess.to_sgf())         # SGF로 내보내기
"
```

## 참고
- 리소스가 제한된 샌드박스(CPU 2코어, RAM ~1GB, GPU 없음) 환경을 고려해
  네트워크를 경량(채널 32, 블록 4)으로 설계했습니다. 더 강해지려면
  더 많은 기보/셀프플레이 데이터와 더 큰 네트워크, GPU 환경이 필요합니다.
