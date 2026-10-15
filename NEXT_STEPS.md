# 작업 현황 및 다음 단계

## 완료되고 검증된 부분
- `goai/board.py`: 19x19 전용 바둑 규칙 엔진 (Zobrist 해시 기반 고성능 버전)
  - 착수/캡처/자살수 금지/패(situational superko) 금지/중국식 계가/기권 처리
  - `tests/test_board.py` 6개 테스트 전체 통과 (캡처, 자살수, 패, 패스, 기권, 계가)
- `goai/features.py`: 신경망 입력 특징 평면 인코딩 (9채널, 19x19)
- `goai/model.py`: 정책망+가치망 ResNet (PyTorch, CPU 전용 경량 구성)
- `goai/sgf.py`: SGF 읽기/쓰기 (GameRecord, parse_sgf, game_to_sgf)
- `goai/timecontrol.py`: 제한시간 1분 + 초읽기 30초x3회 (PlayerClock)
- `goai/handicap.py`: 접바둑 2~9점 화점 배치
- `goai/dataset.py`: sgf_games 폴더 -> 통합 npz 데이터셋 변환 (동작 확인됨)
- `goai/mcts.py`: PUCT MCTS (동작 확인됨, 단 engine/selfplay와 API 재동기화 필요)
- `goai/judge.py`: 떡수/패스/기권 판정 로직 (ActionKind/Decision/CorrectionMemory 체계)

## 남은 작업 (다음 세션에서 이어서 진행 필요)
1. **API 불일치 해소**: `engine.py`, `feedback.py`, `selfplay.py`, `trainer.py`가
   참조하는 클래스명이 현재 `judge.py`/`mcts.py`의 실제 클래스명과 다름.
   - `judge.py`의 최종 API(`ActionKind`, `Decision`, `CorrectionMemory`, `Judge`)를
     기준으로 `engine.py`/`feedback.py`를 재작성해야 함.
   - `mcts.py`의 최종 API(`MCTSResult`, `PASS_INDEX`, `MCTS.run(board, color, n_simulations=...)`)를
     기준으로 `selfplay.py`를 재작성해야 함.
2. **`goai/cli.py` 복구**: 현재 누락됨. `python -m goai.cli play/train/selfplay/build-dataset`
   커맨드 제공 목적으로 재작성 필요.
3. **떡수 피드백 연동 확인**: `judge.py`의 `CorrectionMemory`가 사용자 "아니다" 피드백을
   저장/반영하는 방식으로 설계된 것으로 보이므로, 별도 `feedback.py` 대신
   이 메커니즘을 활용하도록 통일 검토.
4. **통합 테스트**: 전체 모듈 import 체크 + 엔진 한 판 시뮬레이션(사람 vs AI, 접바둑 포함)
   + 셀프플레이 1판 + 지도학습 1 epoch 가 에러 없이 도는지 end-to-end 테스트 추가.
5. **동시 편집 충돌 원인 확인**: 작업 중 동일 파일이 반복적으로 예기치 않게
   바뀌는 현상이 있었음 (다른 프로세스가 같은 워크스페이스를 동시 수정하는 것으로 추정).
   다음 세션 시작 시 먼저 `git diff`/파일 상태를 재확인할 것.

## 요구사항 반영 메모
- 19x19 전용 ✅ (board.py BOARD_SIZE=19 고정)
- SGF 기보 폴더 저장 + 통합 파일 변환 ✅ (dataset.py 동작 확인)
- 제한시간 1분 + 초읽기 30초 3회 ✅ (timecontrol.py)
- 접바둑 2~9점 ✅ (handicap.py)
- 넘기기/기권/떡수 판정 + 사용자 "아니다" 피드백 ⚠️ 로직은 구현됐으나 모듈 간 연결(engine.py) 재정비 필요
- 어느정도 학습 후 셀프플레이 ⚠️ selfplay.py는 작성되었으나 mcts.py 최신 API와 재동기화 필요
