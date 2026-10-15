"""goai - 19x19 전용 바둑 AI 엔진 패키지.

구성 요소:
    board       : 바둑 규칙 엔진 (착수, 따냄, 패, 자살수 금지, 계가)
    sgf         : SGF 기보 포맷 읽기/쓰기
    features    : 신경망 입력 특징 평면 인코딩
    model       : 정책망 + 가치망 (PyTorch)
    mcts        : PUCT 몬테카를로 트리 탐색
    timecontrol : 제한시간 + 초읽기 관리
    handicap    : 접바둑 돌 배치
    judge       : 넘기기 / 기권 / 떡수 판정 로직
    engine      : 위 요소를 종합하는 최상위 엔진
    dataset     : sgf_games 폴더 -> 학습용 통합 파일 변환
    trainer     : 지도학습 + 셀프플레이 강화학습 루프
    selfplay    : 셀프플레이 기보 생성
    feedback    : 사용자의 "아니다" 피드백 저장/반영
    cli         : 커맨드라인 인터페이스
"""

__version__ = "0.1.0"
