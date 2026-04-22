# Seoul Sci Graduation Photo Finder

졸업사진을 업로드하면 얼굴 임베딩 기반으로 자동 그룹화하고, 그룹별 사진을 열람하는 내부용 MVP입니다.

## 실행

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

브라우저에서 `http://127.0.0.1:8000` 접속.

## 기능

- 다중 사진 업로드
- 사진별 얼굴 검출 + 임베딩
- 유사 얼굴 자동 그룹화
- 그룹별 사진 열람
- 관리자 그룹 병합

## 비고

- `face_recognition` 라이브러리가 설치되지 않은 환경에서는 deterministic fallback 모드로 동작합니다.
