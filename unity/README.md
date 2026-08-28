# unity — 렌더(센서 모사) 파트

Unity는 센서 모사 전용이다. 좌표계 매핑(x=East, y=Up, z=North, L 원점 = 월드 (0,0,0))은
`Assets/Scripts/RenderServer.cs` 안에서만 처리한다 (계약 §2.1).

## 수동 설정 절차 (Editor GUI)

1. Unity Hub에서 **Unity 6 LTS(또는 2022.3 LTS 이상)**, 3D (Built-in Render Pipeline) 템플릿으로
   새 프로젝트를 만든다. **Hub는 항상 `<위치>/<프로젝트명>` 하위 폴더를 새로 만들므로**,
   이 저장소의 `unity/`를 프로젝트 루트로 쓰려면:
   1) 아무 위치(임시)에 새 프로젝트 생성 → 에디터가 열리면 그냥 종료
   2) 생성된 프로젝트의 `Packages/`, `ProjectSettings/` 폴더를 이 저장소 `unity/` 바로 밑으로 이동
   3) 임시 프로젝트 폴더 삭제
   4) Hub → **Add**(디스크에서 프로젝트 추가) → 이 저장소의 `unity/` 선택 → 열기
   이렇게 하면 `unity/Assets`의 스크립트가 그대로 잡혀 아래 3번(복사)이 필요 없다.
   (Library/Temp 등 생성 파일은 .gitignore에 이미 있음)
2. 사전 조건: `data/processed/`에 `heightmap.raw`, `heightmap_meta.json`, `texture_L.png`가
   있어야 한다 (`data/crop.py` 산출). 없으면 SceneBuilder가 에러 로그를 낸다.
3. 이 저장소의 `unity/Assets/Editor/SceneBuilder.cs`와 `unity/Assets/Scripts/RenderServer.cs`를
   Unity 프로젝트의 같은 경로(`Assets/Editor/`, `Assets/Scripts/`)에 복사한다.
   (Unity 프로젝트가 `unity/`가 아니라면 `SceneBuilder.cs`의 `ProcessedDir` 상대 경로를
   `data/processed` 절대 경로로 수정한다.)
4. 컴파일 완료 후 메뉴 **LunarTRN → Build Scene** 실행. Terrain("LunarTerrain"),
   Directional Light("Sun"), 카메라("TrnCamera", RenderServer 부착)가 생성된다.
5. TrnCamera 선택 → Inspector에서 RenderServer의 `port`가 config.yaml `unity.port`(5555)와
   같은지 확인한다.
6. **Play** 버튼을 누른다. Console에 `[RenderServer] listening on 127.0.0.1:5555`가 떠야 한다.
7. 저장소 루트에서 연결 확인:
   `.venv\Scripts\python -c "import yaml,numpy as np; from unity.client import RenderClient; c=RenderClient(yaml.safe_load(open('config.yaml',encoding='utf-8'))); img=c.render(np.array([0,0,30000.0]),135,30); print(img.shape)"`
   → `(1024, 1024, 3)`이 나오면 정상.
8. 이후 `scripts/check_projection.py`로 정합(5 px 이내)을 검증한다.

## 흔한 오류

1. **이미지 상하 반전**: `ReadPixels`는 그래픽 API에 따라 세로 방향이 뒤집힐 수 있다.
   check_projection에서 북쪽 크레이터가 화면 아래에 나오면 RenderServer.Render()에서
   PNG 인코딩 전에 세로 플립을 넣는다 (카메라 규약: 이미지 상단 = 북, v는 남쪽으로 증가).
2. **좌우(동서) 반전**: Unity는 왼손 좌표계다. 동쪽 크레이터가 왼쪽에 나오면
   RenderServer의 L→Unity 매핑(x=E, y=U, z=N)과 카메라 up 벡터(Vector3.forward)를 확인한다.
3. **고도 스케일 이상(지형이 평평/과장)**: TerrainData.size의 y가 `z_max_m − z_min_m`인지,
   Terrain 위치의 y가 `z_min_m`인지 확인한다.
4. **텍스처 반전/타일 반복**: TerrainLayer tileSize가 (east, north) 크기와 같은지 확인.
   heightmap은 정사각으로 패딩되므로 북쪽 패딩 영역(사용 영역 밖)은 무시한다.
5. **RAW 바이트 오더**: heightmap.raw는 little-endian uint16, 첫 행 = 북쪽이다.
   지형이 노이즈처럼 보이면 바이트 오더, 남북이 뒤집히면 행 순서 문제다.
6. **포트 사용 중**: 이전 Play 세션이 남아 있으면 포트가 잠긴다. Play를 완전히 멈추고 재시작.
7. **원거리 클리핑**: 카메라 farClipPlane ≥ 200 km(SceneBuilder가 설정). 고고도에서 지형이
   안 보이면 이 값을 확인한다.
