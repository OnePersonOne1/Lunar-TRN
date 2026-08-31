// 시연 재생 (P8): 궤적 csv(t,x,y,z — scripts/run_closed_loop.py --traj-out)를 따라
// 간이 착륙선을 이동시키고 추적 카메라로 보여준다. 센서 경로(RenderServer/TrnCamera)와 무관.
// 좌표 변환은 RenderServer.LToUnity()를 쓴다 (계약 §2.1 준수).

using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class TrajectoryPlayback : MonoBehaviour
{
    public string trajCsv = "../frames/traj_demo.csv"; // 프로젝트 루트(unity/) 기준
    public float timeScale = 8.0f;                     // 단일 배속 — Display 2/3/4 공통 시계
    public bool captureFrames = false;                 // Play 중 PNG 저장 (ffmpeg로 mp4)
    public string captureDir = "../frames/demo";
    public int captureFps = 30;
    public Vector3 chaseOffset = new Vector3(-1100f, 320f, 0f); // 비행(동서 장축) 방향 후방 정렬
    public int targetDisplay = 1;                      // 0=Display1, 1=Display2 (센서 미리보기와 분리)
    public string overlayDir = "../frames/p6";         // P6 실런 오버레이 루트 (t_c = 인덱스+1 초)
    public int gtDisplay = 2;                          // Display 3: 카탈로그 투영 (GT) — overlayDir/gt
    public int detDisplay = 3;                         // Display 4: YOLO 실추론 박스 — overlayDir/det
    public int telemetryDisplay = 4;                   // Display 5: 고도/속력/다운레인지 그래프

    private readonly List<float> _t = new();
    private readonly List<Vector3> _pos = new();       // Unity 좌표
    private readonly List<Vector3> _vel = new();       // Unity 좌표 (csv에 v열 있을 때)
    private TelemetryView _telemetry;
    private Transform _lander;
    private Camera _cam;
    private float _clock;
    private int _frameIdx;
    private float _nextCapture;
    private int _capIdx;
    private class OverlayPanel
    {
        public string dir;
        public Texture2D tex;
        public Material mat;
        public int idx = -1;
    }
    private readonly List<OverlayPanel> _overlays = new();

    void Start()
    {
        if (!LoadCsv(trajCsv)) { enabled = false; return; }
        _lander = BuildLander();
        var camGo = new GameObject("ChaseCamera");
        _cam = camGo.AddComponent<Camera>();
        _cam.farClipPlane = 300000f;
        _cam.nearClipPlane = 5f;
        _cam.fieldOfView = 45f;
        _cam.depth = 10f;                              // Main/기타 카메라보다 위에 표시
        _cam.clearFlags = CameraClearFlags.SolidColor;
        _cam.backgroundColor = Color.black;
        // 캡처(ScreenCapture)는 Display 1만 찍으므로, 캡처 모드에선 Display 1로 강제하고
        // 센서 미리보기(OnGUI)를 꺼서 착륙 장면만 영상에 담는다.
        if (captureFrames)
        {
            _cam.targetDisplay = 0;
            var rs = FindFirstObjectByType<RenderServer>();
            if (rs != null) rs.previewOnScreen = false;
            Directory.CreateDirectory(captureDir);
        }
        else
        {
            _cam.targetDisplay = targetDisplay;
        }
        SetupOverlayView();
        _telemetry = gameObject.AddComponent<TelemetryView>();
        _telemetry.Init(_t, _pos, _vel, telemetryDisplay);
        Debug.Log($"[TrajectoryPlayback] {_t.Count} rows, {_t[_t.Count - 1]:F0} s, x{timeScale}");
    }

    // Display 3/4: P6 실런의 카탈로그 투영(GT)·YOLO 탐지 오버레이를 재생 시각에 동기해 표시.
    // 다른 카메라 far plane(≤300 km) 밖 먼 좌표에 쿼드를 두어 장면 간섭을 피한다.
    void SetupOverlayView()
    {
        if (!Directory.Exists(overlayDir))
        {
            Debug.LogWarning($"[TrajectoryPlayback] 오버레이 없음: {overlayDir} — Display 3/4 생략");
            return;
        }
        // gt/·det/가 없으면(구 프레임) 합본(루트)으로 대체 — 재생성 안내 로그
        string gtDir = Path.Combine(overlayDir, "gt");
        string detDir = Path.Combine(overlayDir, "det");
        if (!Directory.Exists(gtDir) || !Directory.Exists(detDir))
        {
            Debug.LogWarning("[TrajectoryPlayback] gt/·det/ 프레임 없음 — 합본으로 대체. " +
                             "run_closed_loop.py --measurement unity --frames-dir frames/p6 재실행 필요");
            gtDir = overlayDir;
            detDir = overlayDir;
        }
        CreateOverlayPanel(gtDir, gtDisplay, new Vector3(-4000f, 150000f, -500000f), "GtOverlay");
        CreateOverlayPanel(detDir, detDisplay, new Vector3(4000f, 150000f, -500000f), "DetOverlay");
    }

    void CreateOverlayPanel(string dir, int display, Vector3 basePos, string name)
    {
        // float 정밀도 주의: 수백만 좌표는 쿼드가 틀어진다 (TelemetryView 참고)
        const float Q = 100f;
        var quad = GameObject.CreatePrimitive(PrimitiveType.Quad);
        quad.name = name + "Quad";
        Destroy(quad.GetComponent<Collider>());
        quad.transform.position = basePos;
        quad.transform.localScale = new Vector3(Q, Q, 1f);
        var panel = new OverlayPanel
        {
            dir = dir,
            tex = new Texture2D(2, 2),
            mat = new Material(Shader.Find("Unlit/Texture")),
        };
        quad.GetComponent<Renderer>().material = panel.mat;
        var camGo = new GameObject(name + "Camera");
        var cam = camGo.AddComponent<Camera>();
        cam.orthographic = true;
        cam.orthographicSize = Q * 0.5f;
        cam.nearClipPlane = 1f;
        cam.farClipPlane = 1000f;
        cam.transform.position = basePos + new Vector3(0f, 0f, -100f);
        cam.clearFlags = CameraClearFlags.SolidColor;
        cam.backgroundColor = Color.black;
        cam.targetDisplay = display;
        cam.depth = 11f;
        _overlays.Add(panel);
    }

    void UpdateOverlay(float t)
    {
        int idx = Mathf.Max(0, Mathf.FloorToInt(t) - 1);  // t_c = 인덱스 + 1 s
        foreach (var p in _overlays)
        {
            if (idx == p.idx) continue;
            string path = Path.Combine(p.dir, $"{idx:00000}.png");
            if (!File.Exists(path)) continue;              // 밴드 밖: 마지막 프레임 유지
            p.tex.LoadImage(File.ReadAllBytes(path));
            p.mat.mainTexture = p.tex;
            p.idx = idx;
        }
    }

    bool LoadCsv(string path)
    {
        if (!File.Exists(path))
        {
            Debug.LogError($"[TrajectoryPlayback] 궤적 csv 없음: {path} — " +
                           "scripts/run_closed_loop.py --measurement truth --traj-out frames/traj_demo.csv");
            return false;
        }
        foreach (string line in File.ReadAllLines(path))
        {
            string[] p = line.Split(',');
            if (p.Length < 4 || !float.TryParse(p[0], out float t)) continue; // 헤더 스킵
            _t.Add(t);
            _pos.Add(RenderServer.LToUnity(float.Parse(p[1]), float.Parse(p[2]), float.Parse(p[3])));
            _vel.Add(p.Length >= 7
                ? RenderServer.LToUnity(float.Parse(p[4]), float.Parse(p[5]), float.Parse(p[6]))
                : Vector3.zero);
        }
        return _t.Count >= 2;
    }

    static Transform BuildLander()
    {
        var root = new GameObject("Lander").transform;
        var body = GameObject.CreatePrimitive(PrimitiveType.Cube).transform;   // 본체
        body.SetParent(root, false);
        body.localScale = new Vector3(120f, 90f, 120f);                        // 시연용 과장 스케일
        var nozzle = GameObject.CreatePrimitive(PrimitiveType.Cylinder).transform;
        nozzle.SetParent(root, false);
        nozzle.localPosition = new Vector3(0f, -75f, 0f);
        nozzle.localScale = new Vector3(45f, 30f, 45f);
        for (int i = 0; i < 4; i++)                                            // 다리 4개
        {
            var leg = GameObject.CreatePrimitive(PrimitiveType.Cylinder).transform;
            leg.SetParent(root, false);
            float a = Mathf.PI * (0.25f + 0.5f * i);
            leg.localPosition = new Vector3(90f * Mathf.Cos(a), -70f, 90f * Mathf.Sin(a));
            leg.localRotation = Quaternion.Euler(25f * Mathf.Sin(a), 0f, -25f * Mathf.Cos(a));
            leg.localScale = new Vector3(14f, 65f, 14f);
        }
        foreach (var col in root.GetComponentsInChildren<Collider>()) Destroy(col);
        return root;
    }

    void Update()
    {
        float scale = timeScale;
        _clock += Time.deltaTime * scale;
        float tEnd = _t[_t.Count - 1];
        float t = Mathf.Min(_clock, tEnd);
        while (_frameIdx < _t.Count - 2 && _t[_frameIdx + 1] <= t) _frameIdx++;
        float seg = Mathf.InverseLerp(_t[_frameIdx], _t[_frameIdx + 1], t);
        Vector3 p = Vector3.Lerp(_pos[_frameIdx], _pos[_frameIdx + 1], seg);
        _lander.position = p;

        // 추적 카메라: 진행 방향 뒤쪽 상방에서 착륙선을 바라봄, 고도에 따라 줌인
        float h01 = Mathf.Clamp01(p.y / 30000f);
        Vector3 off = chaseOffset * Mathf.Lerp(0.25f, 1.0f, h01);
        _cam.transform.position = p + off;
        _cam.transform.LookAt(p);
        UpdateOverlay(t);
        if (_telemetry != null) _telemetry.SetTime(t);

        if (captureFrames && _clock >= _nextCapture && _clock <= tEnd + 1f)
        {
            ScreenCapture.CaptureScreenshot(Path.Combine(captureDir, $"demo_{_capIdx++:00000}.png"));
            _nextCapture += scale / captureFps;
        }
    }
}
