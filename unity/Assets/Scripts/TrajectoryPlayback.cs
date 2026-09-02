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
    public string overlayDir = "../frames/p6";         // P6 실런 오버레이 (t_c = 인덱스+1 초)
    public int overlayDisplay = 2;                     // Display 3: 매칭/미탐지/FP 색 구분 합본
    public int telemetryDisplay = 3;                   // Display 4: 고도/속력/다운레인지 그래프

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
    private Texture2D _overlayTex;
    private Material _overlayMat;
    private int _overlayIdx = -1;
    private Camera _overlayCam;
    private RenderTexture _rt2, _rt3, _rt4;   // 캡처용 (Display 2/3/4에 대응)
    private Texture2D _cap2, _cap3, _cap4;

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
        // 캡처는 카메라별 RenderTexture로 직접 찍는다 — Game 뷰가 어떤 Display를
        // 보고 있든 무관하고, 세 화면(추적/오버레이/텔레메트리)을 동시에 저장한다.
        _cam.targetDisplay = targetDisplay;
        if (captureFrames)
        {
            Directory.CreateDirectory(captureDir);
            // 이전 녹화 잔여 프레임 제거 — 남으면 번호가 이어져 ffmpeg 인코딩 시 구 녹화와 섞인다
            foreach (string pat in new[] { "d2_*.jpg", "d3_*.jpg", "d4_*.png", "demo_*.png" })
                foreach (string f in Directory.GetFiles(captureDir, pat))
                    File.Delete(f);
            _rt2 = new RenderTexture(1280, 720, 24);
            _cap2 = new Texture2D(1280, 720, TextureFormat.RGB24, false);
            _rt3 = new RenderTexture(1024, 1024, 24);
            _cap3 = new Texture2D(1024, 1024, TextureFormat.RGB24, false);
            _rt4 = new RenderTexture(1024, 720, 24);
            _cap4 = new Texture2D(1024, 720, TextureFormat.RGB24, false);
        }
        SetupOverlayView();
        _telemetry = gameObject.AddComponent<TelemetryView>();
        _telemetry.Init(_t, _pos, _vel, telemetryDisplay);
        Debug.Log($"[TrajectoryPlayback] {_t.Count} rows, {_t[_t.Count - 1]:F0} s, x{timeScale}");
    }

    // Display 3: P6 실런의 센서 카메라+탐지 오버레이(매칭/미탐지/FP 색 구분)를 재생 시각에 동기해 표시.
    // 다른 카메라 far plane(≤300 km) 밖 먼 좌표에 쿼드를 두어 장면 간섭을 피한다.
    void SetupOverlayView()
    {
        if (!Directory.Exists(overlayDir))
        {
            Debug.LogWarning($"[TrajectoryPlayback] 오버레이 없음: {overlayDir} — Display 3 생략");
            return;
        }
        // float 정밀도 주의: 수백만 좌표는 쿼드가 틀어진다 (TelemetryView 참고)
        Vector3 basePos = new Vector3(0f, 150000f, -500000f);
        const float Q = 100f;
        var quad = GameObject.CreatePrimitive(PrimitiveType.Quad);
        quad.name = "SensorOverlayQuad";
        Destroy(quad.GetComponent<Collider>());
        quad.transform.position = basePos;
        quad.transform.localScale = new Vector3(Q, Q, 1f);
        _overlayTex = new Texture2D(2, 2);
        _overlayMat = new Material(Shader.Find("Unlit/Texture"));
        quad.GetComponent<Renderer>().material = _overlayMat;
        var camGo = new GameObject("SensorViewCamera");
        var cam = camGo.AddComponent<Camera>();
        cam.orthographic = true;
        cam.orthographicSize = Q * 0.5f;
        cam.nearClipPlane = 1f;
        cam.farClipPlane = 1000f;
        cam.transform.position = basePos + new Vector3(0f, 0f, -100f);
        cam.clearFlags = CameraClearFlags.SolidColor;
        cam.backgroundColor = Color.black;
        cam.targetDisplay = overlayDisplay;
        cam.depth = 11f;
        _overlayCam = cam;
    }

    // 카메라 한 대를 RenderTexture로 렌더해 PNG/JPG 바이트로 반환 (표시 상태 불변)
    static void CaptureCam(Camera cam, RenderTexture rt, Texture2D tex, string path, bool jpg)
    {
        if (cam == null) return;
        var prevTarget = cam.targetTexture;
        var prevActive = RenderTexture.active;
        cam.targetTexture = rt;
        cam.Render();
        RenderTexture.active = rt;
        tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
        cam.targetTexture = prevTarget;
        RenderTexture.active = prevActive;
        File.WriteAllBytes(path, jpg ? tex.EncodeToJPG(92) : tex.EncodeToPNG());
    }

    void UpdateOverlay(float t)
    {
        if (_overlayMat == null) return;
        int idx = Mathf.Max(0, Mathf.FloorToInt(t) - 1);  // t_c = 인덱스 + 1 s
        if (idx == _overlayIdx) return;
        string path = Path.Combine(overlayDir, $"{idx:00000}.png");
        if (!File.Exists(path)) return;                    // 밴드 밖: 마지막 프레임 유지
        _overlayTex.LoadImage(File.ReadAllBytes(path));
        _overlayMat.mainTexture = _overlayTex;
        _overlayIdx = idx;
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

        if (captureFrames && _rt2 != null && _clock >= _nextCapture && _clock <= tEnd + 1f)
        {
            // 세 화면을 RenderTexture로 동시 캡처 — Game 뷰 표시 상태와 무관.
            // 사진성 화면(추적·오버레이)은 JPG, 평면색 텔레메트리는 PNG.
            CaptureCam(_cam, _rt2, _cap2,
                       Path.Combine(captureDir, $"d2_{_capIdx:00000}.jpg"), true);
            CaptureCam(_overlayCam, _rt3, _cap3,
                       Path.Combine(captureDir, $"d3_{_capIdx:00000}.jpg"), true);
            CaptureCam(_telemetry != null ? _telemetry.Cam : null, _rt4, _cap4,
                       Path.Combine(captureDir, $"d4_{_capIdx:00000}.png"), false);
            _capIdx++;
            _nextCapture += scale / captureFps;
        }
    }
}
