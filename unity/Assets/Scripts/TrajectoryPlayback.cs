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
    public float timeScale = 12.0f;                    // 350 s 비행 → 약 29 s 재생
    public bool captureFrames = false;                 // Play 중 PNG 저장 (ffmpeg로 mp4)
    public string captureDir = "../frames/demo";
    public int captureFps = 30;
    public Vector3 chaseOffset = new Vector3(-900f, 350f, -550f); // 착륙선 기준 카메라 위치

    private readonly List<float> _t = new();
    private readonly List<Vector3> _pos = new();       // Unity 좌표
    private Transform _lander;
    private Camera _cam;
    private float _clock;
    private int _frameIdx;
    private float _nextCapture;
    private int _capIdx;

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
        if (captureFrames) Directory.CreateDirectory(captureDir);
        Debug.Log($"[TrajectoryPlayback] {_t.Count} rows, {_t[_t.Count - 1]:F0} s, x{timeScale}");
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
        _clock += Time.deltaTime * timeScale;
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

        if (captureFrames && _clock >= _nextCapture && _clock <= tEnd + 1f)
        {
            ScreenCapture.CaptureScreenshot(Path.Combine(captureDir, $"demo_{_capIdx++:00000}.png"));
            _nextCapture += timeScale / captureFps;
        }
    }
}
