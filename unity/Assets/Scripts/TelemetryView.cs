// 시연 텔레메트리 (P8): 고도/속력/다운레인지 실시간 스트립차트를 텍스처에 그려
// 지정 Display에 표시. 3-DOF라 자세 상태는 없음(카메라 nadir 고정, CLAUDE.md §1).
// TrajectoryPlayback이 생성·구동한다. 측정 경로와 무관. 밝은 배경(발표·프로젝터용).

using System.Collections.Generic;
using UnityEngine;

public class TelemetryView : MonoBehaviour
{
    const int W = 1024, H = 660;
    static readonly Color Bg = new Color(0.96f, 0.96f, 0.98f);
    static readonly Color Grid = new Color(0.80f, 0.82f, 0.86f);

    private Texture2D _tex;
    private List<float> _t;
    private List<Vector3> _pos;   // Unity 좌표 (y=Up)
    private List<float> _spd;
    private float _tEnd;
    private int _lastCol = -1;
    private TextMesh _readout;

    private struct Panel
    {
        public int y0, h;
        public float lo, hi;
        public Color c;
        public string label;
    }
    private Panel[] _panels;

    public void Init(List<float> t, List<Vector3> pos, List<Vector3> vel, int targetDisplay)
    {
        _t = t; _pos = pos;
        _tEnd = t[t.Count - 1];
        _spd = new List<float>(vel.Count);
        foreach (var v in vel) _spd.Add(v.magnitude);
        float hMax = 0f, sMax = 0f, xMin = 0f;
        foreach (var p in pos) { hMax = Mathf.Max(hMax, p.y); xMin = Mathf.Min(xMin, p.x); }
        foreach (var s in _spd) sMax = Mathf.Max(sMax, s);
        _panels = new[]
        {
            new Panel { y0 = 448, h = 196, lo = 0f, hi = hMax * 1.05f,
                        c = new Color(0.10f, 0.35f, 0.80f), label = $"ALT [km]  0 ~ {hMax / 1000f:F0}" },
            new Panel { y0 = 228, h = 196, lo = 0f, hi = sMax * 1.05f,
                        c = new Color(0.85f, 0.42f, 0.05f), label = $"SPEED [m/s]  0 ~ {sMax:F0}" },
            new Panel { y0 = 8,   h = 196, lo = xMin * 1.05f, hi = 0f,
                        c = new Color(0.05f, 0.52f, 0.18f), label = $"EAST [km]  {xMin / 1000f:F0} ~ 0" },
        };

        _tex = new Texture2D(W, H, TextureFormat.RGB24, false);
        var px = new Color[W * H];
        for (int i = 0; i < px.Length; i++) px[i] = Bg;
        _tex.SetPixels(px);
        foreach (var p in _panels)
        {
            for (int gy = 0; gy <= 4; gy++) DrawHLine(p.y0 + p.h * gy / 4, Grid);
            for (int gx = 0; gx <= 8; gx++)
                DrawVSeg(Mathf.Min(W - 1, gx * (W - 1) / 8), p.y0, p.y0 + p.h, Grid);
        }
        _tex.Apply();

        Vector3 basePos = new Vector3(3000000f, 3000000f, 3000000f);
        float sy = (float)H / W;
        var quad = GameObject.CreatePrimitive(PrimitiveType.Quad);
        quad.name = "TelemetryQuad";
        Destroy(quad.GetComponent<Collider>());
        quad.transform.position = basePos;
        quad.transform.localScale = new Vector3(1f, sy, 1f);
        var mat = new Material(Shader.Find("Unlit/Texture")) { mainTexture = _tex };
        quad.GetComponent<Renderer>().material = mat;

        var camGo = new GameObject("TelemetryCamera");
        var cam = camGo.AddComponent<Camera>();
        cam.orthographic = true;
        cam.orthographicSize = sy * 0.5f + 0.035f;   // 상단 수치 표시 여백
        cam.nearClipPlane = 0.1f;
        cam.farClipPlane = 10f;
        cam.transform.position = basePos + new Vector3(0f, 0.03f, -1f);
        cam.clearFlags = CameraClearFlags.SolidColor;
        cam.backgroundColor = Bg;
        cam.targetDisplay = targetDisplay;
        cam.depth = 12f;

        _readout = MakeText(basePos + new Vector3(-0.49f, sy * 0.5f + 0.055f, -0.1f),
                            Color.black, TextAnchor.UpperLeft, 60);
        // 패널별 컬러 범례 (텍스처 좌표 → 월드 좌표: y_px/H - 0.5 배율 sy)
        foreach (var p in _panels)
        {
            float yTop = ((p.y0 + p.h - 6f) / H - 0.5f) * sy;
            var tm = MakeText(basePos + new Vector3(-0.485f, yTop, -0.1f),
                              p.c, TextAnchor.UpperLeft, 52);
            tm.text = p.label;
        }
    }

    static TextMesh MakeText(Vector3 pos, Color color, TextAnchor anchor, int fontSize)
    {
        var go = new GameObject("TelemetryText");
        var tm = go.AddComponent<TextMesh>();
        tm.characterSize = 0.012f;
        tm.fontSize = fontSize;
        tm.anchor = anchor;
        tm.color = color;
        go.transform.position = pos;
        return tm;
    }

    void DrawHLine(int y, Color c)
    {
        for (int x = 0; x < W; x++) _tex.SetPixel(x, y, c);
    }

    void DrawVSeg(int x, int y0, int y1, Color c)
    {
        for (int y = y0; y < y1; y++) _tex.SetPixel(x, y, c);
    }

    float Sample(List<float> series, float t, out int idx)
    {
        idx = 0;
        while (idx < _t.Count - 2 && _t[idx + 1] <= t) idx++;
        float s = Mathf.InverseLerp(_t[idx], _t[idx + 1], t);
        return Mathf.Lerp(series[idx], series[idx + 1], s);
    }

    public void SetTime(float t)
    {
        if (_tex == null) return;
        int col = Mathf.Clamp(Mathf.RoundToInt(t / _tEnd * (W - 1)), 0, W - 1);
        float spd = Sample(_spd, t, out int idx);
        Vector3 p = Vector3.Lerp(_pos[idx], _pos[Mathf.Min(idx + 1, _pos.Count - 1)],
                                 Mathf.InverseLerp(_t[idx], _t[Mathf.Min(idx + 1, _t.Count - 1)], t));
        float[] vals = { p.y, spd, p.x };
        if (col > _lastCol)
        {
            for (int c2 = _lastCol + 1; c2 <= col; c2++)
                for (int k = 0; k < 3; k++)
                {
                    var pn = _panels[k];
                    float frac = Mathf.Clamp01(Mathf.InverseLerp(pn.lo, pn.hi, vals[k]));
                    int y = pn.y0 + 2 + Mathf.RoundToInt(frac * (pn.h - 5));
                    for (int dy = -2; dy <= 2; dy++)                       // 5 px 두께
                        _tex.SetPixel(c2, Mathf.Clamp(y + dy, pn.y0, pn.y0 + pn.h - 1), pn.c);
                }
            _tex.Apply();
            _lastCol = col;
        }
        _readout.text = $"t {t,6:F1} s    ALT {p.y / 1000f,6:F2} km    SPD {spd,6:F1} m/s    E {p.x / 1000f,7:F1} km";
    }
}
