// 시연 텔레메트리 (P8): 고도/속력/다운레인지 실시간 스트립차트를 텍스처에 그려
// 지정 Display에 표시. 3-DOF라 자세 상태는 없음(카메라 nadir 고정, CLAUDE.md §1).
// TrajectoryPlayback이 생성·구동한다. 측정 경로와 무관.

using System.Collections.Generic;
using UnityEngine;

public class TelemetryView : MonoBehaviour
{
    const int W = 1024, H = 660;
    static readonly Color Bg = new Color(0.06f, 0.07f, 0.09f);
    static readonly Color Grid = new Color(0.22f, 0.24f, 0.28f);

    private Texture2D _tex;
    private List<float> _t;
    private List<Vector3> _pos;   // Unity 좌표 (y=Up)
    private List<float> _spd;
    private float _tEnd;
    private int _lastCol = -1;
    private TextMesh _readout;

    private struct Panel { public int y0, h; public float lo, hi; public Color c; public string name; }
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
            new Panel { y0 = 445, h = 200, lo = 0f, hi = hMax * 1.05f, c = new Color(0.35f, 0.75f, 1f), name = "ALT" },
            new Panel { y0 = 225, h = 200, lo = 0f, hi = sMax * 1.05f, c = new Color(1f, 0.72f, 0.25f), name = "SPD" },
            new Panel { y0 = 5,   h = 200, lo = xMin * 1.05f, hi = 0f,  c = new Color(0.5f, 1f, 0.55f), name = "E"   },
        };

        _tex = new Texture2D(W, H, TextureFormat.RGB24, false);
        var px = new Color[W * H];
        for (int i = 0; i < px.Length; i++) px[i] = Bg;
        _tex.SetPixels(px);
        foreach (var p in _panels)
            for (int gy = 0; gy <= 4; gy++)
                DrawHLine(p.y0 + p.h * gy / 4, Grid);
        _tex.Apply();

        Vector3 basePos = new Vector3(3000000f, 3000000f, 3000000f);
        var quad = GameObject.CreatePrimitive(PrimitiveType.Quad);
        quad.name = "TelemetryQuad";
        Destroy(quad.GetComponent<Collider>());
        quad.transform.position = basePos;
        quad.transform.localScale = new Vector3(1f, (float)H / W, 1f);
        var mat = new Material(Shader.Find("Unlit/Texture")) { mainTexture = _tex };
        quad.GetComponent<Renderer>().material = mat;

        var camGo = new GameObject("TelemetryCamera");
        var cam = camGo.AddComponent<Camera>();
        cam.orthographic = true;
        cam.orthographicSize = (float)H / W * 0.5f;
        cam.nearClipPlane = 0.1f;
        cam.farClipPlane = 10f;
        cam.transform.position = basePos + new Vector3(0f, 0f, -1f);
        cam.clearFlags = CameraClearFlags.SolidColor;
        cam.backgroundColor = Bg;
        cam.targetDisplay = targetDisplay;
        cam.depth = 12f;

        var txtGo = new GameObject("TelemetryReadout");
        _readout = txtGo.AddComponent<TextMesh>();
        _readout.characterSize = 0.014f;
        _readout.fontSize = 64;
        _readout.anchor = TextAnchor.UpperLeft;
        _readout.color = Color.white;
        txtGo.transform.position = basePos + new Vector3(-0.48f, (float)H / W * 0.5f - 0.01f, -0.1f);
    }

    void DrawHLine(int y, Color c)
    {
        for (int x = 0; x < W; x++) _tex.SetPixel(x, y, c);
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
                    int y = pn.y0 + Mathf.RoundToInt(frac * (pn.h - 1));
                    _tex.SetPixel(c2, y, pn.c);
                    _tex.SetPixel(c2, Mathf.Min(y + 1, pn.y0 + pn.h - 1), pn.c);
                }
            _tex.Apply();
            _lastCol = col;
        }
        _readout.text = $"t {t,6:F1} s   ALT {p.y / 1000f,6:F2} km   SPD {spd,6:F1} m/s   E {p.x / 1000f,7:F1} km";
    }
}
