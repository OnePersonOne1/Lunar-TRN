// 시연 텔레메트리 (P8): 고도/속력/다운레인지 실시간 스트립차트 + 범례·눈금·수치를
// 전부 텍스처에 픽셀 폰트로 직접 그린다(폰트 에셋 의존 없음 — TextMesh는 기본 폰트가
// 없으면 렌더되지 않아 교체). 3-DOF라 자세 상태는 없음(CLAUDE.md §1). 측정 경로와 무관.

using System.Collections.Generic;
using UnityEngine;

public class TelemetryView : MonoBehaviour
{
    const int W = 1024, H = 720;
    const int X0 = 78, X1 = 1016;          // 플롯 영역 (왼쪽 여백 = y눈금 라벨)
    static readonly Color Bg = new Color(0.97f, 0.97f, 0.98f);
    static readonly Color Grid = new Color(0.82f, 0.84f, 0.87f);
    static readonly Color Ink = new Color(0.12f, 0.13f, 0.15f);
    static readonly Color Tick = new Color(0.45f, 0.47f, 0.52f);

    // 5x7 픽셀 폰트 (필요 글리프만)
    static readonly Dictionary<char, string[]> Glyphs = new()
    {
        ['0'] = new[] { "01110", "10001", "10011", "10101", "11001", "10001", "01110" },
        ['1'] = new[] { "00100", "01100", "00100", "00100", "00100", "00100", "01110" },
        ['2'] = new[] { "01110", "10001", "00001", "00010", "00100", "01000", "11111" },
        ['3'] = new[] { "11110", "00001", "00001", "01110", "00001", "00001", "11110" },
        ['4'] = new[] { "00010", "00110", "01010", "10010", "11111", "00010", "00010" },
        ['5'] = new[] { "11111", "10000", "11110", "00001", "00001", "10001", "01110" },
        ['6'] = new[] { "00110", "01000", "10000", "11110", "10001", "10001", "01110" },
        ['7'] = new[] { "11111", "00001", "00010", "00100", "01000", "01000", "01000" },
        ['8'] = new[] { "01110", "10001", "10001", "01110", "10001", "10001", "01110" },
        ['9'] = new[] { "01110", "10001", "10001", "01111", "00001", "00010", "01100" },
        ['A'] = new[] { "01110", "10001", "10001", "11111", "10001", "10001", "10001" },
        ['D'] = new[] { "11110", "10001", "10001", "10001", "10001", "10001", "11110" },
        ['E'] = new[] { "11111", "10000", "10000", "11110", "10000", "10000", "11111" },
        ['K'] = new[] { "10001", "10010", "10100", "11000", "10100", "10010", "10001" },
        ['L'] = new[] { "10000", "10000", "10000", "10000", "10000", "10000", "11111" },
        ['M'] = new[] { "10001", "11011", "10101", "10101", "10001", "10001", "10001" },
        ['P'] = new[] { "11110", "10001", "10001", "11110", "10000", "10000", "10000" },
        ['S'] = new[] { "01111", "10000", "10000", "01110", "00001", "00001", "11110" },
        ['T'] = new[] { "11111", "00100", "00100", "00100", "00100", "00100", "00100" },
        ['/'] = new[] { "00001", "00010", "00010", "00100", "01000", "01000", "10000" },
        ['['] = new[] { "01110", "01000", "01000", "01000", "01000", "01000", "01110" },
        [']'] = new[] { "01110", "00010", "00010", "00010", "00010", "00010", "01110" },
        ['.'] = new[] { "00000", "00000", "00000", "00000", "00000", "01100", "01100" },
        ['-'] = new[] { "00000", "00000", "00000", "01110", "00000", "00000", "00000" },
        [' '] = new[] { "00000", "00000", "00000", "00000", "00000", "00000", "00000" },
    };

    private Texture2D _tex;
    private List<float> _t;
    private List<Vector3> _pos;
    private List<float> _spd;
    private float _tEnd;
    private int _lastCol = -1;

    private struct Panel { public int y0, h; public float lo, hi; public Color c; public string legend; }
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
            new Panel { y0 = 470, h = 190, lo = 0f, hi = hMax * 1.05f,
                        c = new Color(0.08f, 0.32f, 0.78f), legend = "ALT [KM]" },
            new Panel { y0 = 250, h = 190, lo = 0f, hi = sMax * 1.05f,
                        c = new Color(0.82f, 0.40f, 0.03f), legend = "SPD [M/S]" },
            new Panel { y0 = 30,  h = 190, lo = xMin * 1.05f, hi = 0f,
                        c = new Color(0.03f, 0.48f, 0.16f), legend = "EAST [KM]" },
        };

        _tex = new Texture2D(W, H, TextureFormat.RGB24, false);
        var px = new Color[W * H];
        for (int i = 0; i < px.Length; i++) px[i] = Bg;
        _tex.SetPixels(px);

        foreach (var p in _panels)
        {
            // 테두리·가로 눈금선 + y눈금 라벨 (0/50/100%)
            for (int gy = 0; gy <= 4; gy++) DrawHLine(X0, X1, p.y0 + p.h * gy / 4, Grid);
            for (int gy = 0; gy <= 2; gy++)
            {
                float val = Mathf.Lerp(p.lo, p.hi, gy / 2f);
                string lab = FormatVal(val, p);
                DrawText(4, p.y0 + p.h * gy / 2 - 7, lab, Tick, 2);
            }
            // 세로 눈금선: 50 s 간격
            for (float ts = 0f; ts <= _tEnd; ts += 50f)
                DrawVLine(TimeToCol(ts), p.y0, p.y0 + p.h, Grid);
            DrawText(X0 + 8, p.y0 + p.h - 18, p.legend, p.c, 2);
        }
        // 시간축 라벨 (최하단)
        for (float ts = 0f; ts <= _tEnd; ts += 50f)
            DrawText(TimeToCol(ts) - 10, 8, $"{ts:F0}", Tick, 2);
        DrawText(X1 - 30, 8, "S", Tick, 2);
        _tex.Apply();

        Vector3 basePos = new Vector3(3000000f, 3000000f, 3000000f);
        float sy = (float)H / W;
        var quad = GameObject.CreatePrimitive(PrimitiveType.Quad);
        quad.name = "TelemetryQuad";
        Destroy(quad.GetComponent<Collider>());
        quad.transform.position = basePos;
        quad.transform.localScale = new Vector3(1f, sy, 1f);
        quad.GetComponent<Renderer>().material =
            new Material(Shader.Find("Unlit/Texture")) { mainTexture = _tex };
        var camGo = new GameObject("TelemetryCamera");
        var cam = camGo.AddComponent<Camera>();
        cam.orthographic = true;
        cam.orthographicSize = sy * 0.5f;
        cam.nearClipPlane = 0.1f;
        cam.farClipPlane = 10f;
        cam.transform.position = basePos + new Vector3(0f, 0f, -1f);
        cam.clearFlags = CameraClearFlags.SolidColor;
        cam.backgroundColor = Bg;
        cam.targetDisplay = targetDisplay;
        cam.depth = 12f;
    }

    string FormatVal(float v, Panel p)
    {
        // ALT/EAST는 km, SPD는 m/s 그대로
        bool km = p.legend.Contains("KM");
        float x = km ? v / 1000f : v;
        return x >= 100f || x <= -100f ? $"{x:F0}" : $"{x:F1}";
    }

    int TimeToCol(float t) => X0 + Mathf.RoundToInt(t / _tEnd * (X1 - X0 - 1));

    void DrawHLine(int x0, int x1, int y, Color c)
    {
        for (int x = x0; x < x1; x++) _tex.SetPixel(x, y, c);
    }

    void DrawVLine(int x, int y0, int y1, Color c)
    {
        for (int y = y0; y < y1; y++) _tex.SetPixel(x, y, c);
    }

    void DrawText(int x, int y, string s, Color c, int scale)
    {
        foreach (char ch in s.ToUpperInvariant())
        {
            if (Glyphs.TryGetValue(ch, out var g))
                for (int r = 0; r < 7; r++)
                    for (int cc = 0; cc < 5; cc++)
                        if (g[r][cc] == '1')
                            for (int dy = 0; dy < scale; dy++)
                                for (int dx = 0; dx < scale; dx++)
                                    _tex.SetPixel(x + cc * scale + dx, y + (6 - r) * scale + dy, c);
            x += 6 * scale;
        }
    }

    void FillRect(int x0, int y0, int x1, int y1, Color c)
    {
        for (int y = y0; y < y1; y++)
            for (int x = x0; x < x1; x++)
                _tex.SetPixel(x, y, c);
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
        int col = Mathf.Clamp(TimeToCol(t), X0, X1 - 1);
        float spd = Sample(_spd, t, out int idx);
        Vector3 p = Vector3.Lerp(_pos[idx], _pos[Mathf.Min(idx + 1, _pos.Count - 1)],
                                 Mathf.InverseLerp(_t[idx], _t[Mathf.Min(idx + 1, _t.Count - 1)], t));
        float[] vals = { p.y, spd, p.x };
        if (col > _lastCol)
        {
            for (int c2 = Mathf.Max(_lastCol + 1, X0); c2 <= col; c2++)
                for (int k = 0; k < 3; k++)
                {
                    var pn = _panels[k];
                    float frac = Mathf.Clamp01(Mathf.InverseLerp(pn.lo, pn.hi, vals[k]));
                    int y = pn.y0 + 2 + Mathf.RoundToInt(frac * (pn.h - 5));
                    for (int dy = -2; dy <= 2; dy++)
                        _tex.SetPixel(c2, Mathf.Clamp(y + dy, pn.y0, pn.y0 + pn.h - 1), pn.c);
                }
            _lastCol = col;
        }
        // 상단 실시간 수치 (매 프레임 재그림)
        FillRect(0, 668, W, H, Bg);
        DrawText(10, 690, $"T {t:F1} S", Ink, 3);
        DrawText(230, 690, $"ALT {p.y / 1000f:F2} KM", _panels[0].c, 3);
        DrawText(520, 690, $"SPD {spd:F0} M/S", _panels[1].c, 3);
        DrawText(790, 690, $"E {p.x / 1000f:F1} KM", _panels[2].c, 3);
        _tex.Apply();
    }
}
