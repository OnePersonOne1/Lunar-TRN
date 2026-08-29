// TCP 렌더 서버 (P4). Unity 좌표계 변환(x=East, y=Up, z=North; L 원점 = 월드 원점)은 이 파일 안에서만 한다 (계약 §2.1).
//
// 프로토콜:
//   요청: UTF-8 JSON 한 줄 ('\n' 종료) {"frame_id":int, "t":float, "r_L":[x,y,z], "sun_az_deg":float, "sun_el_deg":float}
//   응답: 4바이트 big-endian 길이 + 페이로드. 페이로드는 PNG(0x89 'P' 'N' 'G'로 시작) 또는 UTF-8 JSON {"error": "..."}.
// 태양각: sun_az는 북에서 시계방향(deg), sun_el은 지평선 기준 고도각(deg).

using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

[Serializable]
public class RenderRequest
{
    public int frame_id;
    public float t;
    public float[] r_L;
    public float sun_az_deg;
    public float sun_el_deg;
}

public class RenderServer : MonoBehaviour
{
    public int port = 5555;                  // config.yaml unity.port와 일치시킬 것
    public Camera renderCamera;              // SceneBuilder가 연결
    public Light sun;                        // Directional Light "Sun"
    public int imageWidth = 1024;            // config.yaml camera.W
    public int imageHeight = 1024;           // config.yaml camera.H
    public bool previewOnScreen = true;      // Game 뷰에 마지막 렌더 프레임 표시 (검수용)

    private TcpListener _listener;
    private Thread _acceptThread;
    private volatile bool _running;

    private readonly ConcurrentQueue<(RenderRequest req, NetworkStream stream)> _pending = new();

    void Start()
    {
        if (renderCamera == null) renderCamera = GetComponent<Camera>();
        ConfigureQuality();
        _listener = new TcpListener(IPAddress.Loopback, port);
        _listener.Start();
        _running = true;
        _acceptThread = new Thread(AcceptLoop) { IsBackground = true };
        _acceptThread.Start();
        Debug.Log($"[RenderServer] listening on 127.0.0.1:{port}");
    }

    // 센서 모사 품질 설정 (Play 시 1회): 밴드 고도(16.5~30 km)의 원거리 카메라에서도
    // 지형·텍스처가 전해상도로 렌더되고, 조명이 태양 직사광만으로 구성되게 한다.
    void ConfigureQuality()
    {
        // 에디터/플레이어가 상한 없이 최대 fps로 재렌더해 GPU를 소진하는 것 방지.
        // 측정 렌더는 요청 시 renderCamera.Render()로 수동 실행이라 영향 없음.
        Application.targetFrameRate = 30;
        QualitySettings.vSyncCount = 0;
        // 달: 대기 없음 — 주변광·안개 제거, 배경은 검정(우주)
        RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Flat;
        RenderSettings.ambientLight = Color.black;
        RenderSettings.fog = false;
        // 크레이터 림 그림자가 원거리 카메라에서도 렌더되게
        QualitySettings.shadowDistance = 60000f;
        QualitySettings.shadowCascades = 4;
        if (renderCamera != null)
        {
            renderCamera.clearFlags = CameraClearFlags.SolidColor;
            renderCamera.backgroundColor = Color.black;
            renderCamera.allowHDR = false;
            renderCamera.allowMSAA = false;
        }
        foreach (Terrain t in Terrain.activeTerrains)
        {
            t.basemapDistance = 500000f;            // 저해상도 base map 회피 (버전에 따라 클램프될 수 있음)
            t.heightmapPixelError = 1f;             // 지형 기하 LOD 최소화
            // 주의: baseMapResolution을 런타임에 바꾸면 알파맵(패딩 검정 스플랫)이 초기화된다.
            // 해상도는 SceneBuilder(빌드 시)가 설정한다 — 여기서는 절대 대입하지 않는다.
        }
        Debug.Log($"[RenderServer] quality set: basemapDistance={(Terrain.activeTerrain != null ? Terrain.activeTerrain.basemapDistance : -1f)}");
    }

    void AcceptLoop()
    {
        while (_running)
        {
            try
            {
                TcpClient client = _listener.AcceptTcpClient();
                var t = new Thread(() => ClientLoop(client)) { IsBackground = true };
                t.Start();
            }
            catch (SocketException) { /* listener 종료 */ }
        }
    }

    void ClientLoop(TcpClient client)
    {
        using (client)
        using (var stream = client.GetStream())
        using (var reader = new StreamReader(stream, Encoding.UTF8))
        {
            string line;
            try
            {
                while (_running && (line = reader.ReadLine()) != null)
                {
                    RenderRequest req = null;
                    try { req = JsonUtility.FromJson<RenderRequest>(line); }
                    catch (Exception e) { SendError(stream, "bad json: " + e.Message); continue; }
                    if (req == null || req.r_L == null || req.r_L.Length != 3)
                    {
                        SendError(stream, "r_L must be [x,y,z]");
                        continue;
                    }
                    // 응답은 메인 스레드(Update)에서 같은 stream으로 보낸다.
                    // 클라이언트는 동기(응답을 읽은 뒤에야 다음 요청)이므로 여기서는 다음 ReadLine으로 대기.
                    _pending.Enqueue((req, stream));
                }
            }
            catch (IOException) { /* 클라이언트 연결 종료 */ }
        }
    }

    void Update()
    {
        while (_pending.TryDequeue(out var item))
        {
            try
            {
                byte[] png = Render(item.req);
                SendPayload(item.stream, png);
            }
            catch (Exception e)
            {
                try { SendError(item.stream, e.Message); } catch { /* 무시 */ }
            }
        }
    }

    // L → Unity 좌표 변환 (계약 §2.1: 이 변환은 이 파일 안에서만 정의한다).
    // L(ENU: x=East, y=North, z=Up) → Unity(x=East, y=Up, z=North).
    public static Vector3 LToUnity(float xE, float yN, float zU)
    {
        return new Vector3(xE, zU, yN);
    }

    byte[] Render(RenderRequest req)
    {
        // --- L → Unity: x=East, y=Up, z=North. L 원점 = Unity 월드 (0,0,0).
        Vector3 posUnity = LToUnity(req.r_L[0], req.r_L[1], req.r_L[2]);
        renderCamera.transform.position = posUnity;
        // nadir: forward = -Up, 이미지 상단 = 북(+z)
        renderCamera.transform.rotation = Quaternion.LookRotation(Vector3.down, Vector3.forward);

        // --- 태양: az 북 기준 시계방향, el 지평선 고도각 (§2.1)
        float az = req.sun_az_deg * Mathf.Deg2Rad;
        float el = req.sun_el_deg * Mathf.Deg2Rad;
        // L 프레임 태양 방향(지상→태양): E=sin(az)cos(el), N=cos(az)cos(el), U=sin(el)
        Vector3 sunDirL = new Vector3(
            Mathf.Sin(az) * Mathf.Cos(el),
            Mathf.Cos(az) * Mathf.Cos(el),
            Mathf.Sin(el));
        Vector3 sunDirUnity = new Vector3(sunDirL.x, sunDirL.z, sunDirL.y);
        sun.transform.rotation = Quaternion.LookRotation(-sunDirUnity); // 빛의 진행 방향

        // --- 렌더 → PNG
        RenderTexture rt = renderCamera.targetTexture;
        if (rt == null || rt.width != imageWidth || rt.height != imageHeight)
        {
            rt = new RenderTexture(imageWidth, imageHeight, 24);
            renderCamera.targetTexture = rt;
        }
        renderCamera.Render();
        RenderTexture prev = RenderTexture.active;
        RenderTexture.active = rt;
        var tex = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        tex.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        tex.Apply();
        RenderTexture.active = prev;
        byte[] png = tex.EncodeToPNG();
        Destroy(tex);
        return png;
    }

    // Game 뷰 검수용: TrnCamera는 오프스크린 RenderTexture에만 그리므로,
    // 마지막 렌더 프레임을 화면에 그대로 띄운다 (요청이 올 때마다 갱신됨).
    void OnGUI()
    {
        if (!previewOnScreen || renderCamera == null || renderCamera.targetTexture == null) return;
        RenderTexture rt = renderCamera.targetTexture;
        float s = Mathf.Min((float)Screen.width / rt.width, (float)Screen.height / rt.height);
        float w = rt.width * s, h = rt.height * s;
        GUI.DrawTexture(new Rect((Screen.width - w) / 2f, (Screen.height - h) / 2f, w, h), rt,
                        ScaleMode.ScaleToFit, false);
        GUI.Label(new Rect(8, 8, 600, 24), "[RenderServer preview] last rendered frame");
    }

    static void SendPayload(NetworkStream stream, byte[] payload)
    {
        byte[] len = new byte[4];
        len[0] = (byte)((payload.Length >> 24) & 0xFF);
        len[1] = (byte)((payload.Length >> 16) & 0xFF);
        len[2] = (byte)((payload.Length >> 8) & 0xFF);
        len[3] = (byte)(payload.Length & 0xFF);
        lock (stream)
        {
            stream.Write(len, 0, 4);
            stream.Write(payload, 0, payload.Length);
            stream.Flush();
        }
    }

    static void SendError(NetworkStream stream, string msg)
    {
        SendPayload(stream, Encoding.UTF8.GetBytes("{\"error\": " + JsonString(msg) + "}"));
    }

    static string JsonString(string s)
    {
        return "\"" + s.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";
    }

    void OnDestroy()
    {
        _running = false;
        _listener?.Stop();
    }
}
