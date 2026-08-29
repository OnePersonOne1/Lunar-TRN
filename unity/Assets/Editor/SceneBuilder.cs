// 에디터 메뉴 "LunarTRN/Build Scene": DEM 하이트맵으로 달 지형 씬 구성 (P4).
// data/processed/heightmap_meta.json + heightmap.raw + texture_L.png 필요 (data/crop.py 산출).
// L 원점(착륙 목표점)이 Unity 월드 (0,0,0)에 오도록 Terrain을 오프셋한다.

using System;
using System.IO;
using UnityEditor;
using UnityEngine;

[Serializable]
public class HeightmapSize
{
    public float east;
    public float north;
    public float padded_square;
}

[Serializable]
public class HeightmapMeta
{
    public int resolution;
    public HeightmapSize size_m;
    public int used_rows;
    public int used_cols;
    public float z_min_m;
    public float z_max_m;
}

public static class SceneBuilder
{
    // 프로젝트 루트 기준 상대 경로: Unity 프로젝트가 lunar-trn/unity/ 안에 있다고 가정 (README 참고)
    const string ProcessedDir = "../data/processed";
    const float CameraFovVDeg = 60.0f;   // config.yaml camera.fov_v_deg
    const int ImageSize = 1024;          // config.yaml camera.W = H
    const float FarClip = 200000.0f;     // ≥ 200 km

    [MenuItem("LunarTRN/Build Scene")]
    public static void BuildScene() { BuildFrom(ProcessedDir); }

    // 미학습 지역 일반화 test용(STATUS 2026-08-28): 같은 씬을 고지대 데이터로 재구성.
    // 본 site로 되돌리려면 일반 Build Scene을 다시 실행한다.
    [MenuItem("LunarTRN/Build Scene (Highlands)")]
    public static void BuildSceneHighlands() { BuildFrom(ProcessedDir + "/highlands"); }

    // 시연(P8): 궤적 재생 + 간이 착륙선 + 추적 카메라. 지형이 이미 빌드돼 있어야 한다.
    // 궤적: scripts/run_closed_loop.py --measurement truth --traj-out frames/traj_demo.csv
    [MenuItem("LunarTRN/Add Demo Playback")]
    public static void AddDemoPlayback()
    {
        if (GameObject.Find("LunarTerrain") == null)
        {
            Debug.LogError("지형이 없다 — 먼저 LunarTRN/Build Scene을 실행해라.");
            return;
        }
        var go = GameObject.Find("DemoPlayback");
        if (go == null) go = new GameObject("DemoPlayback");
        if (go.GetComponent<TrajectoryPlayback>() == null) go.AddComponent<TrajectoryPlayback>();
        // 시연용 조명 프리셋 (측정과 무관): 저녁 사광으로 그림자 강조
        var sunGo = GameObject.Find("Sun");
        if (sunGo != null)
            sunGo.transform.rotation = Quaternion.Euler(25f, 200f, 0f);
        Debug.Log("[SceneBuilder] DemoPlayback 추가. Inspector에서 captureFrames를 켜면 " +
                  "Play 중 frames/demo/에 PNG를 저장한다 (ffmpeg로 mp4).");
    }

    static void BuildFrom(string processedDirIn)
    {
        string metaPath = Path.Combine(processedDirIn, "heightmap_meta.json");
        string rawPath = Path.Combine(processedDirIn, "heightmap.raw");
        string texPath = Path.Combine(processedDirIn, "texture_L.png");
        if (!File.Exists(metaPath) || !File.Exists(rawPath))
        {
            Debug.LogError($"heightmap 데이터가 없다: {metaPath} / {rawPath} (data/crop.py를 먼저 실행)");
            return;
        }
        var meta = JsonUtility.FromJson<HeightmapMeta>(File.ReadAllText(metaPath));
        int res = meta.resolution;

        // --- RAW(uint16 LE, 첫 행 = 북) → heights[z축(남→북), x축(서→동)]
        byte[] bytes = File.ReadAllBytes(rawPath);
        if (bytes.Length != res * res * 2)
        {
            Debug.LogError($"RAW 크기 불일치: {bytes.Length} != {res}*{res}*2");
            return;
        }
        float[,] heights = new float[res, res];
        for (int row = 0; row < res; row++)          // row 0 = 북
        {
            int zIdx = res - 1 - row;                // Unity: z 인덱스 증가 = 북쪽
            for (int col = 0; col < res; col++)
            {
                ushort v = (ushort)(bytes[2 * (row * res + col)] | (bytes[2 * (row * res + col) + 1] << 8));
                heights[zIdx, col] = v / 65535.0f;
            }
        }

        // --- TerrainData
        var td = new TerrainData();
        td.heightmapResolution = res;
        float ySize = Mathf.Max(meta.z_max_m - meta.z_min_m, 1.0f);
        // heightmap은 정사각(padded_square) 격자: 지형 x=z=padded_square
        td.size = new Vector3(meta.size_m.padded_square, ySize, meta.size_m.padded_square);
        td.SetHeights(0, 0, heights);

        // --- 텍스처 레이어 (사용 영역 east×north에 1:1)
        // 텍스처는 반드시 에셋으로 임포트해 저장한다 — 메모리 Texture2D는 스크립트
        // 재컴파일(도메인 리로드) 때 파괴되어 체크무늬(missing texture)가 된다.
        if (File.Exists(texPath))
        {
            const string texAsset = "Assets/LunarTexture.png";
            File.Copy(texPath, texAsset, true);
            AssetDatabase.ImportAsset(texAsset);
            var imp = (TextureImporter)AssetImporter.GetAtPath(texAsset);
            imp.maxTextureSize = 4096;                    // 원본 3401px — 기본 2048 다운스케일 방지
            imp.npotScale = TextureImporterNPOTScale.None;
            imp.mipmapEnabled = true;
            imp.textureCompression = TextureImporterCompression.Uncompressed;
            imp.wrapMode = TextureWrapMode.Repeat;
            imp.SaveAndReimport();
            var tex = AssetDatabase.LoadAssetAtPath<Texture2D>(texAsset);
            var layer = new TerrainLayer
            {
                diffuseTexture = tex,
                tileSize = new Vector2(meta.size_m.east, meta.size_m.north),
                tileOffset = Vector2.zero,
            };
            AssetDatabase.CreateAsset(layer, "Assets/LunarTerrainLayer.asset");

            // 패딩(사용 영역 북쪽) 반복 텍스처 억제: 검정 레이어를 알파맵으로 패딩에만 적용
            AssetDatabase.DeleteAsset("Assets/LunarPaddingBlack.asset");
            AssetDatabase.DeleteAsset("Assets/LunarPaddingLayer.asset");
            var blackTex = new Texture2D(4, 4);
            var px = new Color[16];
            for (int i = 0; i < 16; i++) px[i] = Color.black;
            blackTex.SetPixels(px);
            blackTex.Apply();
            AssetDatabase.CreateAsset(blackTex, "Assets/LunarPaddingBlack.asset");
            var padLayer = new TerrainLayer
            {
                diffuseTexture = blackTex,
                tileSize = new Vector2(10000f, 10000f),
            };
            AssetDatabase.CreateAsset(padLayer, "Assets/LunarPaddingLayer.asset");
            td.terrainLayers = new[] { layer, padLayer };

            int amapRes = 513;
            td.alphamapResolution = amapRes;
            float[,,] amap = new float[amapRes, amapRes, 2];
            float northFrac = meta.size_m.north / meta.size_m.padded_square;
            for (int j = 0; j < amapRes; j++)          // j: 지형 로컬 z (남→북)
            {
                bool used = (j + 0.5f) / amapRes <= northFrac;
                for (int i = 0; i < amapRes; i++)
                {
                    amap[j, i, 0] = used ? 1f : 0f;
                    amap[j, i, 1] = used ? 0f : 1f;
                }
            }
            td.SetAlphamaps(0, 0, amap);
            Debug.Log($"[SceneBuilder] 패딩 검정 스플랫 적용 (used northFrac={northFrac:F3})");
        }
        else
        {
            Debug.LogWarning($"texture_L.png 없음: {texPath} — 지형 텍스처 생략");
        }
        AssetDatabase.CreateAsset(td, "Assets/LunarTerrainData.asset");

        // --- Terrain 배치: L 원점(사용 영역 중심, z=0 고도)이 월드 (0,0,0)
        // 재실행 시 중복 생성 방지 (Unity 오브젝트는 ??/?. 금지 — 파괴된 객체가 가짜 null이라 == null로만 검사)
        for (var old = GameObject.Find("LunarTerrain"); old != null; old = GameObject.Find("LunarTerrain"))
            UnityEngine.Object.DestroyImmediate(old);
        var terrainGo = Terrain.CreateTerrainGameObject(td);
        terrainGo.name = "LunarTerrain";
        terrainGo.transform.position = new Vector3(
            -meta.size_m.east / 2.0f, meta.z_min_m, -meta.size_m.north / 2.0f);

        // --- 태양
        var sunGo = GameObject.Find("Sun");
        if (sunGo == null) sunGo = new GameObject("Sun");
        var sun = sunGo.GetComponent<Light>();
        if (sun == null) sun = sunGo.AddComponent<Light>();
        sun.type = LightType.Directional;
        sun.shadows = LightShadows.Soft;

        // --- 카메라 + RenderServer
        var camGo = GameObject.Find("TrnCamera");
        if (camGo == null) camGo = new GameObject("TrnCamera");
        var cam = camGo.GetComponent<Camera>();
        if (cam == null) cam = camGo.AddComponent<Camera>();
        cam.fieldOfView = CameraFovVDeg;      // Unity fieldOfView = 수직 FOV
        cam.farClipPlane = FarClip;
        cam.nearClipPlane = 10.0f;
        cam.targetTexture = new RenderTexture(ImageSize, ImageSize, 24);
        var server = camGo.GetComponent<RenderServer>();
        if (server == null) server = camGo.AddComponent<RenderServer>();
        server.renderCamera = cam;
        server.sun = sun;
        server.imageWidth = ImageSize;
        server.imageHeight = ImageSize;

        Debug.Log("[SceneBuilder] 완료. Play 모드에서 RenderServer가 포트를 연다.");
    }
}
