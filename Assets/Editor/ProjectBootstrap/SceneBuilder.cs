using System.IO;
using System.Linq;
using Mujoco;
using TMPro;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;

namespace Microduck.Editor
{
    public static class SceneBuilder
    {
        [MenuItem("Microduck/Rebuild Demo Scene")]
        public static void Build()
        {
            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            Time.fixedDeltaTime = .005f;
            PlayerSettings.runInBackground = true;
            PlayerSettings.companyName = "Microduck Lab"; PlayerSettings.productName = "Microduck Policy Lab";
            PlayerSettings.defaultScreenWidth = 1440; PlayerSettings.defaultScreenHeight = 900;
            PlayerSettings.allowUnsafeCode = true;
            var plugin = AssetImporter.GetAtPath("Assets/Plugins/macOS/mujoco.dylib") as PluginImporter;
            plugin.SetCompatibleWithAnyPlatform(false); plugin.SetCompatibleWithEditor(true);
            plugin.SetEditorData("OS", "OSX"); plugin.SetEditorData("CPU", "AnyCPU");
            plugin.SetCompatibleWithPlatform(BuildTarget.StandaloneOSX, true); plugin.SaveAndReimport();
            var xml = Path.Combine(Application.streamingAssetsPath, "Microduck/robot/scene_bam.xml");
            var root = new MjImporterWithAssets().ImportFile(xml);
            if (!root) throw new System.Exception("MuJoCo importer did not produce a robot.");
            root.name = "Microduck Visuals";
            // ExecuteInEditMode.Update has not ticked during this synchronous import.
            foreach (var filter in root.GetComponentsInChildren<MjMeshFilter>()) filter.SendMessage("Update");
            var bodies = root.GetComponentsInChildren<MjBody>().OrderBy(b => Depth(b.transform)).Select(b => new BodyVisual { bodyName = b.name, transform = b.transform }).ToArray();
            // Retain the importer output as an inspectable reference, but use a
            // stripped visual hierarchy in the demo. Native MJCF remains authoritative.
            Directory.CreateDirectory("Assets/Microduck/Prefabs");
            PrefabUtility.SaveAsPrefabAsset(root, "Assets/Microduck/Prefabs/ImportedMicroduck.prefab");
            foreach (var filter in root.GetComponentsInChildren<MjMeshFilter>()) filter.enabled = false;
            foreach (var filter in root.GetComponentsInChildren<MjMeshFilter>(true)) Object.DestroyImmediate(filter);
            foreach (var component in root.GetComponentsInChildren<MjComponent>(true)) Object.DestroyImmediate(component);
            foreach (var global in root.GetComponentsInChildren<MjGlobalSettings>(true)) Object.DestroyImmediate(global);
            foreach (var scene in Object.FindObjectsByType<MjScene>()) Object.DestroyImmediate(scene.gameObject);
            // Collision-only geoms are hidden. Original materials/meshes are preserved.
            foreach (var renderer in root.GetComponentsInChildren<MeshRenderer>())
            {
                if (renderer.name == "floor" || renderer.name.EndsWith("_collision")) renderer.enabled = false;
                else renderer.shadowCastingMode = ShadowCastingMode.On;
            }
            var app = new GameObject("Microduck Runtime");
            var sim = app.AddComponent<MicroduckSimulation>(); sim.Bodies = bodies;
            sim.PolicyAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/Microduck/Models/microduck_walk_1024_model_1500.onnx");
            var camera = new GameObject("Main Camera", typeof(Camera), typeof(AudioListener)).GetComponent<Camera>();
            camera.tag = "MainCamera"; camera.fieldOfView = 40; camera.nearClipPlane = .01f; camera.farClipPlane = 100;
            camera.clearFlags = CameraClearFlags.SolidColor; camera.backgroundColor = new Color(.085f, .12f, .14f);
            camera.transform.SetPositionAndRotation(new Vector3(.6f, .4f, -.6f), Quaternion.Euler(20, -45, 0));
            var view = app.AddComponent<MicroduckView>(); view.Simulation = sim; view.ViewCamera = camera;
            view.Font = AssetDatabase.LoadAssetAtPath<TMP_FontAsset>("Assets/TextMesh Pro/Resources/Fonts & Materials/LiberationSans SDF.asset");
            if (!view.Font) throw new System.Exception("Import TMP essential resources before building the scene.");
            var sun = new GameObject("Soft Key", typeof(Light)).GetComponent<Light>(); sun.type = LightType.Directional;
            sun.intensity = 1.2f; sun.color = new Color(1, .95f, .86f); sun.shadows = LightShadows.Soft; sun.transform.rotation = Quaternion.Euler(50, -35, 0);
            var fill = new GameObject("Fill", typeof(Light)).GetComponent<Light>(); fill.type = LightType.Directional; fill.intensity = .35f; fill.color = new Color(.7f, .85f, 1); fill.transform.rotation = Quaternion.Euler(40, 140, 0);
            RenderSettings.ambientMode = AmbientMode.Flat; RenderSettings.ambientLight = new Color(.5f, .55f, .6f);
            QualitySettings.shadows = ShadowQuality.All; QualitySettings.shadowDistance = 15;
            BuildFloor();
            Directory.CreateDirectory("Assets/Microduck/Scenes");
            string path = "Assets/Microduck/Scenes/MicroduckLab.unity";
            EditorSceneManager.SaveScene(EditorSceneManager.GetActiveScene(), path);
            EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(path, true) };
            AssetDatabase.SaveAssets();
            Debug.Log("MICRODUCK_SCENE_BUILT " + path + " / bodies=" + bodies.Length);
        }
        static int Depth(Transform t) { int n = 0; while (t.parent) { n++; t = t.parent; } return n; }
        static void BuildFloor()
        {
            Directory.CreateDirectory("Assets/Microduck/Materials");
            var material = AssetDatabase.LoadAssetAtPath<Material>("Assets/Microduck/Materials/Floor.mat");
            bool newMaterial = !material; if (!material) material = new Material(Shader.Find("Standard"));
            material.color = new Color(.2f, .27f, .29f); material.SetFloat("_Glossiness", .12f);
            var tex = AssetDatabase.LoadAssetAtPath<Texture2D>("Assets/Microduck/Materials/Grid.asset");
            bool newTexture = !tex; if (!tex) tex = new Texture2D(128, 128); tex.wrapMode = TextureWrapMode.Repeat;
            for (int y = 0; y < 128; y++) for (int x = 0; x < 128; x++)
                tex.SetPixel(x, y, x < 2 || y < 2 ? new Color(.38f, .46f, .47f) : new Color(.65f, .72f, .73f));
            tex.Apply(); if (newTexture) AssetDatabase.CreateAsset(tex, "Assets/Microduck/Materials/Grid.asset");
            material.mainTexture = tex; material.mainTextureScale = new Vector2(100, 100);
            if (newMaterial) AssetDatabase.CreateAsset(material, "Assets/Microduck/Materials/Floor.mat");
            var floor = GameObject.CreatePrimitive(PrimitiveType.Plane); floor.name = "Visual Grid / 0.1m"; floor.transform.localScale = Vector3.one;
            Object.DestroyImmediate(floor.GetComponent<Collider>()); floor.GetComponent<MeshRenderer>().sharedMaterial = material;
        }
    }
}
