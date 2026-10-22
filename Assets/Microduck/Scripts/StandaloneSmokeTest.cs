using System;
using System.IO;
using UnityEngine;

namespace Microduck
{
    // Opt-in executable health check, dormant during normal use and in the Editor.
    // --microduck-smoke --microduck-report /absolute/path/report.json
    public class StandaloneSmokeTest : MonoBehaviour
    {
        [Serializable] class Report
        {
            public string unity_version;
            public bool passed;
            public int frames_observed, policy_steps, falls, errors;
            public double simulation_time_s, wall_time_s, final_tilt_degrees;
        }
        string output;
        int firstFrame, errors;
        float started;
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        static void Initialize()
        {
            if (Application.isEditor || !Array.Exists(Environment.GetCommandLineArgs(), a => a == "--microduck-smoke")) return;
            var go = new GameObject("Standalone health check"); DontDestroyOnLoad(go); go.AddComponent<StandaloneSmokeTest>();
        }
        void Awake()
        {
            var args = Environment.GetCommandLineArgs();
            int index = Array.IndexOf(args, "--microduck-report");
            output = index >= 0 && index + 1 < args.Length ? args[index + 1] : Path.Combine(Application.persistentDataPath, "microduck-smoke.json");
            firstFrame = Time.frameCount; started = Time.realtimeSinceStartup;
            Application.logMessageReceived += ObserveLog;
        }
        void ObserveLog(string message, string stack, LogType type) { if (type == LogType.Error || type == LogType.Exception || type == LogType.Assert) errors++; }
        void Update()
        {
            var sim = FindAnyObjectByType<MicroduckSimulation>();
            double elapsed = Time.realtimeSinceStartup - started;
            if ((sim && sim.Ready && sim.PolicySteps >= 600) || elapsed > 45 || errors > 0)
            {
                var report = new Report {
                    unity_version = Application.unityVersion, frames_observed = Time.frameCount - firstFrame,
                    policy_steps = sim ? sim.PolicySteps : 0, falls = sim ? sim.Falls : 0, errors = errors,
                    simulation_time_s = sim && sim.Ready ? sim.SimulationTime : 0, wall_time_s = elapsed,
                    final_tilt_degrees = sim ? sim.TiltDegrees : 0
                };
                report.passed = report.policy_steps >= 600 && report.frames_observed > 2 && report.falls == 0 && errors == 0;
                Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output)));
                File.WriteAllText(output, JsonUtility.ToJson(report, true));
                Debug.Log("MICRODUCK_STANDALONE_SMOKE " + JsonUtility.ToJson(report));
                Application.Quit(report.passed ? 0 : 1); enabled = false;
            }
        }
        void OnDestroy() => Application.logMessageReceived -= ObserveLog;
    }
}
