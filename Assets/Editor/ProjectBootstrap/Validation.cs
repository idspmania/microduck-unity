using System;
using System.IO;
using Mujoco;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

namespace Microduck.Editor
{
    public static unsafe class Validation
    {
        [Serializable] class InferenceCase { public string name; public float[] observation, action; }
        [Serializable] class InferenceCases { public InferenceCase[] cases; }
        [Serializable] class BamCase { public double q, target, velocity, previous_motor_torque, external_torque, voltage, torque, friction; }
        [Serializable] class BamCases { public BamCase[] cases; }
        [Serializable] class PhysicsStep { public double[] control, qpos, qvel; }
        [Serializable] class PhysicsCases { public double[] initial_qpos; public PhysicsStep[] steps; }
        [Serializable] class Report
        {
            public string unity_version, native_version, sentis_backend;
            public int inference_cases, bam_cases, physics_steps;
            public double inference_max_absolute_error, bam_max_absolute_error, physics_max_absolute_error;
            public bool passed;
        }
        static string PathFor(string file) => Path.Combine(Application.streamingAssetsPath, "Microduck", file);
        [MenuItem("Microduck/Validate Policy and Physics")]
        public static void Run()
        {
            var report = new Report { unity_version = Application.unityVersion, native_version = System.Runtime.InteropServices.Marshal.PtrToStringAnsi(MujocoLib.mj_versionString()), sentis_backend = "CPU" };
            var cases = JsonUtility.FromJson<InferenceCases>(File.ReadAllText(PathFor("inference_cases.json"))).cases;
            using (var policy = new Policy(AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/Microduck/Models/microduck_walk_1024_model_1500.onnx")))
                foreach (var c in cases)
                {
                    var output = policy.Evaluate(c.observation);
                    for (int i = 0; i < 14; i++) report.inference_max_absolute_error = Math.Max(report.inference_max_absolute_error, Math.Abs(output[i] - c.action[i]));
                    report.inference_cases++;
                }
            var bam = BamParameters.Load();
            foreach (var c in JsonUtility.FromJson<BamCases>(File.ReadAllText(PathFor("bam_cases.json"))).cases)
            {
                report.bam_max_absolute_error = Math.Max(report.bam_max_absolute_error, Math.Abs(bam.Torque(c.target, c.q, c.velocity, c.voltage) - c.torque));
                report.bam_max_absolute_error = Math.Max(report.bam_max_absolute_error, Math.Abs(bam.Friction(c.previous_motor_torque, c.external_torque, c.velocity) - c.friction));
                report.bam_cases++;
            }
            var physics = JsonUtility.FromJson<PhysicsCases>(File.ReadAllText(PathFor("physics_cases.json")));
            var model = MjEngineTool.LoadModelFromFile(PathFor("robot/scene_bam.xml"));
            var data = MujocoLib.mj_makeData(model);
            try
            {
                for (int i = 0; i < (int)model->nq; i++) data->qpos[i] = physics.initial_qpos[i];
                MujocoLib.mj_forward(model, data);
                foreach (var step in physics.steps)
                {
                    for (int i = 0; i < (int)model->nu; i++) data->ctrl[i] = step.control[i];
                    MujocoLib.mj_step(model, data);
                    for (int i = 0; i < (int)model->nq; i++) report.physics_max_absolute_error = Math.Max(report.physics_max_absolute_error, Math.Abs(data->qpos[i] - step.qpos[i]));
                    for (int i = 0; i < (int)model->nv; i++) report.physics_max_absolute_error = Math.Max(report.physics_max_absolute_error, Math.Abs(data->qvel[i] - step.qvel[i]));
                    report.physics_steps++;
                }
            }
            finally { MujocoLib.mj_deleteData(data); MujocoLib.mj_deleteModel(model); }
            report.passed = report.inference_cases == 609 && report.inference_max_absolute_error < 1e-5 && report.bam_max_absolute_error < 1e-10 && report.physics_max_absolute_error < 1e-9;
            Directory.CreateDirectory("Validation"); File.WriteAllText("Validation/unity-validation.json", JsonUtility.ToJson(report, true));
            Debug.Log("MICRODUCK_VALIDATION " + JsonUtility.ToJson(report));
            if (!report.passed) throw new Exception("Policy / BAM / native physics validation failed; see Validation/unity-validation.json.");
        }

        [Serializable] class Scenario
        {
            public string name; public Vector3 command; public int policy_steps; public bool fell;
            public double elapsed_s, distance_m, x_m, y_m, final_tilt_degrees, max_tilt_degrees, yaw_change_radians;
        }
        [Serializable] class ScenarioReport { public Scenario[] scenarios; public string note = "Nominal CPU deployment tests, no training randomization. Performance of the supplied checkpoint, not a new training result."; }
        public static void RunScenarios()
        {
            if (!Application.isPlaying) throw new Exception("Run scenarios in Play mode.");
            var sim = UnityEngine.Object.FindAnyObjectByType<MicroduckSimulation>();
            sim.Paused = true; sim.AutoReset = false; sim.KeyboardControl = false;
            var results = new Scenario[3];
            var commands = new[] { Vector3.zero, new Vector3(.2f, 0, 0), new Vector3(0, 0, .6f) };
            var names = new[] { "idle", "forward_0.2mps", "turn_0.6radps" };
            for (int n = 0; n < results.Length; n++)
            {
                sim.ResetRobot(); sim.VelocityCommand = commands[n];
                var s = new Scenario { name = names[n], command = commands[n] };
                double lastYaw = sim.BaseYawRadians;
                for (int i = 0; i < 2400; i++)
                {
                    sim.StepPhysics(); s.max_tilt_degrees = Math.Max(s.max_tilt_degrees, sim.TiltDegrees);
                    double dy = sim.BaseYawRadians - lastYaw;
                    s.yaw_change_radians += Math.Atan2(Math.Sin(dy), Math.Cos(dy)); lastYaw = sim.BaseYawRadians;
                    if (sim.TiltDegrees > 70 || sim.BasePosition.y < .045f) { s.fell = true; break; }
                }
                var q = sim.GetQpos(); s.x_m = q[0]; s.y_m = q[1]; s.distance_m = Math.Sqrt(q[0] * q[0] + q[1] * q[1]);
                s.elapsed_s = sim.SimulationTime; s.policy_steps = sim.PolicySteps; s.final_tilt_degrees = sim.TiltDegrees; results[n] = s;
            }
            Directory.CreateDirectory("Validation"); File.WriteAllText("Validation/closed-loop-scenarios.json", JsonUtility.ToJson(new ScenarioReport { scenarios = results }, true));
            Debug.Log("MICRODUCK_SCENARIOS " + JsonUtility.ToJson(new ScenarioReport { scenarios = results }));
            sim.AutoReset = true; sim.KeyboardControl = true; sim.VelocityCommand = new Vector3(.2f, 0, 0); sim.ResetRobot(); sim.Paused = false;
        }
    }
}
