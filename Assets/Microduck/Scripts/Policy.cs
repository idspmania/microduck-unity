using System;
using System.IO;
using Unity.InferenceEngine;
using UnityEngine;

namespace Microduck
{
    [Serializable] public class PolicyContract
    {
        public string policy_sha256;
        public string[] joint_names;
        public double[] home_rad;
        public float physics_dt_s, policy_dt_s, action_scale;
        public int physics_steps_per_policy;
        public static PolicyContract Load() => JsonUtility.FromJson<PolicyContract>(File.ReadAllText(Path.Combine(Application.streamingAssetsPath, "Microduck/policy_contract.json")));
    }

    // The ONNX already contains EmpiricalNormalization. Feed physical observations once.
    public sealed class Policy : IDisposable
    {
        readonly Worker worker;
        public Policy(ModelAsset asset) { worker = new Worker(ModelLoader.Load(asset), BackendType.CPU); }
        public float[] Evaluate(float[] observation)
        {
            if (observation.Length != 61) throw new ArgumentException("Microduck expects 61 observations.");
            using var input = new Tensor<float>(new TensorShape(1, 61), observation);
            worker.Schedule(input);
            var output = worker.PeekOutput("actions") as Tensor<float>;
            var action = output.DownloadToArray();
            if (action.Length != 14) throw new InvalidOperationException("Microduck expects 14 actions.");
            foreach (float x in action) if (float.IsNaN(x) || float.IsInfinity(x)) throw new InvalidOperationException("Non-finite policy output.");
            return action;
        }
        public void Dispose() => worker.Dispose();
    }

    // Port of BAM at 62bd8ce, specifically the mjlab M6 budget used for training.
    // The CPU bam.model quadratic gate differs; use the training implementation here.
    [Serializable] public class BamParameters
    {
        public double kt, R, armature, friction_base, friction_stribeck;
        public double load_friction_motor, load_friction_external;
        public double load_friction_motor_stribeck, load_friction_external_stribeck;
        public double load_friction_motor_quad, load_friction_external_quad;
        public double dtheta_stribeck, alpha, friction_viscous;
        public static BamParameters Load() => JsonUtility.FromJson<BamParameters>(File.ReadAllText(Path.Combine(Application.streamingAssetsPath, "Microduck/bam_m6.json")));
        public double Torque(double target, double q, double velocity, double voltage, double kp = 200)
        {
            const double errorGain = 4096.0 / (2 * Math.PI * 256 * 885);
            double duty = Math.Max(-1, Math.Min(1, (target - q) * kp * errorGain));
            // Training has no firmware current limiter.
            return kt * voltage * duty / R - kt * kt * velocity / R;
        }
        public double Friction(double previousMotorTorque, double externalTorque, double velocity)
        {
            double s = Math.Exp(-Math.Pow(Math.Abs(velocity / dtheta_stribeck), alpha));
            double f = friction_base + s * friction_stribeck;
            f += Math.Abs(externalTorque * load_friction_external - previousMotorTorque * load_friction_motor);
            f += s * Math.Abs(externalTorque * load_friction_external_stribeck - previousMotorTorque * load_friction_motor_stribeck);
            f += s * (Math.Abs(previousMotorTorque) > Math.Abs(externalTorque)
                ? load_friction_external_quad * externalTorque * externalTorque
                : load_friction_motor_quad * previousMotorTorque * previousMotorTorque);
            return f;
        }
    }
}
