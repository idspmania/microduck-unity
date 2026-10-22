using System;
using System.IO;
using System.Diagnostics;
using Mujoco;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.InputSystem;

namespace Microduck
{
    [Serializable] public class PhysicsDiagnosticState
    {
        public double time, voltage, yaw;
        public double[] qpos, qvel, ctrl, friction, external;
        public float[] observation, action;
    }
    [Serializable] public class BodyVisual { public string bodyName; public Transform transform; [NonSerialized] public int id; }

    // Single owner of physics. The original MJCF is loaded directly through MuJoCo's
    // official Unity bindings. The importer supplies visuals, not a second MjScene.
    public unsafe class MicroduckSimulation : MonoBehaviour
    {
        public ModelAsset PolicyAsset;
        public BodyVisual[] Bodies;
        [Header("Command in MuJoCo coordinates (metres / radians)")]
        public Vector3 VelocityCommand = new Vector3(.2f, 0, 0);
        public float[] HeadCommand = new float[4];
        public float[] BodyCommand = new float[6];
        [Header("Nominal deployment parameters")]
        [Range(6.5f, 8.2f)] public float SupplyVoltage = 7.4f;
        [Range(0, .2f)] public float VoltageDropGain = .1f;
        [Range(0, 6)] public int ActuatorDelaySteps = 4;
        public bool DelayJointVelocity = true;
        public bool AutoReset = true;
        public bool Paused;
        public bool KeyboardControl = true;
        public double SimulationTime => data == null ? 0 : data->time;
        public int PolicySteps { get; private set; }
        public int Falls { get; private set; }
        public double LastInferenceMilliseconds { get; private set; }
        public double EffectiveVoltage { get; private set; }
        public float TiltDegrees { get; private set; }
        public Vector3 BasePosition { get; private set; }
        public double BaseYawRadians { get; private set; }
        public bool Ready => model != null && data != null;
        public float[] LastObservation { get; private set; } = new float[61];
        public float[] LastAction { get; private set; } = new float[14];
        public double HorizontalTravel => Math.Sqrt(data->qpos[0] * data->qpos[0] + data->qpos[1] * data->qpos[1]);
        MujocoLib.mjModel_* model;
        MujocoLib.mjData_* data;
        Policy policy;
        PolicyContract contract;
        BamParameters bam;
        int[] qids = new int[14], vids = new int[14], aids = new int[14];
        int trunkId, imuVelocityAddress, physicsSteps;
        double[] previousMotor = new double[14], delayedVelocity = new double[14];
        double[,] targets = new double[16, 14];
        int delayCursor;
        readonly Stopwatch watch = new Stopwatch();
        float previousFixedDelta;

        void Awake()
        {
            Application.runInBackground = true;
            contract = PolicyContract.Load(); bam = BamParameters.Load();
            previousFixedDelta = Time.fixedDeltaTime;
            Time.fixedDeltaTime = contract.physics_dt_s;
            try
            {
                string xml = Path.Combine(Application.streamingAssetsPath, "Microduck/robot/scene_bam.xml");
                model = MjEngineTool.LoadModelFromFile(xml);
                if (MujocoLib.mj_version() != 3010000) throw new InvalidOperationException("MuJoCo 3.10.0 is required by these bindings.");
                data = MujocoLib.mj_makeData(model);
                if (data == null) throw new InvalidOperationException("mj_makeData failed.");
                trunkId = RequiredId(MujocoLib.mjtObj.mjOBJ_BODY, "trunk_base");
                int sensor = RequiredId(MujocoLib.mjtObj.mjOBJ_SENSOR, "imu_ang_vel");
                imuVelocityAddress = model->sensor_adr[sensor];
                for (int i = 0; i < 14; i++)
                {
                    int joint = RequiredId(MujocoLib.mjtObj.mjOBJ_JOINT, contract.joint_names[i]);
                    qids[i] = model->jnt_qposadr[joint]; vids[i] = model->jnt_dofadr[joint];
                    aids[i] = -1;
                    for (int a = 0; a < (int)model->nu; a++) if (model->actuator_trnid[2 * a] == joint) aids[i] = a;
                    if (aids[i] < 0) throw new InvalidOperationException("Missing actuator: " + contract.joint_names[i]);
                }
                foreach (var b in Bodies) b.id = RequiredId(MujocoLib.mjtObj.mjOBJ_BODY, b.bodyName);
                policy = new Policy(PolicyAsset);
                ResetRobot();
                UnityEngine.Debug.Log("MICRODUCK_READY MuJoCo 3.10.0 / Sentis CPU / obs61 / act14 / physics200Hz / policy50Hz");
            }
            catch { Release(); throw; }
        }

        int RequiredId(MujocoLib.mjtObj type, string name)
        {
            int id = MujocoLib.mj_name2id(model, (int)type, name);
            if (id < 0) throw new InvalidOperationException("Missing MJCF name: " + name);
            return id;
        }

        public void ResetRobot()
        {
            if (!Ready) return;
            MujocoLib.mj_resetData(model, data);
            data->qpos[2] = .12;
            for (int i = 0; i < 14; i++)
            {
                data->qpos[qids[i]] = contract.home_rad[i];
                previousMotor[i] = delayedVelocity[i] = 0;
                model->dof_frictionloss[vids[i]] = 0;
                model->dof_damping[vids[i]] = 0;
                for (int t = 0; t < 16; t++) targets[t, i] = contract.home_rad[i];
            }
            Array.Clear(LastAction, 0, 14); physicsSteps = delayCursor = 0; PolicySteps = 0;
            MujocoLib.mj_forward(model, data); SyncVisuals();
        }

        void Update()
        {
            if (!KeyboardControl || Keyboard.current == null) return;
            var k = Keyboard.current;
            if (k.rKey.wasPressedThisFrame) ResetRobot();
            if (k.spaceKey.wasPressedThisFrame) { VelocityCommand = Vector3.zero; Paused = false; }
            if (k.pKey.wasPressedThisFrame) Paused = !Paused;
            if (k.wKey.isPressed || k.sKey.isPressed || k.aKey.isPressed || k.dKey.isPressed || k.qKey.isPressed || k.eKey.isPressed)
                VelocityCommand = new Vector3((k.wKey.isPressed ? .2f : 0) - (k.sKey.isPressed ? .2f : 0),
                    (k.aKey.isPressed ? .15f : 0) - (k.dKey.isPressed ? .15f : 0),
                    (k.qKey.isPressed ? .6f : 0) - (k.eKey.isPressed ? .6f : 0));
        }

        void FixedUpdate()
        {
            if (Ready && !Paused) StepPhysics();
        }

        public void StepPhysics()
        {
            if (physicsSteps % contract.physics_steps_per_policy == 0)
            {
                CollectObservations();
                watch.Restart(); LastAction = policy.Evaluate(LastObservation); watch.Stop();
                LastInferenceMilliseconds = watch.Elapsed.TotalMilliseconds; PolicySteps++;
            }
            EffectiveVoltage = Math.Max(6, SupplyVoltage - VoltageDropGain * SumAbsolute(previousMotor));
            // Read previous solver friction force before updating any of the budgets.
            var frictionForce = new double[14];
            for (int e = 0; e < data->nefc; e++)
                if (data->efc_type[e] == (int)MujocoLib.mjtConstraint.mjCNSTR_FRICTION_DOF)
                    for (int i = 0; i < 14; i++)
                        if (data->efc_id[e] == vids[i]) frictionForce[i] += data->efc_force[e];
            for (int i = 0; i < 14; i++)
            {
                targets[delayCursor, i] = contract.home_rad[i] + LastAction[i] * contract.action_scale;
                int at = (delayCursor - Math.Clamp(ActuatorDelaySteps, 0, 6) + 16) % 16;
                int v = vids[i];
                double velocity = data->qvel[v];
                double external = -data->qfrc_bias[v] + data->qfrc_constraint[v] - frictionForce[i];
                model->dof_frictionloss[v] = bam.Friction(data->qfrc_actuator[v], external, velocity);
                model->dof_damping[v] = bam.friction_viscous;
                double torque = bam.Torque(targets[at, i], data->qpos[qids[i]], velocity, EffectiveVoltage);
                previousMotor[i] = torque; data->ctrl[aids[i]] = torque;
            }
            delayCursor = (delayCursor + 1) % 16;
            MujocoLib.mj_step(model, data); physicsSteps++;
            for (int i = 0; i < (int)model->nq; i++)
                if (double.IsNaN(data->qpos[i]) || double.IsInfinity(data->qpos[i])) throw new InvalidOperationException("Non-finite MuJoCo state.");
            SyncVisuals();
            if (AutoReset && (TiltDegrees > 70 || BasePosition.y < .045f)) { Falls++; ResetRobot(); }
        }

        void CollectObservations()
        {
            for (int i = 0; i < 3; i++) LastObservation[i] = (float)data->sensordata[imuVelocityAddress + i];
            // R^T * [0,0,-1] from the trunk rotation, entirely in MuJoCo coordinates.
            for (int i = 0; i < 3; i++) LastObservation[3 + i] = (float)-data->xmat[9 * trunkId + 6 + i];
            for (int i = 0; i < 14; i++)
            {
                LastObservation[6 + i] = (float)(data->qpos[qids[i]] - contract.home_rad[i]);
                LastObservation[20 + i] = (float)(DelayJointVelocity ? delayedVelocity[i] : data->qvel[vids[i]]);
                delayedVelocity[i] = data->qvel[vids[i]];
                LastObservation[34 + i] = LastAction[i];
            }
            LastObservation[48] = Mathf.Clamp(VelocityCommand.x, -.4f, .4f);
            LastObservation[49] = Mathf.Clamp(VelocityCommand.y, -.3f, .3f);
            LastObservation[50] = Mathf.Clamp(VelocityCommand.z, -1, 1);
            for (int i = 0; i < 4; i++) LastObservation[51 + i] = HeadCommand[i];
            for (int i = 0; i < 6; i++) LastObservation[55 + i] = BodyCommand[i];
        }

        void SyncVisuals()
        {
            foreach (var b in Bodies)
            {
                if (!b.transform) continue;
                b.transform.SetPositionAndRotation(MjEngineTool.UnityVector3(data->xpos + 3 * b.id),
                    MjEngineTool.UnityQuaternion(data->xquat + 4 * b.id));
            }
            BasePosition = MjEngineTool.UnityVector3(data->xpos + 3 * trunkId);
            double up = data->xmat[9 * trunkId + 8];
            TiltDegrees = (float)(Math.Acos(Math.Max(-1, Math.Min(1, up))) * 180 / Math.PI);
            BaseYawRadians = Math.Atan2(data->xmat[9 * trunkId + 3], data->xmat[9 * trunkId]);
        }
        static double SumAbsolute(double[] values) { double sum = 0; foreach (double x in values) sum += Math.Abs(x); return sum; }
        public double[] GetQpos() { var q = new double[model->nq]; for (int i = 0; i < q.Length; i++) q[i] = data->qpos[i]; return q; }
        // Read-only telemetry for the matched Python/Unity comparison harness.
        public PhysicsDiagnosticState GetDiagnosticState()
        {
            var s = new PhysicsDiagnosticState { time = data->time, voltage = EffectiveVoltage, yaw = BaseYawRadians,
                qpos = GetQpos(), qvel = new double[model->nv], ctrl = new double[14], friction = new double[14], external = new double[14],
                observation = (float[])LastObservation.Clone(), action = (float[])LastAction.Clone() };
            for (int i = 0; i < s.qvel.Length; i++) s.qvel[i] = data->qvel[i];
            for (int i = 0; i < 14; i++)
            {
                int v = vids[i]; s.ctrl[i] = data->ctrl[aids[i]]; s.friction[i] = model->dof_frictionloss[v];
                s.external[i] = -data->qfrc_bias[v] + data->qfrc_constraint[v];
                for (int e = 0; e < data->nefc; e++)
                    if (data->efc_type[e] == (int)MujocoLib.mjtConstraint.mjCNSTR_FRICTION_DOF && data->efc_id[e] == v)
                        s.external[i] -= data->efc_force[e];
            }
            return s;
        }
        void OnDestroy() { Release(); Time.fixedDeltaTime = previousFixedDelta; }
        void Release()
        {
            policy?.Dispose(); policy = null;
            if (data != null) { MujocoLib.mj_deleteData(data); data = null; }
            if (model != null) { MujocoLib.mj_deleteModel(model); model = null; }
        }
    }
}
