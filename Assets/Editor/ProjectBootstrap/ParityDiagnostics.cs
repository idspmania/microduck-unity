using System;
using System.IO;
using UnityEngine;

namespace Microduck.Editor
{
    public static class ParityDiagnostics
    {
        // Run in Play mode via Pipeline. Manual stepping is independent of Editor frame rate.
        public static string Run()
        {
            if (!Application.isPlaying) throw new InvalidOperationException("Enter Play mode first.");
            var sim = UnityEngine.Object.FindAnyObjectByType<MicroduckSimulation>();
            bool paused = sim.Paused, auto = sim.AutoReset, keyboard = sim.KeyboardControl;
            var command = sim.VelocityCommand;
            var head = (float[])sim.HeadCommand.Clone(); var body = (float[])sim.BodyCommand.Clone();
            float voltage = sim.SupplyVoltage, drop = sim.VoltageDropGain;
            int delay = sim.ActuatorDelaySteps; bool velDelay = sim.DelayJointVelocity;
            Directory.CreateDirectory("Validation/Parity");
            try
            {
                sim.Paused = true; sim.AutoReset = false; sim.KeyboardControl = false;
                sim.SupplyVoltage = 7.4f; sim.VoltageDropGain = .1f; sim.ActuatorDelaySteps = 4; sim.DelayJointVelocity = true;
                Array.Clear(sim.HeadCommand, 0, 4); Array.Clear(sim.BodyCommand, 0, 6);
                var names = new[] { "idle", "forward", "turn_left", "turn_right" };
                var commands = new[] { Vector3.zero, new Vector3(.2f, 0, 0), new Vector3(0, 0, .6f), new Vector3(0, 0, -.6f) };
                for (int n = 0; n < names.Length; n++)
                {
                    sim.ResetRobot(); sim.VelocityCommand = commands[n];
                    using var output = new StreamWriter("Validation/Parity/unity_" + names[n] + ".jsonl");
                    output.WriteLine(JsonUtility.ToJson(sim.GetDiagnosticState()));
                    for (int i = 0; i < 2400; i++) { sim.StepPhysics(); output.WriteLine(JsonUtility.ToJson(sim.GetDiagnosticState())); }
                }
                return "Recorded 4 x 2400 physics steps in Validation/Parity/unity_*.jsonl";
            }
            finally
            {
                sim.SupplyVoltage = voltage; sim.VoltageDropGain = drop; sim.ActuatorDelaySteps = delay; sim.DelayJointVelocity = velDelay;
                sim.HeadCommand = head; sim.BodyCommand = body; sim.VelocityCommand = command;
                sim.ResetRobot(); sim.AutoReset = auto; sim.KeyboardControl = keyboard; sim.Paused = paused;
            }
        }
    }
}
