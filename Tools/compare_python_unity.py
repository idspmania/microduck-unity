"""Matched CPU MuJoCo/ONNX comparison; does not require CUDA or PyTorch.

Run with Python containing mujoco==3.10.0, onnxruntime and numpy:
  python Tools/compare_python_unity.py
Unity traces: in Play mode run Microduck.Editor.ParityDiagnostics.Run().
Raw trajectories are generated locally and ignored by Git.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'Assets/StreamingAssets/Microduck'
OUT = ROOT / 'Validation/Parity'
CONTRACT = json.loads((ASSETS / 'policy_contract.json').read_text())
BAM = json.loads((ASSETS / 'bam_m6.json').read_text())
COMMANDS = {'idle': (0, 0, 0), 'forward': (.2, 0, 0),
            'turn_left': (0, 0, .6), 'turn_right': (0, 0, -.6)}


def motor(target, q, velocity, voltage, current_limit=False):
    gain = 4096 / (2 * math.pi * 256 * 885)
    duty = (target - q) * 200 * gain
    if current_limit:
        center = BAM['kt'] * velocity / voltage
        span = BAM['R'] * 1.75 / voltage
        duty = max(center - span, min(center + span, duty))
    duty = max(-1, min(1, duty))
    return BAM['kt'] * voltage * duty / BAM['R'] - BAM['kt'] ** 2 * velocity / BAM['R']


def friction(previous, external, velocity, cpu_gate=False):
    s = math.exp(-abs(velocity / BAM['dtheta_stribeck']) ** BAM['alpha'])
    f = BAM['friction_base'] + s * BAM['friction_stribeck']
    f += abs(external * BAM['load_friction_external'] - previous * BAM['load_friction_motor'])
    f += s * abs(external * BAM['load_friction_external_stribeck'] - previous * BAM['load_friction_motor_stribeck'])
    quad = (BAM['load_friction_external_quad'] * external ** 2 if abs(previous) > abs(external)
            else BAM['load_friction_motor_quad'] * previous ** 2)
    if cpu_gate and (np.sign(external) == np.sign(previous) or abs(previous) == abs(external)):
        quad = 0
    return f + s * quad


def load_trace(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return {k: np.asarray([r[k] for r in rows]) for k in rows[0]}


class Runner:
    def __init__(self, *, forward=False, current_limit=False, cpu_gate=False,
                 delay=4, velocity_delay=True, voltage=7.4, drop=.1, model_path=None):
        self.forward, self.current_limit, self.cpu_gate = forward, current_limit, cpu_gate
        self.delay, self.velocity_delay = delay, velocity_delay
        # Unity Inspector fields are float32, MuJoCo/BAM state is float64.
        self.voltage, self.drop = float(np.float32(voltage)), float(np.float32(drop))
        self.m = mujoco.MjModel.from_xml_path(str(model_path or ASSETS / 'robot/scene_bam.xml'))
        self.d = mujoco.MjData(self.m)
        ids = [self.m.joint(n).id for n in CONTRACT['joint_names']]
        self.qids, self.vids = self.m.jnt_qposadr[ids], self.m.jnt_dofadr[ids]
        self.aids = [np.flatnonzero(self.m.actuator_trnid[:, 0] == j)[0] for j in ids]
        self.trunk = self.m.body('trunk_base').id
        self.home = np.asarray(CONTRACT['home_rad'])
        options = ort.SessionOptions(); options.intra_op_num_threads = 1
        self.policy = ort.InferenceSession(str(ROOT / 'Assets/Microduck/Models' / CONTRACT['policy']),
                                           sess_options=options, providers=['CPUExecutionProvider'])

    def external(self):
        force = np.zeros(self.m.nv)
        for e in range(self.d.nefc):
            if self.d.efc_type[e] == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF:
                force[self.d.efc_id[e]] += self.d.efc_force[e]
        return (-self.d.qfrc_bias + self.d.qfrc_constraint - force)[self.vids]

    def run(self, name, steps=2400, replay=None):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[2] = .12; d.qpos[self.qids] = self.home
        m.dof_frictionloss[self.vids] = 0; m.dof_damping[self.vids] = 0
        mujoco.mj_forward(m, d)
        previous = np.zeros(14); lagvel = np.zeros(14)
        targets = np.tile(self.home, (16, 1)); cursor = 0
        action = np.zeros(14, np.float32); obs = np.zeros(61, np.float32)
        rows = []
        voltage = 0.

        def record():
            rows.append(dict(time=float(d.time), voltage=voltage,
                             yaw=math.atan2(d.xmat[self.trunk, 3], d.xmat[self.trunk, 0]),
                             qpos=d.qpos.copy(), qvel=d.qvel.copy(), ctrl=d.ctrl[self.aids].copy(),
                             friction=m.dof_frictionloss[self.vids].copy(), external=self.external(),
                             observation=obs.copy(), action=action.copy()))

        record()
        for step in range(steps):
            if step % 4 == 0:
                if self.forward:
                    mujoco.mj_forward(m, d)
                obs[:3] = d.sensor('imu_ang_vel').data
                obs[3:6] = -d.xmat[self.trunk, 6:9]
                obs[6:20] = d.qpos[self.qids] - self.home
                obs[20:34] = lagvel if self.velocity_delay else d.qvel[self.vids]
                lagvel = d.qvel[self.vids].copy()
                obs[34:48] = action
                obs[48:51] = COMMANDS[name]
                action = (self.policy.run(None, {'obs': obs[None]})[0][0] if replay is None
                          else replay[step // 4].astype(np.float32))
            voltage = max(6, self.voltage - self.drop * sum(abs(float(x)) for x in previous))
            external = self.external()
            for i, v in enumerate(self.vids):
                targets[cursor, i] = self.home[i] + float(action[i])
                m.dof_frictionloss[v] = friction(float(d.qfrc_actuator[v]), float(external[i]), float(d.qvel[v]), self.cpu_gate)
                m.dof_damping[v] = BAM['friction_viscous']
                previous[i] = motor(targets[(cursor - self.delay) % 16, i], float(d.qpos[self.qids[i]]), float(d.qvel[v]), voltage, self.current_limit)
                d.ctrl[self.aids[i]] = previous[i]
            cursor = (cursor + 1) % 16
            mujoco.mj_step(m, d)
            record()
        return {k: np.asarray([r[k] for r in rows]) for k in rows[0]}


def metrics(trace, name):
    q = trace['qpos']; duration = float(trace['time'][-1])
    yaw = np.unwrap(trace['yaw']); dyaw = float(yaw[-1] - yaw[0])
    settled = trace['time'] >= 2
    rate = np.diff(yaw) / .005
    return dict(duration_s=duration, displacement_m=float(np.linalg.norm(q[-1, :2] - q[0, :2])),
                x_m=float(q[-1, 0]), y_m=float(q[-1, 1]), yaw_change_rad=dyaw,
                average_yaw_rate_rad_s=dyaw / duration,
                settled_yaw_rate_rad_s=float(np.mean(rate[settled[1:]])),
                commanded_yaw_rate_rad_s=COMMANDS[name][2],
                min_height_m=float(q[:, 2].min()))


def difference(a, b):
    result = {}
    for key in ('qpos', 'qvel', 'ctrl', 'friction', 'external', 'observation', 'action'):
        error = np.max(np.abs(a[key] - b[key]), axis=1)
        result[key] = dict(max_abs=float(error.max()), at_step=int(error.argmax()),
                           first_over_1e_5=int(np.flatnonzero(error > 1e-5)[0]) if np.any(error > 1e-5) else None,
                           first_100_steps_max=float(error[:101].max()))
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--ablation', action='store_true')
    args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    report = dict(mujoco=mujoco.__version__, onnxruntime=ort.__version__,
                  policy_sha256=CONTRACT['policy_sha256'],
                  xml_sha256=hashlib.sha256((ASSETS / 'robot/scene_bam.xml').read_bytes()).hexdigest(),
                  conditions=dict(physics_dt=.005, policy_dt=.02, steps=2400, delay_steps=4,
                                  joint_velocity_delay_policy_steps=1, voltage_float32=7.4,
                                  drop_float32=.1, head=[0]*4, body=[0]*6, randomization=False), scenarios={})
    variants = {'baseline': {}}
    if args.ablation:
        variants.update(forward={'forward': True}, current_limit={'current_limit': True},
                        forward_current_limit={'forward': True, 'current_limit': True},
                        cpu_friction={'cpu_gate': True}, delay0={'delay': 0},
                        instant_velocity={'velocity_delay': False})
    for name in COMMANDS:
        unity = load_trace(OUT / f'unity_{name}.jsonl')
        item = dict(unity=metrics(unity, name), python={})
        for variant, config in variants.items():
            trace = Runner(**config).run(name)
            np.savez_compressed(OUT / f'python_{variant}_{name}.npz', **trace)
            item['python'][variant] = metrics(trace, name)
            if variant == 'baseline':
                item['closed_loop_difference'] = difference(unity, trace)
                policy_obs = unity['observation'][1::4].astype(np.float32)
                outputs = np.asarray([RunnerPolicy.run(None, {'obs': o[None]})[0][0] for o in policy_obs])
                item['same_observation_inference_max_abs'] = float(np.max(np.abs(outputs-unity['action'][1::4])))
        replay = Runner().run(name, replay=unity['action'][1::4])
        np.savez_compressed(OUT / f'python_unity_actions_{name}.npz', **replay)
        item['same_action_replay_difference'] = difference(unity, replay)
        report['scenarios'][name] = item
        print(name, json.dumps(dict(unity=item['unity'], python=item['python'])), flush=True)
    (OUT / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    RunnerPolicy = Runner().policy
    main()
