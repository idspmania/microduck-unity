"""Reproduce the nominal training model. Run with mujoco==3.10.0, onnxruntime, onnx, numpy.
The source handoff remains read-only. No torch or CUDA is needed.
"""
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT.parent / 'Microduck_Mac_Unity_model_1500'
ASSETS = ROOT / 'Assets/StreamingAssets/Microduck'
contract = json.loads((ASSETS / 'policy_contract.json').read_text())
env = json.loads((BUNDLE / 'training/env.portable.json').read_text())
bam = json.loads((ASSETS / 'bam_m6.json').read_text())

def dump(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')

spec = mujoco.MjSpec.from_file(str(ASSETS / 'robot/scene_walk.xml'))
options = env['sim']['mujoco']
spec.option.timestep = options['timestep']
spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
spec.option.iterations = options['iterations']
spec.option.ls_iterations = options['ls_iterations']
spec.option.tolerance = options['tolerance']
spec.option.ls_tolerance = options['ls_tolerance']
spec.option.ccd_iterations = options['ccd_iterations']
spec.option.impratio = options['impratio']
spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
spec.option.jacobian = mujoco.mjtJacobian.mjJAC_AUTO
spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
spec.option.gravity = options['gravity']
spec.njmax = 1500
spec.nconmax = 35
for g in spec.geoms:
    # CollisionCfg(disable_other_geoms=True), exactly as the walking recipe.
    if g.name == 'floor':
        g.contype = g.conaffinity = 1
    elif g.name.endswith('_collision'):
        g.contype = g.conaffinity = 1
        g.condim = 3 if g.name in ('left_foot_collision', 'right_foot_collision') else 1
        if g.condim == 3:
            g.priority = 1
            g.friction = [1., .005, .0001]
    else:
        g.contype = g.conaffinity = 0
for a in spec.actuators:
    a.set_to_motor()
    a.ctrllimited = False
    a.forcelimited = True
    a.forcerange = [-8.2 * bam['kt'] / bam['R'], 8.2 * bam['kt'] / bam['R']]
    a.gear = [1, 0, 0, 0, 0, 0]
for j in spec.joints:
    if j.name in contract['joint_names']:
        j.armature = bam['armature']
        j.damping = np.zeros((3, 1))
        j.frictionloss = 0
        j.solref_friction = [-5e4, -2e2]
        j.solimp_friction = [.99, .9999, .001, .5, 2]
spec.compile()
text = spec.to_xml()
# XML is beside the original assets, so the normalized mesh paths stay portable.
(ASSETS / 'robot/scene_bam.xml').write_text(text)

policy = ROOT / 'Assets/Microduck/Models/microduck_walk_1024_model_1500.onnx'
model = onnx.load(policy)
onnx.checker.check_model(model)
session = ort.InferenceSession(str(policy), providers=['CPUExecutionProvider'])
cases = json.loads((ASSETS / 'reference_cases.json').read_text())['cases']
trace = np.load(BUNDLE / 'rollouts/model_1500_evaluation/observations_actions.npz')
for i, (obs, action) in enumerate(zip(trace['observation'], trace['action'])):
    cases.append({'name': f'rollout_{i:03}', 'observation': obs.tolist(), 'action': action.tolist()})
dump(ASSETS / 'inference_cases.json', {'cases': cases})
max_error = max(float(np.max(np.abs(session.run(None, {'obs': np.array(c['observation'], np.float32)[None]})[0][0] - c['action']))) for c in cases)

def motor(target, q, vel, voltage):
    gain = 4096 / (2 * np.pi * 256 * 885)
    pwm = np.clip((target - q) * 200 * gain, -1, 1)
    return bam['kt'] * voltage * pwm / bam['R'] - bam['kt'] ** 2 * vel / bam['R']

def friction(motor_torque, external, vel):
    # Training bam.mjlab._compute_friction_budget. Its M6 quadratic term is
    # deliberately used, including its behavior when load signs agree.
    s = np.exp(-(np.abs(vel / bam['dtheta_stribeck']) ** bam['alpha']))
    value = bam['friction_base'] + s * bam['friction_stribeck']
    value += np.abs(external * bam['load_friction_external'] - motor_torque * bam['load_friction_motor'])
    value += s * np.abs(external * bam['load_friction_external_stribeck'] - motor_torque * bam['load_friction_motor_stribeck'])
    quad = np.where(np.abs(motor_torque) > np.abs(external), bam['load_friction_external_quad'] * external ** 2, bam['load_friction_motor_quad'] * motor_torque ** 2)
    return value + s * quad

rng = np.random.default_rng(42)
bcases = []
for i in range(128):
    q, target, vel, prev, external = rng.uniform(-1.5, 1.5), rng.uniform(-2, 2), rng.uniform(-12, 12), rng.uniform(-1.2, 1.2), rng.uniform(-1.5, 1.5)
    voltage = rng.uniform(6., 8.2)
    bcases.append(dict(q=q, target=target, velocity=vel, previous_motor_torque=prev, external_torque=external,
                       voltage=voltage, torque=float(motor(target, q, vel, voltage)), friction=float(friction(prev, external, vel))))
dump(ASSETS / 'bam_cases.json', {'cases': bcases})

native = mujoco.MjModel.from_xml_path(str(ASSETS / 'robot/scene_bam.xml'))
data = mujoco.MjData(native)
ids = [native.joint(n).id for n in contract['joint_names']]
qids, vids = native.jnt_qposadr[ids], native.jnt_dofadr[ids]
home = np.array(contract['home_rad'])
data.qpos[qids] = home
data.qpos[2] = .12
mujoco.mj_forward(native, data)
initial = data.qpos.copy()
steps = []
for i in range(12):
    ctrl = (.1 * np.sin(np.arange(14) + i * .2)).tolist()
    data.ctrl[:] = ctrl
    mujoco.mj_step(native, data)
    steps.append(dict(control=ctrl, qpos=data.qpos.tolist(), qvel=data.qvel.tolist()))
dump(ASSETS / 'physics_cases.json', {'initial_qpos': initial.tolist(), 'steps': steps})

report = dict(bundle_policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
              mujoco_version=mujoco.__version__, ort_version=ort.__version__,
              onnx_opset=model.opset_import[0].version,
              initializer_shapes={x.name: list(x.dims) for x in model.graph.initializer},
              input_shape=[1, 61], output_shape=[1, 14], actor_mlp_parameters=197774,
              static_and_recorded_cases=len(cases), mac_ort_max_abs_error=max_error,
              native_nq=native.nq, native_nv=native.nv, native_nu=native.nu,
              native_body_count=native.nbody, native_mesh_count=native.nmesh,
              bam_commit='62bd8ce12154340be97e06f7f41a0ca8f116d967',
              physics_model='scene_bam.xml', nominal_voltage=7.4, voltage_drop_gain=.1,
              differences_from_training=['Single nominal environment; no sensor noise, encoder bias, pushes, mass/CoM DR.',
                 'Deterministic four physics-step actuator delay within training range 3..6.',
                 'Native CPU MuJoCo instead of MuJoCo Warp; no PPO at runtime.'])
dump(ROOT / 'Validation/source-analysis.json', report)
print(json.dumps(report, indent=2))
