# model_1500 자료 분석 및 Unity 이식 기록

분석일: 2026-10-22, Asia/Seoul. 분석 기준은 전달받은 자료이며 최신 upstream으로 임의 갱신하지 않았습니다.

## 자료 무결성과 출처

`VERIFY_FILES.py`로 매니페스트의 298개 파일이 모두 일치했습니다. 원본 폴더는 보존했습니다. `training/provenance.json`, 버전 목록, 실제 argv, portable 설정, 소스, reference/rollout 기록을 함께 확인했습니다. 46개 체크포인트가 포함된 로그는 두 실행의 스냅샷이며 최종 학습 완료 로그가 아닙니다. 이 프로젝트는 루트의 model_1500 ONNX를 사용하며, 로그 폴더에 있는 더 최신 자동 ONNX를 대신 사용하지 않습니다.

- 과제: `Mjlab-Velocity-Flat-MicroDuck`.
- WSL2 Ubuntu 24.04, RTX 3070 Laptop 8GB.
- 최초 512환경 실행 후 model_950에서 1024환경으로 재개.
- Python 3.12, mjlab 1.3.0, MuJoCo 3.10.0, mujoco-warp 3.8.1, Warp 1.12.0, PyTorch 2.9.1, RSL-RL 5.0.1.
- source commit: `b02fdea4902c2b93b7d34a83a6f05da4ececb7d3`, 전달 기록상 clean.
- BAM 1.0.1 commit: `62bd8ce12154340be97e06f7f41a0ca8f116d967`.
- ONNX SHA-256: `a9eac6302096630fc6f6c1d64f5a6d62e207895fea661786b8923bad6ed2b1c4`.
- ONNX와 체크포인트 가중치 일치 여부는 동봉된 검증 기록을 참조했습니다. 이 Mac 작업에서는 ONNX 구조·해시·609개 출력·물리·BAM을 직접 재검증했습니다. PyTorch 체크포인트를 다시 학습하거나 재변환하지 않았습니다.

## 정책과 PPO

Actor는 `61 → 512 → 256 → 128 → 14`, ELU, 마지막 Linear입니다. ONNX 초기화 텐서의 shape로 직접 확인했습니다. MLP weight+bias는 197,774개이며 관측 정규화의 평균과 분모도 그래프에 포함됩니다. 입력 `obs [1,61]`, 출력 `actions [1,14]`, float32, opset18입니다. Unity에서 정규화를 다시 적용하거나 tanh·EMA를 추가하지 않습니다.

Critic은 학습 전용이며 actor와 별도 512→256→128 MLP입니다. 기본 walking 구성에서 privileged base velocity와 발 높이·접촉·힘 등이 추가되어 76D 입력으로 산출되며 MLP 파라미터는 203,777개입니다. 학습 Actor의 Gaussian std 파라미터 14개는 배포용 deterministic ONNX의 일부가 아닙니다.

실제 저장 agent 설정은 Adam, learning rate 0.001/adaptive, gamma0.99, GAE lambda0.95, PPO clip0.2, entropy0.01, value loss1, epochs5, mini-batches4, desired KL0.01, grad norm1, rollout24입니다. 1024환경에서는 rollout 24,576개, mini-batch 6,144개입니다. 재개 설정 max_iterations5050은 그 실행의 추가 반복 횟수이며 model_1500과 혼동하지 않습니다.

보행 보상은 속도 추종(선형·각속도 각각2), upright2, HOME 주변 leg pose1, air_time3, head_pose_tracking2, 관절 제한·발 미끄러짐·발 높이·자기 충돌·행동 변화 등의 비용으로 구성됩니다. body_pose_tracking 가중치는 기본 walking에서0입니다. action_rate 비용은 -0.1에서 -1로, head_pose_bias 비용은0에서3으로 커리큘럼이 적용됩니다. 이 값들은 별도 재학습을 하지 않는 Unity 추론에서 계산할 필요가 없습니다. 설정 시작값과 체크포인트의 curriculum 진행 상태를 구분합니다.

## 관측과 행동 규약

| 인덱스 | 내용 |
|---|---|
| 0–2 | IMU/body 좌표 각속도 rad/s |
| 3–5 | 몸통 좌표로 투영한 단위 중력 |
| 6–19 | joint angle − 정확한 HOME, rad |
| 20–33 | joint velocity, rad/s |
| 34–47 | 이전 raw action |
| 48–50 | vx, vy, wz 명령 |
| 51–54 | neck/head 명령 4개 |
| 55–60 | body pose 명령 6개 |

관절은 왼쪽 다리5개 → 목/머리4개 → 오른쪽 다리5개 순서입니다. `policy_contract.json`의 정확한 HOME 소수 값을 사용합니다. ONNX metadata의 HOME은 반올림되어 있어 재사용하지 않습니다. raw qpos address를 하드코딩하지 않고 joint 이름을 native 모델에서 조회합니다. Action은 각도 자체나 토크가 아니라 `HOME + 1.0 × raw_action`의 목표 각도 오프셋입니다.

MuJoCo는 Z-up/right-handed, Unity는 Y-up/left-handed입니다. 물리 관측과 정책은 전부 MuJoCo 좌표에서 계산하고, 렌더링할 때만 공식 `MjEngineTool`의 좌표 변환을 사용합니다. 이를 통해 IMU 각속도를 Unity Euler angle 차분으로 계산하는 오류를 피합니다.

## 물리와 BAM 이식

원본 robot_walk.xml, scene_walk.xml, 38개 메시를 확인했습니다. 로봇은 nq21/nv20/actuator14/body16(world 포함)입니다. physics dt0.005, decimation4, policy dt0.02, implicitfast, solver10회, line-search20회, CCD50회입니다.

`prepare_assets.py`는 원본 MJCF를 읽고 walking CollisionCfg, motor actuator, BAM armature, stiff friction constraint를 반영한 별도 `scene_bam.xml`을 만듭니다. 기본 position actuator XML만 사용하지 않습니다.

BAM은 duty cycle = clamp(position_error × firmware KP200 × error_gain, -1,1), DC motor torque = kt×V/R − kt²×velocity/R, 배터리 sag, Coulomb/Stribeck/부하 마찰을 계산합니다. frictionloss와 damping은 MuJoCo native constraint solver에 전달합니다. 학습은 firmware current limiter를 사용하지 않았으므로 임의로1.75A 제한을 추가하지 않습니다.

중요한 소스 차이: 같은 BAM 커밋에서도 `bam.model.compute_frictions`의 M6 quadratic 항은 부하 부호 게이트가 있고, 학습 경로 `bam.mjlab._compute_friction_budget`에는 그 게이트가 없습니다. 이 프로젝트는 **실제 학습 경로**를 이식했습니다. q_offset은 모델 식별용 testbench 오프셋으로, 런타임 관절에 임의 적용하지 않습니다.

배포 nominal 값은7.4V, sag gain0.1, 최소6V, actuator delay4 physics steps(20ms), joint velocity delay1 policy step(20ms)입니다. 학습 DR의 actuator delay 범위는3~6 physics steps입니다. IMU noise·bias·질량/관성·CoM 무작위화 및 외란은 nominal 재현에서 껐습니다. 이 차이는 성능 시험 조건에 기록했습니다.

## Unity 통합과 호환성

공식 MuJoCo Unity package와 native library를 같은3.10.0으로 고정했습니다. Unity6000.6.4f1은 GetInstanceID를 제거했으므로 로컬 패키지 `MjMeshFilter.cs`의 디버그 mesh 이름 한 곳에서 호출을 제거했습니다. 물리 바인딩은 수정하지 않았습니다.

공식 MJCF 임포터로 메시·재질·계층을 생성한 뒤 실행용 계층에서는 MjMeshFilter → MjComponent 순서로 제거합니다. MjMeshFilter가 MjGeom을 요구하므로 반대 순서는 제거 오류를 일으키고, 남은 MjGeom은 MjScene을 다시 만들어 시각 좌표를 덮어씁니다. 최종 씬은 native physics owner 하나와 body Transform 매핑만 사용합니다.

Sentis CPU backend를 선택했습니다. 197k 파라미터 정책 한 개의 추론은 HUD에서 실측 시간을 표시하며, Mac에서 CUDA를 구성할 필요가 없습니다. CPU backend와 GPU backend의 전송·추론 비용 비교가 필요해지면 측정 후 변경합니다.

## 검증 범위

`Validation/unity-validation.json`: 609개 관측의 최대 출력 오차 약1.67e-6, BAM128개 최대오차 약2.22e-16, Python/Unity native physics12스텝 최대오차 약2.11e-15로 통과했습니다. 관측 정규화가 포함된 실제 ONNX, 실제 Sentis와 실제 native MuJoCo를 사용했습니다.

`Validation/closed-loop-scenarios.json`: 정지·전진·회전 명령별12초, 자동 리셋을 끈 상태로 넘어짐·이동량·자세를 측정했습니다. 단일 nominal 조건의 진단이며 모든 random seed·외란·지형에 대한 보행 성공 보장은 아닙니다. model_1500의 정지 drift와 속도 추종 오차는 남아 있습니다. 회전 +0.6rad/s의 12초 명령에서 누적 yaw는 -0.231rad로, 목표 +7.2rad를 추종하지 못했습니다. 넘어지지 않는 것과 명령을 제대로 수행하는 것을 구분합니다.

원본 rollout600스텝은 WSL GPU MuJoCo Warp에서 새로 평가한 기록입니다. 이 데이터를 Sentis 입력 검증에 사용했지만, 센서 noise·DR·외란이 다른 native CPU 폐루프 궤적과 프레임별 일치를 주장하지 않습니다. Unity에서 실행 중 프레임 증가와 화면도 확인했습니다.
