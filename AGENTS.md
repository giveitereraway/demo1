# AGENTS.md

## 基本协作约束

- 本项目对应硕士毕业设计《基于分层强化学习的飞行器智能自主决策技术》。后续修改、解释和实验记录都应围绕“飞行器自主决策、分层强化学习、空战对抗、JSBSim 物理仿真”这条主线组织。

## 项目总体定位

这是一个基于 JSBSim 飞行动力学引擎的空战强化学习实验框架。项目在 Light Aircraft Game / CloseAirCombat 风格的环境基础上，组织了单机飞行控制、1v1 空战、2v2 空战、导弹攻防、分层强化学习、自博弈训练、人机交互和 Tacview 可视化等模块。

核心目标是让智能体在较真实的飞行动力学约束下学习自主决策：

- 低层控制：学习或调用预训练策略，把目标航向、高度、速度转换为副翼、升降舵、方向舵、油门等执行器控制。
- 高层决策：在空战场景中选择机动方向、高度变化、速度变化和导弹发射时机。
- 对抗训练：支持固定基线对手、自博弈历史策略池、ELO 评价、1v1 和 2v2 多智能体协同/对抗。
- 可视化验证：生成 Tacview `.acmi` 文件，或通过 Tacview Advanced 实时遥测观察训练/评估过程。

## 技术栈与依赖

- 主要语言：Python。
- 深度学习：PyTorch。
- 强化学习接口：代码中大量使用 `gymnasium`，README 中仍提到 `gym==0.20.0`，配置环境时需要以实际导入为准。
- 飞行动力学：`jsbsim==1.1.6`，飞机数据位于 `envs/JSBSim/data` 子模块。
- 地理坐标转换：`pymap3d`。
- 配置解析：YAML，入口在 `envs/JSBSim/utils/utils.py` 的 `parse_config()`。
- 实验记录：`wandb` 可选。
- 进程标题：`setproctitle`。
- 可视化：Tacview / Tacview Advanced。

注意：仓库没有 `requirements.txt` 或 `pyproject.toml`。如果重新搭环境，优先根据实际 import 和 README 同步补全依赖。

## 目录结构速览

- `README.md`：原始项目说明，覆盖安装、任务类型、训练、渲染和 Tacview 实时遥测。
- `config.py`：统一命令行参数入口，定义训练、网络、PPO、自博弈、保存、日志、评估、渲染等参数。
- `scripts/train/train_jsbsim.py`：主训练入口，根据 `--env-name` 创建环境，根据场景选择 PPO / MAPPO runner。
- `scripts/train/train_gym.py`：普通 Gym 示例训练入口，不是本项目空战主线。
- `scripts/*.sh`：常用实验脚本，包括航向控制、自博弈、导弹发射、多机协同等。
- `scripts/render/render_jsbsim.py`：加载模型并生成 Tacview 渲染结果的通用入口。
- `scripts/human_combat/`：人机交互脚本入口。
- `runner/`：训练、评估、渲染、自博弈和 Tacview 实时通信的运行器。
- `algorithms/ppo/`：单智能体/双智能体 PPO 实现。
- `algorithms/mappo/`：集中式 critic 的 MAPPO 实现，主要用于 2v2。
- `algorithms/utils/`：网络层、GRU、动作分布、buffer、自博弈采样等通用工具。
- `envs/env_wrappers.py`：向量化环境包装，支持单进程和子进程 rollout。
- `envs/JSBSim/core/`：JSBSim 飞机仿真器、导弹仿真器、Tacview 渲染和属性 catalog。
- `envs/JSBSim/envs/`：Gymnasium 环境封装，包含单机、1v1、2v2。
- `envs/JSBSim/tasks/`：任务定义，负责观测、动作、奖励、终止条件和分层控制。
- `envs/JSBSim/reward_functions/`：高度、姿态、导弹姿态、事件驱动、射击惩罚等奖励。
- `envs/JSBSim/termination_conditions/`：低高度、极端姿态、过载、超时、安全返航等终止条件。
- `envs/JSBSim/configs/`：所有 JSBSim 场景 YAML 配置，`--scenario-name` 与这里的路径一一对应。
- `envs/JSBSim/model/`：预训练/基线模型，例如 `actor_heading.pt`、`baseline_model.pt`、导弹躲避模型等。
- `envs/JSBSim/data/`：JSBSim 官方数据子模块，包含飞机、发动机、系统、测试数据等，体积较大，通常作为外部仿真资源对待。
- `docs/`：导弹建模、参数化射击、人机交互说明。
- `renders/`：若干固定渲染示例脚本。
- `tests/`：PPO、环境、向量环境、runner 的 pytest 测试。
- `assets/`：README 图片与 GIF。
- `*.acmi`：Tacview 可打开的飞行轨迹记录文件。

## 核心运行链路

训练主链路如下：

1. `scripts/train/train_jsbsim.py` 调用 `config.get_config()` 构建参数解析器。
2. `parse_args()` 增加 JSBSim 专用参数，如 `--scenario-name`、`--render-mode`。
3. `make_train_env()` / `make_eval_env()` 根据 `--env-name` 创建：
   - `SingleControlEnv`
   - `SingleCombatEnv`
   - `MultipleCombatEnv`
4. 环境初始化时通过 `parse_config(scenario_name)` 读取 `envs/JSBSim/configs/{scenario_name}.yaml`。
5. `BaseEnv.load_simulator()` 为 YAML 中每个 `aircraft_configs` 条目创建 `AircraftSimulator`。
6. 环境根据 YAML 的 `task` 字段分派到具体 Task 类。
7. runner 进入训练循环：`warmup()` -> `collect()` -> `envs.step()` -> `buffer.insert()` -> `compute()` -> `train()` -> `eval()` -> `save()`。
8. 模型默认保存到 `scripts/results/{env}/{scenario}/{algorithm}/{experiment}/runX/`。
9. 渲染时会输出 `.txt.acmi`，可用 Tacview 打开。

## 环境层设计

### BaseEnv

`envs/JSBSim/envs/env_base.py` 是所有空战环境的公共基类，继承 `gymnasium.Env`。它负责：

- 读取 YAML 配置中的 `max_steps`、`sim_freq`、`agent_interaction_steps`、`battle_field_center`。
- 创建飞机仿真器 `AircraftSimulator`。
- 根据飞机 ID 首字符划分队伍：默认与第一个智能体首字符相同的是己方 `ego_ids`，不同首字符的是敌方 `enm_ids`。
- 建立 `partners` 和 `enemies` 关系。
- 管理临时仿真器 `_tempsims`，主要用于导弹。
- 将字典形式的智能体观测/奖励/终止状态按固定顺序打包为 numpy 数组。
- 在 `step()` 中完成动作归一化、JSBSim 多步推进、导弹推进、任务更新、奖励和终止判断。

环境时间步：

```text
time_interval = agent_interaction_steps / sim_freq
```

例如多数空战 YAML 使用 `sim_freq: 60`、`agent_interaction_steps: 12`，即每次智能体交互推进 0.2 秒物理时间。

### SingleControlEnv

`envs/JSBSim/envs/singlecontrol_env.py` 只支持单架飞机。当前主要任务是 `heading`，用于训练低层航向/高度/速度控制策略。该策略后续会作为分层空战任务的低层控制器。

### SingleCombatEnv

`envs/JSBSim/envs/singlecombat_env.py` 支持 1v1。根据 YAML 的 `task` 字段加载：

- `singlecombat`
- `hierarchical_singlecombat`
- `singlecombat_dodge_missile`
- `hierarchical_singlecombat_dodge_missile`
- `singlecombat_shoot`
- `hierarchical_singlecombat_shoot`
- `HumanSingleCombat`
- `HumanSingleCombat_shoot`

reset 时会随机打乱初始状态，实现换边训练。

### MultipleCombatEnv

`envs/JSBSim/envs/multiplecombat_env.py` 支持 2v2，多智能体 MAPPO 使用集中式共享观测 `share_observation_space`。`step()` 会返回：

```text
obs, share_obs, rewards, dones, info
```

奖励会先按单个智能体计算，再在同队内求平均，使同队成员共享团队奖励。

## 仿真器层设计

### AircraftSimulator

位置：`envs/JSBSim/core/simulatior.py`。

职责：

- 封装 `jsbsim.FGFDMExec`。
- 从 `envs/JSBSim/data` 加载飞机模型，如 `f16`。
- 初始化经纬度、高度、航向、速度等 JSBSim 属性。
- 将 JSBSim 属性同步为内部状态：
  - geodetic：经度、纬度、高度
  - position：北、东、上
  - posture：滚转、俯仰、航向
  - velocity：北、东、上速度
- 管理飞机状态：
  - `ALIVE`
  - `CRASH`
  - `SHOTDOWN`
- 管理导弹关系：
  - `launch_missiles`
  - `under_missiles`
- 通过 `log()` 生成 Tacview ACMI 记录行。

### MissileSimulator

位置：`envs/JSBSim/core/simulatior.py`。

导弹模型是项目自实现的质点动力学和比例导引模型，不依赖 JSBSim 飞机 FDM。默认类似 AIM-9L：

- 最大飞行时间 `_t_max = 60`
- 发动机工作时间 `_t_thrust = 3`
- 比例导引系数 `_K = 3`
- 最大过载 `_nyz_max = 30`
- 爆炸半径 `_Rc = 300`
- 最小速度 `_v_min = 150`

导弹命中后会调用目标飞机的 `shotdown()`；失速、超时或距离持续增大时判为 `MISS`。相关理论说明在 `docs/missile_engine.md`。

## Task 层设计

Task 是项目最关键的研究层，定义强化学习问题本身：观测空间、动作空间、动作归一化、奖励函数、终止条件和任务内状态更新。

### HeadingTask

位置：`envs/JSBSim/tasks/heading_task.py`。

用于单机低层飞行控制：

- 观测空间：`Box(shape=(12,))`
- 动作空间：`MultiDiscrete([41, 41, 41, 30])`
- 动作含义：副翼、升降舵、方向舵、油门。
- 动作归一化：
  - 前三维映射到 `[-1, 1]`
  - 油门映射到 `[0.4, 0.9]`
- 奖励：`HeadingReward`、`AltitudeReward`
- 终止：`UnreachHeading`、`ExtremeState`、`Overload`、`LowAltitude`、`Timeout`

该任务训练出的 `envs/JSBSim/model/actor_heading.pt` 是分层空战任务的重要低层策略。

### SingleCombatTask

位置：`envs/JSBSim/tasks/singlecombat_task.py`。

用于 1v1 无导弹空战：

- 观测空间：`Box(shape=(15,))`
- 动作空间：`MultiDiscrete([41, 41, 41, 30])`
- 观测内容包括本机高度、姿态、速度，以及相对敌机的速度差、高度差、AO、TA、距离和侧向标志。
- 奖励：`AltitudeReward`、`PostureReward`、`EventDrivenReward`
- 终止：低高度、极端状态、过载、安全返航、超时。
- 如果 YAML 设置 `use_baseline: true`，敌方由规则/基线模型控制，训练智能体数量变为 1。

### HierarchicalSingleCombatTask

位置：`envs/JSBSim/tasks/singlecombat_task.py`。

这是毕业设计主题中“分层强化学习”的核心实现之一。

高层策略输出：

```text
MultiDiscrete([3, 5, 3])
```

含义：

- 高度变化：`[-0.1, 0, 0.1]`
- 航向变化：`[-pi/6, -pi/12, 0, pi/12, pi/6]`
- 速度变化：`[0.05, 0, -0.05]`

然后 Task 将高层动作与当前状态拼成 12 维输入，交给预训练低层 `PPOActor`，低层输出 4 维飞控动作。这样把原始约 41 x 41 x 41 x 30 的低层离散控制空间，压缩为 45 个高层机动动作。

### 导弹任务

位置：`envs/JSBSim/tasks/singlecombat_with_missle_task.py`。

注意：文件名是 `missle`，不是标准拼写 `missile`。引用路径时保持现状。

主要类：

- `SingleCombatDodgeMissileTask`：导弹由规则控制发射，智能体学习规避。
- `HierarchicalSingleCombatDodgeMissileTask`：分层版规避导弹任务。
- `SingleCombatShootMissileTask`：动作空间增加导弹发射决策，智能体学习何时发射。
- `HierarchicalSingleCombatShootTask`：高层机动决策加导弹发射决策。

导弹观测在原 15 维基础上扩展到 21 维，新增来袭导弹相对速度、高度差、AO、TA、距离和侧向标志。

射击任务动作空间：

```text
Tuple([MultiDiscrete([41, 41, 41, 30]), Discrete(2)])
```

分层射击任务动作空间：

```text
Tuple([MultiDiscrete([3, 5, 3]), Discrete(2)])
```

`--use-prior` 会让 PPO actor 在发射决策中引入基于距离和攻击角的 Beta-Bernoulli 先验，理论说明见 `docs/parameterized_shooting.md`。

### MultipleCombatTask

位置：`envs/JSBSim/tasks/multiplecombat_task.py`。

用于 2v2：

- `num_agents = 4`
- 每个智能体有 1 个 partner 和 2 个 enemies。
- 观测长度为 `9 + (num_agents - 1) * 6`。
- MAPPO 使用 `share_observation_space = num_agents * obs_length`。
- `HierarchicalMultipleCombatTask` 同样使用预训练低层策略把高层机动动作转成底层控制。
- `HierarchicalMultipleCombatShootTask` 额外加入导弹发射动作。

## 奖励函数

奖励函数统一继承 `BaseRewardFunction`。基础类支持：

- `RewardClass_scale`：奖励缩放。
- `RewardClass_potential`：是否使用势函数差分奖励。
- `reward_trajectory`：记录单局奖励轨迹。

主要奖励：

- `AltitudeReward`：鼓励保持安全高度，低于危险高度惩罚。
- `HeadingReward`：用于单机航向控制，鼓励接近目标航向、高度、速度。
- `PostureReward`：空战姿态优势奖励，核心变量是 AO、TA 和距离 R。
- `MissilePostureReward`：导弹攻防场景下的姿态/来袭导弹相关奖励。
- `EventDrivenReward`：事件奖励，击落敌机加分，被击落或坠毁扣分。
- `ShootPenaltyReward`：射击任务中约束无效/过度发射。

## 终止条件

主要终止条件位于 `envs/JSBSim/termination_conditions/`：

- `Timeout`：超过 `max_steps`。
- `LowAltitude`：高度过低。
- `ExtremeState`：姿态等极端状态。
- `Overload`：过载超过限制。
- `SafeReturn`：己方被击落/坠毁，或所有敌机已失效且无来袭导弹。
- `UnreachHeading`：单机航向控制任务中长期无法达到目标。

## 算法实现

### PPO

位置：`algorithms/ppo/`。

结构：

- `PPOActor`：`MLPBase -> GRU 可选 -> ACTLayer`
- `PPOCritic`：状态价值网络。
- `PPOPolicy`：封装 actor、critic、optimizer 和推理接口。
- `PPOTrainer`：实现 PPO clipped objective、value loss、entropy loss、梯度裁剪和多 epoch 更新。

默认网络配置来自 `config.py`：

- `hidden_size = "128 128"`
- `act_hidden_size = "128 128"`
- `use_recurrent_policy = True`
- `recurrent_hidden_size = 128`
- `recurrent_hidden_layers = 1`

`algorithms/utils/distributions.py` 支持 `Discrete`、`MultiDiscrete`、`MultiBinary`、`Box` 和射击先验用的 `BetaShootBernoulli`。

### MAPPO

位置：`algorithms/mappo/`。

与 PPO 类似，但 policy 接口区分：

- actor 使用个体局部观测 `obs`。
- critic 使用集中式共享观测 `cent_obs` / `share_obs`。

主要由 `ShareJSBSimRunner` 和 `MultipleCombatEnv` 使用。

### 自博弈

位置：

- `runner/selfplay_jsbsim_runner.py`
- `runner/share_jsbsim_runner.py`
- `algorithms/utils/selfplay.py`

支持的对手采样策略：

- `sp`：始终选择最新策略。
- `fsp`：从历史策略中均匀随机选择。
- `pfsp`：根据 ELO 分布做优先采样。

1v1 的 `SelfplayJSBSimRunner` 会维护：

- `policy_pool`：历史策略及 ELO。
- `opponent_policy`：训练时加载的历史对手。
- `eval_opponent_policy`：评估时使用。
- `actor_{episode}.pt`：每次保存的历史 actor。
- `actor_latest.pt` / `critic_latest.pt`：最新模型。

评估时会根据当前策略与对手平均回合奖励差更新 ELO。2v2 的 `ShareJSBSimRunner` 也支持自博弈，但当前 ELO 更新逻辑比 1v1 简化。

## 典型训练脚本

进入 `scripts/` 后运行：

```bash
bash train_heading.sh
bash train_vsbaseline.sh
bash train_selfplay.sh
bash train_selfplay_shoot.sh
bash train_share_selfplay.sh
```

常见含义：

- `train_heading.sh`：训练 SingleControl 的低层航向控制。
- `train_vsbaseline.sh`：1v1 对固定基线训练。
- `train_selfplay.sh`：1v1 无导弹分层自博弈。
- `train_selfplay_shoot.sh`：1v1 导弹射击自博弈，可配合 `--use-prior`。
- `train_share_selfplay.sh`：2v2 MAPPO 自博弈。

重要参数：

- `--env-name`：`SingleControl`、`SingleCombat`、`MultipleCombat`。
- `--scenario-name`：对应 `envs/JSBSim/configs` 下的 YAML 路径，不带 `.yaml`。
- `--algorithm-name`：`ppo` 或 `mappo`。
- `--use-selfplay`：启用历史策略池对抗。
- `--selfplay-algorithm`：`sp`、`fsp`、`pfsp`。
- `--n-choose-opponents`：并行 rollout 中选择多少个不同历史对手。
- `--use-eval`：训练中周期性评估。
- `--eval-interval`、`--eval-episodes`、`--n-eval-rollout-threads`：评估频率和规模。
- `--render-mode real_time`：评估时通过 Tacview Advanced 实时遥测。
- `--use-prior`：导弹射击任务中启用领域先验。

注意：脚本中有 `--clip-params 0.2`，但 `config.py` 定义的是 `--clip-param`。由于训练入口使用 `parse_known_args()`，未知参数可能被静默忽略并使用默认值。正式实验前应核对是否需要统一为 `--clip-param`。

## 场景配置

所有场景都在 `envs/JSBSim/configs/`。

常用配置：

- `1/heading.yaml`：单机航向控制。
- `1/HumanFreeFly.yaml`：人类自由飞行。
- `1v1/NoWeapon/Selfplay.yaml`：1v1 无武器自博弈。
- `1v1/NoWeapon/HierarchySelfplay.yaml`：1v1 无武器分层自博弈。
- `1v1/NoWeapon/vsBaseline.yaml`：1v1 无武器对基线。
- `1v1/DodgeMissile/Selfplay.yaml`：1v1 规避导弹自博弈。
- `1v1/DodgeMissile/HierarchySelfplay.yaml`：1v1 分层规避导弹。
- `1v1/ShootMissile/Selfplay.yaml`：1v1 导弹发射自博弈。
- `1v1/ShootMissile/HierarchySelfplay.yaml`：1v1 分层导弹发射自博弈。
- `2v2/NoWeapon/Selfplay.yaml`：2v2 无武器。
- `2v2/NoWeapon/HierarchySelfplay.yaml`：2v2 分层无武器。
- `2v2/ShootMissile/HierarchySelfplay.yaml`：2v2 分层导弹场景。

新增场景时，一般需要：

1. 在 `envs/JSBSim/configs/` 下新增 YAML。
2. 设置 `task` 字段，并确保对应 Env 的 `load_task()` 支持该 task 名。
3. 配置 `aircraft_configs`，智能体 ID 首字符决定队伍。
4. 配置仿真频率、交互步数、最大步数、高度/过载限制。
5. 配置 reward 参数，如 `PostureReward_scale`、`AltitudeReward_safe_altitude`。
6. 如涉及导弹，配置 `missile`、`max_attack_angle`、`max_attack_distance`、`min_attack_interval`。

## 渲染与可视化

离线渲染：

```bash
cd renders
python render_1v1.py
python render_2v2.py
```

或使用通用入口：

```bash
python scripts/render/render_jsbsim.py --model-dir <模型目录> --env-name <环境> --scenario-name <场景>
```

输出是 Tacview 可打开的 `.txt.acmi` / `.acmi` 文件。

实时渲染：

```bash
bash train_selfplay.sh --render-mode real_time --use-eval --eval-interval 32
```

实时渲染依赖 Tacview Advanced。`runner/tacview.py` 会开启 TCP 服务并等待 Tacview 连接，Tacview 中需要进入 `Record -> Real-time Telemetry` 输入控制台打印的 IP 和端口。

## 人机交互

相关文件：

- `docs/Human-agent.md`
- `runner/human_in_loop.py`
- `envs/JSBSim/human_agent/`
- `envs/JSBSim/human_task/`
- `scripts/human_combat/`
- `scripts/human_free_fly.sh`
- `scripts/human_shoot_1v1.sh`

键盘控制大致包括：

- 方向键控制副翼/升降舵。
- `Z` / `X` 控制方向舵。
- `Page Up` / `Page Down` 控制油门。

当前人机交互也可通过 Tacview Advanced 实时观察。`HumanSingleCombatTask` 在文档中仍标注有 TODO，继续开发时应先检查实际任务类和脚本是否完整。

## 测试与验证

测试文件：

- `tests/test_ppo.py`：PPO actor、critic、buffer、trainer 形状和基本流程。
- `tests/test_jsbsim.py`：SingleControl、SingleCombat、MultipleCombat、向量环境、部分 runner 流程。

常用测试命令：

```bash
python -m pytest tests
```

注意：

- JSBSim 环境测试可能较慢，并依赖本机已正确安装 `jsbsim`、`gymnasium`、`torch` 等包。
- 部分测试会真正创建环境并推进仿真，不是纯单元测试。
- Windows PowerShell 中运行 `.sh` 需要 Git Bash、WSL，或手动改成 PowerShell 命令。

## 重要产物

- `envs/JSBSim/model/actor_heading.pt`：分层任务低层飞行控制模型，缺失会导致分层任务加载失败。
- `envs/JSBSim/model/baseline_model.pt`：追击/机动等基线智能体依赖的模型。
- `envs/JSBSim/model/dodge_missile_model.pt`：规避导弹基线模型。
- `scripts/results/.../actor_latest.pt`：训练得到的最新 actor。
- `scripts/results/.../critic_latest.pt`：训练得到的最新 critic。
- `scripts/results/.../actor_{episode}.pt`：自博弈历史策略池中的 actor。
- `.txt.acmi` / `.acmi`：Tacview 轨迹文件。

## 与论文主题相关的亮点

- 分层强化学习：高层策略负责机动/战术意图，低层策略负责飞控执行，降低动作空间维度。
- 物理真实性：基于 JSBSim 飞行动力学推进飞机状态，而不是简单二维粒子环境。
- 序列决策：Actor/Critic 支持 GRU，适合部分可观测空战中的历史信息建模。
- 自博弈：维护历史策略池，可用 `sp`、`fsp`、`pfsp` 选择对手，并在 1v1 中更新 ELO。
- 导弹攻防：实现比例导引导弹动力学，支持规避导弹和学习发射时机。
- 领域先验融合：导弹射击动作可通过 Beta-Bernoulli 先验融合攻击角、距离等专家知识。
- 多场景泛化：覆盖单机控制、1v1、2v2、无武器、导弹、人机交互等任务。
- 可解释可视化：通过 Tacview 离线/实时观察轨迹，便于论文实验分析和行为诊断。

## 后续开发建议

- 新增代码注释保持中文，尤其是任务、奖励、终止条件、动作转换等研究相关逻辑。
- 修改 Task 时同步检查：
  - `load_observation_space()`
  - `load_action_space()`
  - `get_obs()`
  - `normalize_action()`
  - `reset()`
  - `step()`
  - 对应 YAML 的 `task` 字段和参数。
- 修改动作空间时，同步检查 `algorithms/utils/act.py`、`distributions.py`、buffer 中动作 shape 处理，以及测试。
- 修改分层低层策略输入时，同步检查 `HeadingTask.get_obs()` 和 `Hierarchical*Task.normalize_action()` 中拼接的 12 维输入。
- 修改导弹发射逻辑时，同步检查：
  - `MissileSimulator`
  - `SingleCombatDodgeMissileTask.step()`
  - `SingleCombatShootMissileTask.step()`
  - `HierarchicalMultipleCombatShootTask.step()`
  - `ShootPenaltyReward`
- 修改自博弈时，同步检查 `policy_pool`、`actor_{episode}.pt` 保存/加载、评估对手选择和 ELO 更新。
- 不要轻易改动 `envs/JSBSim/data`，它是 JSBSim 数据子模块，除非明确要更新飞行动力学数据。
- 仓库内有中文文件名和中文注释，Windows 终端可能显示乱码；编辑文档和代码时使用 UTF-8。
