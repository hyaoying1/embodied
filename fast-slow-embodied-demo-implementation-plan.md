# Fast–Slow Embodied Agent Demo：逐步实施路线

> 本文将 [`fast-slow-embodied-demo.md`](./fast-slow-embodied-demo.md) 中的整体设计拆分为可执行步骤。目标是先完成一个可录屏、可解释的 3D Demo，再逐步升级为可做研究实验的原型。

---

## 0. 先确定范围与完成标准

### 第一版只做什么

第一版只验证一个机制：

```text
Agent 记得目标物体在位置 A
→ 环境在执行中将物体移到位置 B
→ 快速决策层检测记忆与观察冲突
→ 系统按需调用慢速规划层
→ Agent 重新找到目标并完成任务
```

### 第一版不做什么

```text
- 不训练大型 VLA；
- 不实现真实机械臂；
- 不实现开放世界语言理解；
- 不一开始接入昂贵的 VLM/LLM API；
- 不追求论文级 benchmark 结果；
- 不把环境真值伪装成 Agent 运行时观察。
```

### 第一版验收标准

完成以下内容即视为 MVP 完成：

- [ ] Agent 能在一个 3D 房间中执行“找杯子并放到桌上”；
- [ ] Agent 具有可查看的结构化记忆；
- [ ] 目标在 Agent 观察后可被环境移动；
- [ ] `Always Fast` baseline 会因旧记忆失败；
- [ ] `Fast–Slow Router` 能检测冲突、调用 Slow Brain 并恢复；
- [ ] 屏幕或日志能展示动作、置信度、记忆状态和委托事件；
- [ ] 可以录制一段 60–90 秒的对比 demo 视频。

---

## 1. 创建项目骨架

### 目标

建立一个职责清晰、以后可以替换模型和环境的 Python 项目。

### 建议目录

```text
embodied/
├── fast-slow-embodied-demo.md
├── fast-slow-embodied-demo-implementation-plan.md
├── README.md
├── pyproject.toml
├── src/
│   └── fast_slow_agent/
│       ├── __init__.py
│       ├── config.py
│       ├── environment/
│       │   ├── __init__.py
│       │   ├── interface.py
│       │   ├── ai2thor_env.py
│       │   └── events.py
│       ├── memory/
│       │   ├── __init__.py
│       │   └── belief_memory.py
│       ├── planning/
│       │   ├── __init__.py
│       │   ├── candidates.py
│       │   └── slow_planner.py
│       ├── decision/
│       │   ├── __init__.py
│       │   ├── fast_router.py
│       │   └── delegation.py
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── always_fast.py
│       │   └── fast_slow_agent.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   └── runner.py
│       └── visualization/
│           ├── __init__.py
│           └── dashboard.py
├── scripts/
│   ├── run_demo.py
│   └── run_comparison.py
└── tests/
    ├── test_memory.py
    ├── test_events.py
    └── test_delegation.py
```

### 要做的事

1. 创建 Python 虚拟环境与项目元数据；
2. 设置可执行入口，例如 `python -m scripts.run_demo`；
3. 写一个最小 `README.md`，说明如何安装、运行和录制 Demo；
4. 先只安装运行第一版所需的依赖：模拟器、数值计算、日志与测试工具；
5. 不要在第一天加入 VLM、Web UI、数据库或复杂训练框架。

### 验证

```text
- 空项目能启动；
- 可以运行一个 hello-world 脚本；
- 测试命令能发现并执行一个空测试；
- 项目路径与 import 正常工作。
```

### 完成条件

```text
开发者能在新终端中根据 README 初始化环境并运行一个最小命令。
```

---

## 2. 接入 3D 环境并验证基本交互

### 目标

使用 AI2-THOR 建立一个包含厨房或客厅的 3D 家庭场景，让 Agent 能获得第一人称观察并执行基础动作。

### 要做的事

1. 创建 `Environment` 抽象接口：

```python
reset(task) -> observation
step(action) -> observation, reward, done, info
get_visible_objects() -> list[ObjectObservation]
```

2. 在 `ai2thor_env.py` 实现该接口；
3. 固定选择一个可复现的场景，避免初期随机场景导致调试困难；
4. 在场景中定位以下对象：

```text
blue mug
left cabinet
right cabinet
dining table
```

5. 实现基础动作包装：

```text
MoveAhead
RotateLeft
RotateRight
OpenObject
PickupObject
PutObject
```

6. 每一步保存：

```text
step 编号
第一人称 RGB 图像
执行动作
成功/失败状态
可见物体摘要
```

### 验证

手工运行一个固定动作序列：

```text
reset
→ 转向厨房
→ 移动到柜子前
→ 打开柜子
→ 观察到杯子
→ 拿起杯子
→ 移动到餐桌
→ 放下杯子
```

### 完成条件

```text
Agent 能稳定地在同一场景完成一次人工指定的“开柜—拿杯—放桌”动作序列，且每步有日志或截图。
```

### 常见问题

| 问题 | 处理方式 |
|---|---|
| 场景里没有目标物体 | 先固定 scene 与 spawn/摆放逻辑。 |
| 物体 ID 不稳定 | 用名称、类别和空间位置建立可复现定位规则。 |
| Agent 到不了柜子前 | 先记录可达位置，或使用环境提供的可达点。 |
| macOS 图形渲染异常 | 优先做 headless/低分辨率试跑，再单独处理渲染与录屏。 |

---

## 3. 定义任务与可复现的初始场景

### 目标

把“找杯子并放桌上”定义成可程序检查成功与失败的任务，而不是人工观察结果。

### 要做的事

1. 定义任务对象：

```python
Task(
    target_object="blue_mug",
    destination="dining_table",
    instruction="找到蓝色马克杯，并把它放到餐桌上。"
)
```

2. 编写 `reset_task()`：

```text
- 重置固定场景；
- 将蓝色马克杯放入左侧柜子；
- 关闭左侧柜子；
- 放置 Agent 初始位置与朝向；
- 清空 Agent 记忆与历史。
```

3. 编写成功检查：

```text
蓝色马克杯位于餐桌表面
→ task success = true
```

4. 编写失败检查：

```text
- 超过最大行动步数；
- 非法动作过多；
- 进入重复循环；
- 对错误物体执行不可恢复操作。
```

### 验证

- 每次 `reset_task()` 后，目标、柜子、桌子和 Agent 初始状态一致；
- 手动成功轨迹能触发 `success = true`；
- 未完成任务的轨迹不会误报成功。

### 完成条件

```text
任务的初始状态、成功条件和失败条件都能被程序稳定检查。
```

---

## 4. 实现基础 Agent：Always Fast Baseline

### 目标

先实现一个会依赖旧记忆、但不会主动升级推理的 baseline。它用于制造清晰的失败对照。

### 行为规则

```text
1. 初次发现目标时，将目标位置写入 Memory；
2. 根据记忆生成路线并前往最后已知位置；
3. 若在目标位置找不到物体：
   - 继续尝试有限次数，或直接判定任务失败；
   - 不调用 Slow Brain；
   - 不做全局重新搜索。
```

### 要做的事

1. 编写 `always_fast.py`；
2. 为 Agent 定义统一接口：

```python
act(observation) -> Action
```

3. 初期允许该 baseline 使用简单、确定性的导航和交互脚本；
4. 记录它每一步使用的记忆、选择的动作和失败原因。

### 验证

在没有环境变化时：

```text
Always Fast 应完成任务。
```

在杯子被移动后：

```text
Always Fast 应继续去左柜，找不到目标，最终失败或明显浪费动作。
```

### 完成条件

```text
同一任务在“无变化”和“目标已移动”两种条件下得到可重复、可解释的不同结果。
```

---

## 5. 实现结构化时序记忆

### 目标

让 Agent 能明确表达：它知道什么、何时知道、该信息是否仍可信。

### 数据结构

```python
BeliefRecord(
    object_name="blue_mug",
    last_known_location="left_cabinet",
    observed_step=3,
    confidence=0.92,
    environment_version=1,
    freshness="fresh",  # fresh / stale / contradicted
)
```

### 要做的事

1. 在观察中发现目标物体时写入或更新记忆；
2. 为每条记忆计算 `memory_age = current_step - observed_step`；
3. 当出现下列信号时降低置信度：

```text
- 到达最后已知位置但未观察到目标；
- 目标容器被打开但目标不在其中；
- 环境事件发生在最后一次观察之后；
- 距离最后一次有效观察已经过很长时间。
```

4. 定义记忆状态：

```text
fresh：近期被当前观察支持；
stale：可能因时间或环境事件失效；
contradicted：当前观察明确否定旧记忆。
```

5. 把每次记忆更新写入事件日志。

### 验证

```text
Step 3：看到杯子在左柜，confidence = 0.92，fresh。
Step 8：杯子被移动，环境版本增加，记忆变为 stale。
Step 10：打开左柜未发现杯子，记忆变为 contradicted，confidence 大幅下降。
```

### 完成条件

```text
Demo 面板或终端日志能清楚显示：目标位置、最后观察时间、置信度与 fresh/stale/contradicted 状态。
```

---

## 6. 实现动态环境事件

### 目标

在 Agent 已形成记忆后修改环境，构造真正需要委托的状态。

### 第一版只实现一个事件

```text
move_target_after_observation
```

触发规则：

```text
当 Agent 已观察到 blue_mug 位于 left_cabinet，
且当前 step >= 8 时，
将 blue_mug 移动到 right_cabinet。
```

### 事件接口

```python
class EnvironmentEvent:
    def should_trigger(state, history) -> bool: ...
    def apply(environment) -> EventResult: ...
```

### 后续可加的事件

```text
- partial_occlusion：用其他物体遮挡目标；
- add_distractor：加入颜色、形状相近的干扰物；
- close_container：关闭原本打开的容器；
- lock_container：使原计划中的容器无法打开；
- replace_object：将目标替换成相似但错误的物体。
```

### 验证

- 事件只在 Agent 已看到杯子后触发；
- 目标确实从左柜移动到右柜；
- 事件被记录到日志和可视化时间线；
- 不变场景中不触发事件。

### 完成条件

```text
可以通过配置开关稳定复现“目标被移动”的场景，也可以关闭事件运行正常任务。
```

---

## 7. 实现候选行动生成器

### 目标

把 Fast Brain 的问题变成“从有限候选中快速选择”，而不是直接生成任意复杂动作。

### 第一版候选动作

```text
navigate_to(last_known_location)
open(last_known_container)
inspect(last_known_location)
search_alternative_container
replan_from_memory
delegate_to_slow_brain
abstain
```

### 每个候选行动附带的元数据

```python
CandidateAction(
    name="inspect_last_known_location",
    estimated_cost=2,
    estimated_risk="low",
    information_gain="high",
    reversible=True,
)
```

### 要做的事

1. 根据当前子目标、Memory 与环境状态生成合法候选；
2. 去除不可能执行的候选，例如“当前没有面对柜子却立即打开柜子”；
3. 为每个候选加入代价、风险、是否可逆、预计信息增益等特征；
4. 把候选列表提供给 Fast Brain 与日志面板。

### 验证

在目标记忆为 fresh 时：

```text
候选优先包含 navigate_to(last_known_location)。
```

在目标记忆 contradicted 时：

```text
候选必须包含 inspect / search / replan / delegate，
且不应只重复“继续拿取目标”。
```

### 完成条件

```text
每一步日志都能显示候选集合、候选属性和最终被选择的行动。
```

---

## 8. 实现 Fast Brain 与 Delegation Router

### 目标

实现快速、可解释的选择机制。第一版使用规则；第二版再将规则替换为学习模型。

### 8.1 第一版：规则 Fast Router

推荐的初始策略：

```text
若目标当前可见：
  选择 act。

若记忆 fresh 且目标位置尚未被验证：
  选择 navigate_to(last_known_location)。

若记忆 stale 且低风险观察可获得新信息：
  选择 observe 或 verify。

若记忆 contradicted：
  选择 delegate_to_slow_brain。

若风险高、行动不可逆且置信度低：
  选择 abstain。
```

### 8.2 置信度计算

第一版可用清晰的分数函数：

```text
confidence =
  + 当前看见目标
  + 记忆新鲜
  + 当前路线可达
  - 记忆年龄
  - 观察与记忆冲突
  - 环境变化信号
  - 高风险或不可逆行动
```

这不是最终研究方法，但能让 Demo 的选择原因透明。

### 8.3 第二版：轻量学习 Fast Brain

在收集足够轨迹后，用小模型替换规则：

```text
输入：观察摘要、记忆年龄、置信度、冲突状态、候选行动特征、预算。
输出：每个候选行动的分数、选择概率、delegate 概率。
模型：Logistic Regression、LightGBM 或两层 MLP。
```

训练标签可来自：

```text
- 环境 oracle 的最短成功行动；
- Slow Brain 的正确行动；
- hindsight：回看轨迹后标记“此时是否应委托”。
```

### 验证

在三个状态下检查路由结果：

| 状态 | 预期路由 |
|---|---|
| 目标可见、记忆新鲜 | `act` |
| 目标位置未验证、记忆过期 | `observe` 或 `verify` |
| 目标位置已检查但与记忆冲突 | `delegate` 或 `replan` |

### 完成条件

```text
Fast Brain 能输出：候选行动、选择、置信度、触发委托的原因与决策耗时。
```

---

## 9. 实现 Slow Brain

### 目标

让系统在委托时得到一条更可靠的高层建议，并把执行权交还给 Fast Brain。

### 第一版：Oracle / 规则 Slow Brain

第一版可使用环境状态或规则搜索实现：

```text
输入：任务、当前记忆、当前观察、候选行动。
输出：更新后的目标位置、下一个子目标、新候选行动。
```

示例：

```text
输入：左柜已打开但未发现蓝杯，记忆 contradicted。
输出：搜索右柜；将 blue_mug 的假设位置更新为 right_cabinet。
```

> 注意：此版本只能用于验证控制流。界面与文档必须明确标为 oracle/规则 Slow Brain，不能将其当作真实 VLM 能力。

### 第二版：真实 Slow Brain

可替换为 API VLM/LLM 或冻结的开源 VLM。输入必须是 Agent 可见的信息：

```text
- 当前第一人称图像；
- 最近几帧或对象摘要；
- 用户任务；
- 结构化记忆；
- 当前候选行动；
- 委托原因。
```

输出采用固定 JSON，便于解析：

```json
{
  "subgoal": "search_right_cabinet",
  "target_location": "right_cabinet",
  "recommended_action": "navigate_to_right_cabinet",
  "confidence": 0.74
}
```

### 验证

```text
Fast Brain 发出 delegate
→ Slow Brain 返回更新计划
→ Memory 更新
→ Fast Brain 接回执行
→ Agent 不需要全程依赖 Slow Brain。
```

### 完成条件

```text
委托只在必要时发生，且 Slow Brain 的输出能被解析并改变之后的行动轨迹。
```

---

## 10. 组装 Fast–Slow Agent

### 目标

把 Environment、Memory、Candidate Generator、Fast Brain、Router 与 Slow Brain 串成完整闭环。

### 每个时间步的控制循环

```text
1. 接收当前 observation。
2. 更新可见物体与 Memory。
3. 判断是否存在环境变化或记忆冲突。
4. 生成合法 Candidate Actions。
5. Fast Brain 对候选行动打分并输出置信度。
6. Router 决定：act / observe / verify / replan / delegate / abstain。
7. 若 delegate：调用 Slow Brain，更新计划与 Memory。
8. 执行动作。
9. 保存轨迹、耗时、置信度、预算和结果。
10. 判断任务是否成功、失败或继续。
```

### 伪代码

```python
while not done:
    observation = env.observe()
    memory.update(observation, history)
    candidates = candidate_generator.build(task, observation, memory)

    decision = fast_brain.choose(observation, memory, candidates, budget)

    if router.should_delegate(decision, memory, budget):
        plan = slow_brain.resolve(task, observation, memory, candidates)
        memory.update_from_plan(plan)
        candidates = candidate_generator.build(task, observation, memory)
        decision = fast_brain.choose(observation, memory, candidates, budget)

    result = env.step(decision.action)
    logger.record(observation, memory, candidates, decision, result)
```

### 验证

- 无事件任务中，Agent 主要使用 Fast Brain；
- 杯子移动事件中，Agent 在冲突发生后触发委托；
- 委托后，Agent 的目标/路径发生可见改变；
- 超过最大步数时能干净退出并输出失败报告。

### 完成条件

```text
一个命令可以完整运行 Fast–Slow Agent，并生成结构化轨迹日志。
```

---

## 11. 实现可视化与录屏

### 目标

让观众无需阅读代码，就能理解系统为何比普通 Agent 更可靠和高效。

### MVP 可视化方案

先使用终端日志 + 定期保存截图；不需要第一天就写完整 Web UI。

每步输出：

```text
[step=10]
observation: left cabinet open; blue mug not visible
memory: blue_mug -> left cabinet | confidence=0.31 | contradicted
candidates: inspect, search_right_cabinet, replan, delegate
fast choice: delegate | confidence=0.38 | latency=4ms
slow call: #2/10
slow plan: search_right_cabinet
```

### 第二阶段可视化方案

使用 Streamlit 或前端页面显示：

```text
- 第一人称画面；
- Agent 位置/简化俯视地图；
- Memory 卡片；
- 候选行动与置信度条；
- Slow Brain 调用预算；
- 事件时间线；
- Always Fast 与 Fast–Slow 并排回放。
```

### 录屏脚本

录制以下顺序：

```text
1. 展示任务与正常场景。
2. 展示 Always Fast：目标移动后仍信任旧记忆，失败。
3. 重置场景。
4. 展示 Fast–Slow：出现冲突、降低置信度、调用 Slow Brain、重规划、成功。
5. 展示对比：调用次数、延迟、成功状态。
```

### 完成条件

```text
60–90 秒录屏中，非技术观众也能说出：系统在不确定时没有盲目行动，而是只在必要时请求更强推理。
```

---

## 12. 加入对照实验与自动评估

### 目标

从“单次 demo 成功”升级到“机制是否普遍有效”的小型实验。

### 对照策略

```text
Always Fast：从不调用 Slow Brain。
Always Slow：每一步调用 Slow Brain。
Fixed Frequency：每 N 步调用一次 Slow Brain。
Fast–Slow Router：基于置信度、冲突与风险动态调用。
```

### 第一批环境变化

```text
E0：无变化。
E1：目标物体被移动。
E2：目标被部分遮挡。
E3：加入相似颜色的干扰物。
E4：原目标容器变为无法打开。
```

### 记录指标

```text
success_rate
steps_to_completion
slow_calls
slow_call_budget_used
average_fast_latency
estimated_total_latency
estimated_total_cost
stale_memory_errors
high_confidence_errors
recovery_success_rate
```

### 实验矩阵

```text
4 种 Agent 策略
× 5 种环境条件
× 每条件至少 20 个随机种子或场景变体
```

### 验证目标

Fast–Slow Router 至少应显示一个清楚趋势：

```text
相对 Always Fast：动态变化任务成功率更高。
相对 Always Slow：Slow Brain 调用数与延迟更低。
相对 Fixed Frequency：相同预算下，更少无效升级或更高成功率。
```

### 完成条件

```text
运行一条实验命令后，生成 CSV/JSON 结果与一张汇总表或图。
```

---

## 13. 从规则 Demo 升级到学习型 Fast Brain

### 目标

将“人为设定阈值”升级为可学习、可比较的轻量 decision model。

### 13.1 收集训练数据

针对不同场景随机运行：

```text
- 初始目标位置；
- Agent 初始位置；
- 目标移动时机；
- 目标移动目的地；
- 遮挡程度；
- 干扰物数量；
- Slow Brain 剩余预算。
```

每个时间步记录：

```text
状态特征
候选行动
oracle/teacher 最优行动
是否应委托
最终任务是否成功
成本与风险
```

### 13.2 训练小模型

推荐循序渐进：

```text
Baseline 1：Logistic Regression。
Baseline 2：LightGBM / Random Forest。
Model 1：两层 MLP。
```

先训练两个预测头：

```text
- candidate action score；
- delegate probability。
```

### 13.3 做校准

验证模型的 `0.8 confidence` 是否约对应 80% 的正确率。

可使用：

```text
- reliability diagram；
- Expected Calibration Error (ECE)；
- temperature scaling；
- budget-aware threshold selection。
```

### 完成条件

```text
学习型 Fast Brain 至少与规则 Router、Always Fast、Always Slow 完成可重复比较；
并报告成功率、成本、延迟与校准指标。
```

---

## 14. 建议时间表

### 第 1 周：环境与任务

| 天数 | 工作 | 交付物 |
|---|---|---|
| Day 1 | 项目骨架、运行环境、基础 README | 可启动项目 |
| Day 2 | AI2-THOR 场景、第一人称观察、基础动作 | 可操作 3D 房间 |
| Day 3 | 杯子—柜子—桌子固定任务与成功检查 | 可复现任务 |
| Day 4 | Always Fast baseline | 正常成功、变化后失败 |
| Day 5 | Memory 与日志 | 记忆状态可见 |

### 第 2 周：Fast–Slow 机制与 Demo

| 天数 | 工作 | 交付物 |
|---|---|---|
| Day 6 | 目标移动事件 | 动态变化场景 |
| Day 7 | 候选行动生成 | 候选列表与属性 |
| Day 8 | 规则 Fast Brain + Router | act/verify/delegate 决策 |
| Day 9 | oracle Slow Brain | 委托后可恢复 |
| Day 10 | 端到端运行、日志整理 | Fast–Slow 完整闭环 |
| Day 11–12 | 截图、面板、并排轨迹 | 可解释可视化 |
| Day 13–14 | 对比录屏、README、修复稳定性问题 | MVP Demo 视频 |

### 第 3–4 周：实验原型

```text
- 生成多个环境变化场景；
- 批量运行对照策略；
- 输出指标与图表；
- 加入轻量学习型 Fast Brain；
- 做置信度校准与预算控制；
- 形成初步技术报告。
```

---

## 15. 每一阶段的最终交付物

| 阶段 | 交付物 | 用途 |
|---|---|---|
| Environment | 可交互的 3D 房间与任务 reset | 提供具身闭环 |
| Baseline | Always Fast 轨迹与失败案例 | 展示问题存在 |
| Memory + Event | 过期记忆与动态变化机制 | 制造研究情境 |
| Router | Fast–Slow 委托与恢复轨迹 | 展示核心方法 |
| Visualization | 截图、面板、录屏 | 让系统可理解 |
| Evaluation | 对照表、CSV、图表 | 证明不是单案例 |
| Learning Prototype | 小型可训练 decision model | 形成研究扩展 |

---

## 16. 实现原则与防止误导的规则

```text
1. 先固定一个场景和一个任务，先让链路完整，再追求泛化。
2. 先写对照 baseline，再写你的 Fast–Slow Agent；否则无法证明机制价值。
3. 环境真值仅可用于 reset、oracle 标签和评估；不能当作运行时 Agent 输入。
4. 首版 oracle Slow Brain 必须在 UI/README 中明确标注，不应冒充真实 VLM 推理。
5. 每个动作、记忆更新、委托决定和环境事件都必须写入日志。
6. 不仅报告成功率；必须报告调用次数、延迟、预算、错误和恢复。
7. 先完成“一个变化事件 + 一个成功恢复案例”，再增加遮挡、干扰和多房间。
8. 不要在机制与评估还未稳定前训练大型模型或购买机器人硬件。
```

---

## 17. 开始执行时的第一个任务

第一件实际要做的事不是模型训练，而是：

```text
在 AI2-THOR 中选择一个固定厨房场景，
写一个脚本让 Agent 完成：
“移动到柜子 → 打开柜子 → 拿蓝色杯子 → 放到餐桌”。
```

只有在这条人工指定的动作序列稳定运行后，才继续实现：

```text
任务定义 → Always Fast baseline → Memory → 目标移动事件 → Router → Slow Brain。
```

这能避免一开始同时调试模拟器、导航、模型、记忆和可视化，导致问题无法定位。
