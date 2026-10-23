# Fast–Slow Embodied Agent Demo

一个可直接运行的 MVP，用于演示具身 Agent 如何在动态家庭环境中结合：

- **Fast Brain**：快速、低成本地从候选行动中选择下一步；
- **Memory**：保存目标最后出现的位置、观察时间和置信度；
- **Delegation Router**：当记忆与现实冲突时，选择请求更强的 Slow Brain；
- **Slow Brain**：第一版使用明确标注的 oracle planner，负责重新定位目标并修正计划；
- **Dynamic Event**：Agent 看见目标后，环境将目标从左柜移动到右柜。

> 当前实现是可复现的机制 MVP，而不是训练完成的 VLA。环境是无外部依赖的家庭任务模拟器，便于在 macOS 上直接运行和测试；后续可将 `HouseholdEnvironment` 替换为 AI2-THOR adapter。

## 运行

要求：Python 3.9 或更新版本。

运行终端版轨迹展示：

```zsh
PYTHONPATH=src python3 -m fast_slow_agent.cli
```

生成可直接在浏览器打开的视觉 dashboard：

```zsh
PYTHONPATH=src python3 -m fast_slow_agent.dashboard
open artifacts/fast-slow-demo-dashboard.html
```

程序会连续运行三种情境：

1. `Always Fast` + 目标移动：依赖旧记忆，最终失败；
2. `Fast–Slow Router` + 目标移动：检测冲突，委托一次 Slow Brain，恢复成功；
3. `Fast–Slow Router` + 正常环境：不调用 Slow Brain，直接完成任务；
4. `Fast–Slow Router` + 零 Slow Brain 预算：检测冲突后安全地 `abstain`，而不是重复错误。

## 可选：AI2-THOR 3D 后端

项目在隔离环境 `.venv` 中已安装固定版本 `ai2thor==5.0.0`，并已完成最小 3D 启动验证：`FloorPlan1_physics` 成功返回 300×300 RGB 图像和 77 个场景物体。首次启动已下载 macOS Unity 场景运行组件（约 775 MB）。

如需在另一台机器安装：

```zsh
python3 -m venv .venv
.venv/bin/python -m pip install "ai2thor==5.0.0"
```

运行真实 AI2-THOR 3D 目标移动与恢复 episode：

```zsh
PYTHONPATH=src .venv/bin/python -m fast_slow_agent.thor_demo
open artifacts/thor
```

该 episode 已验证：杯子在真实物理场景中被移动到另一张台面；Agent 在旧位置抓取失败；Fast Brain 对矛盾记忆输出高 `delegate_probability`；明确标注为 oracle 的 Slow Brain 给出新目标区域；Agent 在新位置抓取并放置到第三张台面。运行会保存五张 300×300 的第一人称 PNG 帧到 `artifacts/thor/`。

当前常规 dashboard 和单元测试仍使用无依赖的确定性家庭模拟器，以便快速、稳定地回归测试。真实 3D episode 已使用同一个 `TinyDelegationModel`，并且现在支持两种 Slow Brain 模式：

```text
oracle：默认离线模式；使用已知目标区域，保证可复现的物理回归测试。
vlm：真实视觉搜索模式；VLM 审阅每个候选台面的多张第一人称 PNG，返回目标是否可见、置信度和理由，并选择搜索区域。
```

启用 `vlm` 模式前，需在终端显式配置 API 凭据和视觉模型名：

```zsh
export FAST_SLOW_VLM_API_KEY="<your-api-key>"
export FAST_SLOW_VLM_MODEL="<your-vision-model>"
# 可选：使用兼容 Chat Completions 的自定义端点
export FAST_SLOW_VLM_ENDPOINT="https://api.openai.com/v1/chat/completions"

PYTHONPATH=src .venv/bin/python -m fast_slow_agent.thor_demo --slow-brain vlm
```

`vlm` 模式才会将 AI2-THOR 生成的候选第一人称 PNG 和固定任务提示发送到你配置的 endpoint；密钥不会被写入文件、日志或 dashboard。没有配置 key/model 时，程序会在**发送任何图像之前**报错。一次委托的默认远程预算为**最多 3 次请求、每次最多 4 张图片**；`VisionSearchSlowBrain` 会在发送前截断超出预算的候选和图片。为保证结果透明，AI2-THOR 的 `GetInteractablePoses` 与 `TeleportFull` 既用于候选视角采集，也用于选区后的抓取/放置低层控制桥接；这不是学习到的视觉导航或机器人控制器。

## 无外部视觉模型的 3D 模式

如果不使用外部 VLM/API，直接运行完全本地的 AI2-THOR 感知模式：

```zsh
PYTHONPATH=src .venv/bin/python -m fast_slow_agent.thor_local_demo
open artifacts/thor/local-perception
```

该模式的任务是移动后的绿色 Lettuce：

```text
旧位置抓取失败
→ Fast Brain 将记忆标为 contradicted
→ Fast Brain 输出高 delegate_probability
→ Slow Brain 主动遍历候选台面视角
→ 使用 AI2-THOR 本地 visible 传感结果找到目标
→ 抓取并放到第三张台面
```

它不发送图片、不使用 API、不需要 GPU 或额外模型。需要注意，它使用的是模拟器提供的局部可见性传感器，因此这是一个稳定的**本地 simulator-perception baseline**，而不是现实世界可泛化的视觉模型。实测本次 episode 扫描 11 个视角后发现目标，并成功完成恢复和放置。

## 运行测试

```zsh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## 当前功能

- 固定家庭任务：将蓝色马克杯从柜子放到餐桌；
- 结构化时序记忆：`fresh`、`unverified`、`contradicted`；
- 动态物体移动事件；
- Always Fast 与 Fast–Slow Agent 对比；
- 显示每一步动作、记忆状态、置信度、Slow Brain 调用与最终指标；
- 生成包含对照表与完整决策轨迹的自包含 HTML dashboard；
- 不依赖 GPU、API 或第三方 Python 包即可运行核心 MVP。

## 目录

```text
src/fast_slow_agent/
├── environment.py  # 家庭环境和动态事件
├── memory.py       # 结构化时序记忆
├── router.py       # Fast Brain 和委托判断
├── planner.py      # 明确标记的 oracle Slow Brain
├── agent.py        # 两种 Agent 与控制循环
├── evaluation.py   # 对照实验和指标
├── cli.py          # 终端 demo 输出
├── dashboard.py    # 生成浏览器可视化报告
└── thor_demo.py    # 真实 AI2-THOR 3D target-move episode
```

更多的系统功能说明见 [fast-slow-embodied-demo.md](./fast-slow-embodied-demo.md)，逐步实施计划见 [fast-slow-embodied-demo-implementation-plan.md](./fast-slow-embodied-demo-implementation-plan.md)。

## Fast Brain 的当前实现

当前 `FastRouter` 的普通动作（导航、检查、抓取、放置）仍使用透明规则，便于观察控制流；但**是否委托给 Slow Brain** 已升级为 `TinyDelegationModel`，一个无第三方依赖的四特征 logistic classifier。

它使用以下特征快速输出 `delegate_probability`：

```text
- memory_uncertainty：1 - 记忆置信度；
- normalized_memory_age：距离最后一次观察的步数；
- memory_contradiction：当前观察是否否定旧记忆；
- unverified_slow_hint：Slow Brain 给出的地点是否尚未亲自验证。
```

为保证本 MVP 可重复运行，分类器首先在透明的**合成 teacher 标签**上训练：矛盾记忆应升级给 Slow Brain，稳定且高置信的记忆应保留在 Fast path。研究版本应把这些标签替换为模拟器轨迹、真实 VLM/LLM teacher 或人工评估数据。

## 3D VLM 候选视角诊断

真实 VLM 搜索的质量取决于候选区域的第一人称图中是否真正看得到目标。可运行离线诊断：

```zsh
PYTHONPATH=src .venv/bin/python -m fast_slow_agent.thor_diagnostics
open artifacts/thor/coverage
```

它会在杯子移动后，对三个候选台面各采样 8 个均匀分布的可交互视角，保存图片并生成 `coverage-report.json`。该诊断只为评估图像覆盖率而读取 AI2-THOR 的 `visible` 元数据；**运行时 VLM 不会接收该元数据**。

当前实测结果：移动目标的台面有 `6/8` 个视角显示杯子，其余两个候选区域都是 `0/8`。因此 `--slow-brain vlm` 模式会向视觉模型提供每个候选区域最多 8 张多角度图片，而不是只提供单一视角。
