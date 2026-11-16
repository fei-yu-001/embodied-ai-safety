# PRISM — 具身智能安全测评平台

基于 RoboCasa / MuJoCo 的真实厨房仿真，提供 Three.js 场景、机械臂控制、工具安全观测与可复现评测。

## 当前能力

- 厨房和 PandaOmron 的真实网格、纹理及物理状态同步，支持自由视角与仿真相机。
- XY 方向盘、Z 升降、夹爪开合；精细 / 标准 / 快速档，点击移动、按住连续移动、松开停止。
- 独立仿真工作进程、命令确认、控制租约、断连停止，以及结束后的资源释放。
- 可配置工具几何与保护区，记录净距、越界和暴露时间；未启用或缺数据时不生成安全结论。
- 随机 / 静止保持 / 视觉质心策略和研究用攻击、防御与评测代码。

目前机器人固定为 PandaOmron，尚无通用 CAD 导入或自动持刀切菜策略。工具安全为几何近似观测，不自动停机。静止保持策略只推进物理时间，手动移动请使用方向盘。

## 目录

| 路径 | 内容 |
| --- | --- |
| `web/` | Vue 3 + Three.js 工作台 |
| `server/` | FastAPI、会话状态机、工作进程、资源导出与评测 |
| `experiments/` | 策略、攻击、防御、SDK 与研究流程 |
| `deploy/` | Linux 依赖、systemd、Nginx 与环境变量模板 |
| `docs/` | 两份研究说明、工具安全说明、部署文档 |

运行数据、密钥、模型素材、第三方源码、node_modules 和临时验收材料均不提交到 Git。

## 本地启动

Python 3.10–3.12，Node.js 22+。先准备 RoboCasa / robosuite 与对应素材，具体版本及路径见 [部署与运行](docs/部署与运行.md)。

```bash
python -m pip install -r server/requirements.txt -r deploy/requirements-simulation.txt
cd web
npm ci
npm run build
cd ..
```

设置 `PRISM_BACKEND=robocasa`、`PRISM_TOKEN` 和第三方环境路径后运行：

```bash
python -m server.main
```

服务默认监听 `127.0.0.1:8766`。Windows 可通过 `start_workbench.ps1 -Port 8880` 或启动脚本打开；自定义 Python 使用 `PRISM_PYTHON`。

页面选择“收集切菜工具”并加载环境。手动控制默认“标准”档；安全观测在左侧配置，下次加载生效。任务不等于自动策略，选择“静止保持”不会自动拾取刀具。

## 测试

```bash
python -m pip install -r server/requirements-dev.txt
python -m unittest discover -s server/tests -q
node web/tests/scene-hierarchy.mjs
node web/tests/visual-assets-and-glass.mjs
node web/tests/hold-gesture.mjs
```

## 部署

采用 Nginx HTTPS → loopback FastAPI → 独立 MuJoCo 工作进程。部署入口、依赖、素材安装和资源限制见 [部署与运行](docs/部署与运行.md)。不要把开发令牌用于公网。

## 研究文档

- [具身智能中的对抗样本](docs/具身智能中对抗样本.md)
- [具身智能中的决策安全](docs/具身智能中决策安全.md)
- [厨房工具安全最小用例](docs/厨房工具安全最小用例说明.md)

第三方模拟器及素材遵循各自上游许可；本仓库许可见 [LICENSE](LICENSE)。
