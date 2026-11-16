"""最小仿真循环验证：创建 RoboCasa 厨房环境，随机动作 rollout，保存视频。

这是具身安全系统的第一个地基脚本——验证「仿真环境 → 相机渲染 → 图像/视频输出」
整条管线。后续攻击注入（attacks/）与防御管线（defenses/）都将挂在这条管线上。

用法：
    D:/ruangong/conda-envs/robocasa/python.exe run_kitchen_scene.py
"""

from pathlib import Path

import gymnasium as gym
import robocasa  # noqa: F401  # 注册 robocasa 环境
from robocasa.utils.env_utils import run_random_rollouts

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

ENV_ID = "robocasa/PickPlaceCounterToCabinet"  # 厨房任务：从台面抓放到橱柜
VIDEO_PATH = OUTPUT_DIR / "kitchen_rollout.mp4"


def main() -> None:
    print(f"[1/3] 创建环境: {ENV_ID} ...")
    env = gym.make(
        ENV_ID,
        split="pretrain",  # pretrain 厨房场景与物体集
        seed=0,            # 固定种子保证可复现
    )
    print(f"      成功。观测空间: {type(env.observation_space).__name__}, "
          f"动作空间: {env.action_space}")

    print("[2/3] 随机动作 rollout（100 步）...")
    run_random_rollouts(
        env,
        num_rollouts=1,
        num_steps=100,
        video_path=str(VIDEO_PATH),
    )

    print(f"[3/3] 完成。视频已保存: {VIDEO_PATH}")
    size_mb = VIDEO_PATH.stat().st_size / 1024 / 1024
    print(f"      文件大小: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
