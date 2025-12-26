#!/usr/bin/env python3
"""
修复USD文件中的双重articulation root问题

问题：MJCF转USD后，同时存在：
  - /jiyuan/worldBody (空的articulation root)
  - /jiyuan/base_link/base_link (实际的机器人articulation root)

解决方案：移除worldBody的ArticulationRootAPI，只保留base_link/base_link的
"""

import sys
from pathlib import Path
from pxr import Usd, UsdPhysics

def fix_double_articulation_root(usd_path: str):
    """修复USD文件中的双重articulation root"""

    print(f"正在修复: {usd_path}")

    # 打开USD文件
    stage = Usd.Stage.Open(usd_path)

    # 查找所有articulation roots
    articulation_roots = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(prim)

    print(f"\n找到 {len(articulation_roots)} 个ArticulationRootAPI:")
    for prim in articulation_roots:
        print(f"  - {prim.GetPath()}")

    if len(articulation_roots) != 2:
        print(f"\n⚠️  警告: 期望找到2个articulation roots，但实际找到{len(articulation_roots)}个")
        print("   请手动检查USD文件结构")
        return False

    # 找到worldBody和base_link
    world_body = None
    base_link = None

    for prim in articulation_roots:
        path_str = str(prim.GetPath())
        if 'worldBody' in path_str:
            world_body = prim
        elif 'base_link' in path_str:
            base_link = prim

    if not world_body or not base_link:
        print("\n✗ 错误: 未找到worldBody或base_link")
        return False

    print(f"\n识别到:")
    print(f"  worldBody: {world_body.GetPath()}")
    print(f"  base_link: {base_link.GetPath()}")

    # 移除worldBody的ArticulationRootAPI
    print(f"\n移除 {world_body.GetPath()} 的ArticulationRootAPI...")
    world_body.RemoveAPI(UsdPhysics.ArticulationRootAPI)

    # 验证
    remaining_roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]

    if len(remaining_roots) == 1:
        print(f"\n✓ 成功! 现在只有1个articulation root: {remaining_roots[0].GetPath()}")

        # 保存文件
        print(f"\n保存到: {usd_path}")
        stage.Save()
        print("✓ 文件已保存")
        return True
    else:
        print(f"\n✗ 错误: 修复后仍有{len(remaining_roots)}个articulation roots")
        return False

def main():
    usd_path = "assets/usd/jiyuan/jiyuan.usd"

    if not Path(usd_path).exists():
        print(f"✗ 错误: USD文件不存在: {usd_path}")
        sys.exit(1)

    print("=" * 80)
    print("修复USD文件 - 移除双重Articulation Root")
    print("=" * 80)

    success = fix_double_articulation_root(usd_path)

    print("\n" + "=" * 80)
    if success:
        print("✓ 修复完成")
        print("\n下一步: 运行训练测试验证修复")
        print("  make train-test")
    else:
        print("✗ 修复失败")
        sys.exit(1)
    print("=" * 80)

if __name__ == "__main__":
    main()
