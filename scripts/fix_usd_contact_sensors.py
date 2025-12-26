#!/usr/bin/env python3
"""
为USD中的foot和toe bodies添加CollisionAPI

这是为了让Isaac Lab的ContactSensor能够正常工作。
"""

from pxr import Usd, UsdPhysics

def add_collision_api_to_bodies(usd_path: str):
    """为指定的bodies添加CollisionAPI"""

    print(f"打开USD文件: {usd_path}")
    stage = Usd.Stage.Open(usd_path)

    # 需要添加CollisionAPI的body路径
    target_bodies = [
        '/jiyuan/base_link/right_foot_link',
        '/jiyuan/base_link/left_foot_link',
        '/jiyuan/base_link/right_toe_link',
        '/jiyuan/base_link/left_toe_link',
    ]

    print("\n" + "=" * 80)
    print("为以下bodies添加CollisionAPI:")
    print("=" * 80)

    modified_count = 0
    for body_path in target_bodies:
        prim = stage.GetPrimAtPath(body_path)

        if not prim.IsValid():
            print(f"✗ 未找到: {body_path}")
            continue

        # 检查是否已有CollisionAPI
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            print(f"  跳过 {body_path} (已有CollisionAPI)")
            continue

        # 添加CollisionAPI
        UsdPhysics.CollisionAPI.Apply(prim)
        print(f"✓ 已添加CollisionAPI: {body_path}")
        modified_count += 1

    print("\n" + "=" * 80)
    print(f"修改了 {modified_count} 个bodies")
    print("=" * 80)

    # 保存文件
    if modified_count > 0:
        print(f"\n保存到: {usd_path}")
        stage.Save()
        print("✓ 文件已保存")

        # 验证
        print("\n验证修改:")
        stage = Usd.Stage.Open(usd_path)
        for body_path in target_bodies:
            prim = stage.GetPrimAtPath(body_path)
            if prim.IsValid():
                has_api = prim.HasAPI(UsdPhysics.CollisionAPI)
                status = "✓" if has_api else "✗"
                print(f"  {status} {body_path}: HasCollisionAPI = {has_api}")
    else:
        print("\n没有需要修改的内容")

    return modified_count

def main():
    usd_path = "assets/usd/jiyuan/jiyuan.usd"

    print("=" * 80)
    print("为foot和toe bodies添加CollisionAPI")
    print("=" * 80)
    print()

    modified = add_collision_api_to_bodies(usd_path)

    print("\n" + "=" * 80)
    if modified > 0:
        print("✓ 修复完成")
        print("\n下一步: 重新启用contact_forces并测试训练")
        print("  1. 编辑 jiyuan_scene_cfg.py，取消注释 contact_forces")
        print("  2. 运行 make train-test")
    else:
        print("✓ 已经正确配置")
    print("=" * 80)

if __name__ == "__main__":
    main()
