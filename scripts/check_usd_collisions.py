#!/usr/bin/env python3
"""检查USD中的body和collision结构"""

from pxr import Usd, UsdPhysics, UsdGeom

stage = Usd.Stage.Open('assets/usd/jiyuan/jiyuan.usd')

print('=' * 80)
print('查找所有包含 "foot" 或 "toe" 的 bodies:')
print('=' * 80)

foot_bodies = []
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if ('foot' in path.lower() or 'toe' in path.lower()):
        if prim.GetTypeName() == 'Xform':
            print(f'\n{path}')
            print(f'  Type: {prim.GetTypeName()}')

            # 检查是否有RigidBodyAPI
            has_rigid_body = prim.HasAPI(UsdPhysics.RigidBodyAPI)
            print(f'  HasRigidBodyAPI: {has_rigid_body}')

            # 检查是否有CollisionAPI
            has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
            print(f'  HasCollisionAPI: {has_collision}')

            # 查看子元素
            children = list(prim.GetChildren())
            if children:
                print(f'  Children ({len(children)}):')
                for child in children[:10]:  # 只显示前10个
                    child_type = child.GetTypeName()
                    child_name = child.GetName()
                    print(f'    - {child_name} (type: {child_type})')

                    # 检查子元素的collision
                    if 'collision' in child_name.lower():
                        has_coll_api = child.HasAPI(UsdPhysics.CollisionAPI)
                        print(f'      HasCollisionAPI: {has_coll_api}')

                        # 检查是否有geometry
                        if child.GetTypeName() == 'Mesh':
                            mesh = UsdGeom.Mesh(child)
                            points = mesh.GetPointsAttr().Get()
                            if points:
                                print(f'      Mesh vertices: {len(points)}')

            foot_bodies.append(path)

print('\n' + '=' * 80)
print(f'总共找到 {len(foot_bodies)} 个foot/toe bodies')
print('=' * 80)

# 检查一个典型的body结构
print('\n' + '=' * 80)
print('检查典型body结构（base_link）:')
print('=' * 80)
base_link = stage.GetPrimAtPath('/jiyuan/base_link/base_link')
if base_link:
    print(f'base_link type: {base_link.GetTypeName()}')
    print(f'HasRigidBodyAPI: {base_link.HasAPI(UsdPhysics.RigidBodyAPI)}')
    print(f'HasCollisionAPI: {base_link.HasAPI(UsdPhysics.CollisionAPI)}')

    children = list(base_link.GetChildren())
    print(f'Children ({len(children)}):')
    for child in children[:5]:
        print(f'  - {child.GetName()} (type: {child.GetTypeName()})')
