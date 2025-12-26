#!/usr/bin/env python3
"""检查USD文件中的articulation root结构"""

from pxr import Usd, UsdPhysics

stage = Usd.Stage.Open('assets/usd/jiyuan/jiyuan.usd')

# 查找所有有ArticulationRootAPI的prim
print('=' * 80)
print('Prims with ArticulationRootAPI:')
print('=' * 80)
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        print(f'  ✓ {prim.GetPath()}')

# 查看顶层结构
print('\n' + '=' * 80)
print('USD Structure (first 3 levels):')
print('=' * 80)
root = stage.GetPseudoRoot()

for child in root.GetAllChildren():
    print(f'\n/{child.GetName()} (type: {child.GetTypeName()})')

    for subchild in child.GetAllChildren():
        has_api = subchild.HasAPI(UsdPhysics.ArticulationRootAPI)
        marker = ' *** ARTICULATION ROOT ***' if has_api else ''
        print(f'  ├── {subchild.GetName()} (type: {subchild.GetTypeName()}){marker}')

        # 只显示前5个子元素
        subchildren = list(subchild.GetAllChildren())
        for i, subsubchild in enumerate(subchildren[:5]):
            has_api2 = subsubchild.HasAPI(UsdPhysics.ArticulationRootAPI)
            marker2 = ' *** ARTICULATION ROOT ***' if has_api2 else ''
            is_last = (i == len(subchildren) - 1) or (i == 4)
            prefix = '      └──' if is_last else '      ├──'
            print(f'{prefix} {subsubchild.GetName()} (type: {subsubchild.GetTypeName()}){marker2}')

        if len(subchildren) > 5:
            print(f'      ... and {len(subchildren) - 5} more children')

print('\n' + '=' * 80)
print('Summary:')
print('=' * 80)
articulation_roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
print(f'Total ArticulationRootAPI prims: {len(articulation_roots)}')
if len(articulation_roots) > 1:
    print('⚠️  WARNING: Multiple articulation roots found!')
    print('   Isaac Lab expects exactly ONE articulation root per robot.')
elif len(articulation_roots) == 1:
    print('✓ Correct: Exactly one articulation root found.')
else:
    print('✗ ERROR: No articulation root found!')
