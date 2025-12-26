#!/usr/bin/env python3
"""
MJCF Modular Tool - Split and Merge MuJoCo XML Files

Splits MJCF models into modular structure for easier editing:
- params/: Frequently modified physical parameters
- geometry/: Stable geometric definitions
- Functional modules: actuators, sensors, constraints, etc.

Usage:
    python xml_modular_tool.py split <input.xml> -o <output_dir>
    python xml_modular_tool.py merge <input_dir> -o <output.xml>
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
import sys


class MJCFModularSplitter:
    """Splits MJCF into parameter and geometry modules"""

    def __init__(self, input_file: Path, output_dir: Path):
        self.input_file = input_file
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        (self.output_dir / 'geometry').mkdir(exist_ok=True)
        (self.output_dir / 'params').mkdir(exist_ok=True)

        self.tree = ET.parse(input_file)
        self.root = self.tree.getroot()

    def split(self):
        """Execute modular split"""
        print(f"[Split] {self.input_file} → {self.output_dir}/\n")

        print("📝 Parameters:")
        self._extract_default_classes()
        self._extract_materials()
        self._extract_tendons()
        self._extract_friction()

        print("\n🔧 Geometry:")
        self._extract_assets()
        self._extract_world()
        self._extract_robot()

        print("\n⚙️  Modules:")
        self._extract_constraints()
        self._extract_actuators()
        self._extract_sensors()
        self._extract_contact()

        self._generate_index()
        self._generate_readme()

        print(f"\n✓ Complete: {self.output_dir / 'index.xml'}")

    def _extract_default_classes(self):
        elem = self.root.find('default')
        if elem is not None:
            root = ET.Element('mujoco')
            root.append(ET.Comment(' Default classes '))
            root.append(self._copy(elem))
            self._write('params/default_classes.xml', root)
            print("  ✓ default_classes.xml")

    def _extract_materials(self):
        asset = self.root.find('asset')
        if asset is None:
            return

        root = ET.Element('mujoco')
        root.append(ET.Comment(' Materials '))
        asset_new = ET.SubElement(root, 'asset')

        for mat in asset.findall('material'):
            asset_new.append(self._copy(mat))

        if len(asset_new) > 0:
            self._write('params/materials.xml', root)
            print("  ✓ materials.xml")

    def _extract_tendons(self):
        elem = self.root.find('tendon')
        if elem is not None:
            root = ET.Element('mujoco')
            root.append(ET.Comment(' Tendon parameters '))
            root.append(self._copy(elem))
            self._write('params/tendons.xml', root)
            print("  ✓ tendons.xml")

    def _extract_friction(self):
        """Extract ground plane geometry (if exists)"""
        wb = self.root.find('worldbody')
        if wb is None:
            return

        # Find ground plane geometry
        ground_geom = None
        for geom in wb.findall('geom'):
            if geom.get('type') == 'plane':
                ground_geom = geom
                break

        # Only create file if ground exists
        if ground_geom is not None:
            root = ET.Element('mujoco')
            root.append(ET.Comment(' Ground plane '))
            wb_new = ET.SubElement(root, 'worldbody')
            wb_new.append(self._copy(ground_geom))
            self._write('params/ground.xml', root)
            print("  ✓ ground.xml")


    def _extract_assets(self):
        asset = self.root.find('asset')
        if asset is None:
            return

        root = ET.Element('mujoco')
        root.append(ET.Comment(' Mesh assets '))
        asset_new = ET.SubElement(root, 'asset')

        for mesh in asset.findall('mesh'):
            mesh_copy = self._copy(mesh)
            # Fix mesh paths: geometry/meshes.xml → assets/meshes/
            # Path from assets/xmls/models/jiyuan/geometry/meshes.xml to assets/meshes/
            # is ../../../../meshes/
            file_path = mesh_copy.get('file')
            if file_path:
                # Extract just the filename from the path
                filename = Path(file_path).name
                # Set unified path: from geometry/meshes.xml to assets/meshes/
                mesh_copy.set('file', f"../../../../meshes/{filename}")
            asset_new.append(mesh_copy)

        if len(asset_new) > 0:
            self._write('geometry/meshes.xml', root)
            print("  ✓ meshes.xml")

    def _extract_world(self):
        wb = self.root.find('worldbody')
        if wb is None:
            return

        root = ET.Element('mujoco')
        root.append(ET.Comment(' World environment '))
        world = ET.SubElement(root, 'worldbody')

        for light in wb.findall('light'):
            world.append(self._copy(light))

        for geom in wb.findall('geom'):
            if geom.get('type') == 'plane':
                world.append(self._copy(geom))

        self._write('geometry/world.xml', root)
        print("  ✓ world.xml")

    def _extract_robot(self):
        wb = self.root.find('worldbody')
        if wb is None:
            return

        base = wb.find(".//body[@name='base_link']")
        if base is None:
            return

        # Base content - extract direct children (not body elements)
        root = ET.Element('mujoco')
        for child in base:
            if child.tag != 'body':
                root.append(self._copy(child))
        self._write('geometry/base.xml', root)
        print("  ✓ base.xml")

        # Legs
        for side in ['right', 'left']:
            self._extract_leg(base, side)

        # Robot assembly
        root = ET.Element('mujoco')
        wb_new = ET.SubElement(root, 'worldbody')
        robot = ET.SubElement(wb_new, 'body')
        robot.set('name', 'base_link')
        robot.set('pos', base.get('pos', '0 0 0.9915'))
        robot.set('quat', base.get('quat', '0.70710678 0.70710678 0 0'))
        robot.set('childclass', base.get('childclass', 'robot'))

        # IMPORTANT: Include paths are relative to index.xml, not robot.xml
        for name, file in [('Base', 'geometry/base.xml'), ('Right leg', 'geometry/right_leg.xml'), ('Left leg', 'geometry/left_leg.xml')]:
            robot.append(ET.Comment(f' {name} '))
            inc = ET.SubElement(robot, 'include')
            inc.set('file', file)

        self._write('geometry/robot.xml', root)
        print("  ✓ robot.xml")

    def _extract_leg(self, base, side: str):
        root = ET.Element('mujoco')

        for name in [f'{side}_hip_pitch_engine_link', f'{side}_hip_yaw_engine_link', f'{side}_hip_roll_link']:
            body = base.find(f"./body[@name='{name}']")
            if body is not None:
                root.append(self._copy(body))

        self._write(f'geometry/{side}_leg.xml', root)
        print(f"  ✓ {side}_leg.xml")

    def _extract_constraints(self):
        elem = self.root.find('equality')
        if elem is not None:
            root = ET.Element('mujoco')
            root.append(ET.Comment(' Equality constraints '))
            root.append(self._copy(elem))
            self._write('constraints.xml', root)
            print("  ✓ constraints.xml")

    def _extract_actuators(self):
        elem = self.root.find('actuator')
        if elem is not None:
            root = ET.Element('mujoco')
            root.append(ET.Comment(' Actuators '))
            root.append(self._copy(elem))
            self._write('actuators.xml', root)
            print("  ✓ actuators.xml")

    def _extract_sensors(self):
        elem = self.root.find('sensor')
        if elem is not None:
            root = ET.Element('mujoco')
            root.append(ET.Comment(' Sensors '))
            root.append(self._copy(elem))
            self._write('sensors.xml', root)
            print("  ✓ sensors.xml")

    def _extract_contact(self):
        elem = self.root.find('contact')
        if elem is not None:
            root = ET.Element('mujoco')
            root.append(ET.Comment(' Contact exclusions '))
            root.append(self._copy(elem))
            self._write('contact_exclude.xml', root)
            print("  ✓ contact_exclude.xml")

    def _generate_index(self):
        root = ET.Element('mujoco')
        root.set('model', self.root.get('model', 'robot'))

        # Visual and compiler
        for tag in ['visual', 'compiler']:
            elem = self.root.find(tag)
            if elem is not None:
                root.append(self._copy(elem))

        # Parameters
        root.append(ET.Comment(' === Parameters === '))
        for f in ['default_classes.xml', 'materials.xml', 'ground.xml', 'tendons.xml']:
            if (self.output_dir / 'params' / f).exists():
                inc = ET.SubElement(root, 'include')
                inc.set('file', f'params/{f}')

        # Geometry
        root.append(ET.Comment(' === Geometry === '))
        for f in ['meshes.xml', 'world.xml', 'robot.xml']:
            inc = ET.SubElement(root, 'include')
            inc.set('file', f'geometry/{f}')

        # Modules
        root.append(ET.Comment(' === Modules === '))
        for f in ['constraints.xml', 'actuators.xml', 'sensors.xml', 'contact_exclude.xml']:
            if (self.output_dir / f).exists():
                inc = ET.SubElement(root, 'include')
                inc.set('file', f)

        self._write('index.xml', root)

    def _generate_readme(self):
        content = """# MJCF Modular Model

## Structure

```
.
├── index.xml           # Entry point (load this)
├── params/             # Frequently modified
│   ├── default_classes.xml
│   ├── materials.xml
│   ├── friction.xml
│   └── tendons.xml
├── geometry/           # Rarely modified
│   ├── meshes.xml
│   ├── world.xml
│   ├── robot.xml
│   ├── base.xml
│   └── *_leg.xml
└── *.xml               # Functional modules
```

## Usage

```python
import mujoco as mj
model = mj.MjModel.from_xml_path('index.xml')
```

## Common Modifications

### Friction
Edit `params/friction.xml`:
```xml
<ground_params friction="1.0 0.005 0.0001" />
```

### Tendons
Edit `params/tendons.xml`:
```xml
<spatial ... stiffness="8000" damping="100" />
```

### Materials
Edit `params/materials.xml`:
```xml
<material name="..." rgba="R G B A" />
```
"""
        (self.output_dir / 'README.md').write_text(content, encoding='utf-8')

    def _copy(self, elem):
        """Deep copy element"""
        new = ET.Element(elem.tag, elem.attrib)
        new.text = elem.text
        new.tail = elem.tail
        for child in elem:
            new.append(self._copy(child))
        return new

    def _write(self, filename: str, root: ET.Element):
        """Write formatted XML"""
        self._indent(root)
        tree = ET.ElementTree(root)
        tree.write(self.output_dir / filename, encoding='utf-8', xml_declaration=True)

    def _indent(self, elem, level=0):
        """Format indentation"""
        i = "\n" + "  " * level
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for child in elem:
                self._indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i


class MJCFModularMerger:
    """Merges modular MJCF back to single file"""

    def __init__(self, module_dir: Path, output_file: Path):
        self.module_dir = module_dir
        self.output_file = output_file
        self.cache = {}

    def merge(self):
        """Execute merge"""
        print(f"[Merge] {self.module_dir}/ → {self.output_file}")

        index = self.module_dir / 'index.xml'
        if not index.exists():
            print(f"Error: {index} not found")
            sys.exit(1)

        tree = ET.parse(index)
        root = tree.getroot()

        print(f"[DEBUG] Before processing: {len(list(root.findall('.//include')))} include tags")

        self._process_includes(root)

        print(f"[DEBUG] After processing: {len(list(root.findall('.//include')))} include tags")

        # 修正mesh路径：从 ../../../../meshes/ 改为 ../../meshes/
        # 因为合并后的文件在 assets/xmls/models/jiyuan.xml
        # 而不是 assets/xmls/models/jiyuan/geometry/meshes.xml
        self._fix_mesh_paths(root)

        print(f"[DEBUG] After fixing mesh paths")

        # 最终检查：在写入前验证没有include标签
        final_includes = list(root.findall('.//include'))
        if final_includes:
            print(f"[ERROR] Found {len(final_includes)} include tags after processing!")
            for inc in final_includes:
                print(f"  - file=\"{inc.get('file')}\"")
                # 查找父元素
                for p in root.iter():
                    if inc in list(p):
                        print(f"    parent: {p.tag} name=\"{p.get('name', '')}\"")
                        break

        self._indent(root)

        print(f"[DEBUG] After indent: {len(list(root.findall('.//include')))} include tags")

        ET.ElementTree(root).write(self.output_file, encoding='utf-8', xml_declaration=True)

        # 验证写入的文件
        verify_tree = ET.parse(self.output_file)
        verify_root = verify_tree.getroot()
        verify_includes = list(verify_root.findall('.//include'))
        print(f"[DEBUG] Verification: {len(verify_includes)} include tags in written file")
        if verify_includes:
            print(f"[ERROR] Found include tags in written file:")
            for inc in verify_includes:
                print(f"  - file=\"{inc.get('file')}\"")

        print(f"✓ Complete: {self.output_file}")

    def _process_includes(self, elem, current_dir: Path = None):
        """Recursively process includes in element and all descendants

        使用深度优先遍历，确保所有层级的include标签都被展开。

        关键设计：所有include路径都相对于module_dir（index.xml所在目录），
        即使这些include标签嵌套在已经被include进来的文件中。
        """
        if current_dir is None:
            current_dir = self.module_dir

        # 使用while循环持续处理，直到没有include标签为止
        # 这确保了多层嵌套的include都会被展开
        max_iterations = 100  # 防止无限循环
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # 查找所有include标签（包括嵌套的）
            includes = list(elem.findall('.//include'))

            if not includes:
                # 没有找到include标签，处理完成
                break

            print(f"[DEBUG] Iteration {iteration}: Found {len(includes)} include tags")

            # 处理第一个include标签
            inc = includes[0]
            file_attr = inc.get('file')

            if not file_attr:
                print(f"[DEBUG] Include tag has no 'file' attribute, removing it")
                # 找到父元素并移除
                for p in elem.iter():
                    if inc in list(p):
                        p.remove(inc)
                        break
                continue

            # 找到include的父元素
            parent = None
            for p in elem.iter():
                if inc in list(p):
                    parent = p
                    break

            if parent is None:
                print(f"[DEBUG] Could not find parent for include tag")
                break

            # 所有include路径都相对于module_dir
            # 因为分割工具生成的include路径是相对于index.xml的
            inc_path = self.module_dir / file_attr

            print(f"[DEBUG] Processing include: {file_attr} -> {inc_path}")

            if not inc_path.exists():
                print(f"Warning: {inc_path} not found, removing include tag")
                parent.remove(inc)
                continue

            # Load and extract content
            inc_tree = self._load(inc_path)
            inc_root = inc_tree.getroot()

            children = list(inc_root) if inc_root.tag == 'mujoco' else [inc_root]

            print(f"[DEBUG] Loaded {len(children)} children from {file_attr}")

            # Replace include
            idx = list(parent).index(inc)
            parent.remove(inc)
            for i, child in enumerate(children):
                parent.insert(idx + i, child)

            print(f"[DEBUG] Replaced include tag with {len(children)} children")

        if iteration >= max_iterations:
            print(f"Warning: Reached maximum iterations ({max_iterations}), may have circular includes")
        else:
            print(f"[DEBUG] Include processing completed after {iteration} iterations")

        # 调试：检查是否还有include标签
        remaining_includes = list(elem.findall('.//include'))
        if remaining_includes:
            print(f"[WARNING] After processing, still found {len(remaining_includes)} include tags:")
            for inc in remaining_includes:
                print(f"  - file=\"{inc.get('file')}\"")
                # 查找父元素
                for p in elem.iter():
                    if inc in list(p):
                        print(f"    parent: {p.tag} name=\"{p.get('name', '')}\"")
                        break

    def _fix_mesh_paths(self, root):
        """修正合并后的mesh路径

        分割工具生成的mesh路径是从 assets/xmls/models/jiyuan/geometry/meshes.xml
        到 assets/meshes/ 的相对路径（../../../../meshes/）。

        但合并后的文件在 assets/xmls/models/jiyuan.xml，
        所以需要改为 ../../meshes/
        """
        meshes = root.findall('.//mesh')
        fixed_count = 0

        for mesh in meshes:
            file_path = mesh.get('file')
            if file_path and file_path.startswith('../../../../meshes/'):
                # 替换为正确的相对路径
                new_path = file_path.replace('../../../../meshes/', '../../meshes/')
                mesh.set('file', new_path)
                fixed_count += 1

        if fixed_count > 0:
            print(f"[DEBUG] Fixed {fixed_count} mesh paths")

    def _load(self, path: Path):
        """Load module with cache"""
        if path not in self.cache:
            self.cache[path] = ET.parse(path)
        return self.cache[path]

    def _indent(self, elem, level=0):
        """Format indentation"""
        i = "\n" + "  " * level
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for child in elem:
                self._indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i


def main():
    parser = argparse.ArgumentParser(
        description='MJCF Modular Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python xml_modular_tool.py split model.xml -o model_dir
  python xml_modular_tool.py merge model_dir -o model.xml
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command')

    split = subparsers.add_parser('split', help='Split XML into modules')
    split.add_argument('input', type=str, help='Input XML file')
    split.add_argument('-o', '--output', type=str, required=True, help='Output directory')

    merge = subparsers.add_parser('merge', help='Merge modules into XML')
    merge.add_argument('input', type=str, help='Module directory')
    merge.add_argument('-o', '--output', type=str, required=True, help='Output XML file')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == 'split':
        input_file = Path(args.input)
        output_dir = Path(args.output)
        if not input_file.exists():
            print(f"Error: {input_file} not found")
            sys.exit(1)
        splitter = MJCFModularSplitter(input_file, output_dir)
        splitter.split()

    elif args.command == 'merge':
        module_dir = Path(args.input)
        output_file = Path(args.output)
        if not module_dir.exists():
            print(f"Error: {module_dir} not found")
            sys.exit(1)
        merger = MJCFModularMerger(module_dir, output_file)
        merger.merge()


if __name__ == '__main__':
    main()
