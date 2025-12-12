# Utility Tools

Unified command-line tools for robot model conversion and manipulation.

## Package Structure

All tools follow consistent patterns with proper module structure:

```
utils/
├── __init__.py                 # Package entry
├── README.md                   # This file
├── xml_tools/
│   ├── __init__.py            # Module exports
│   └── xml_modular_tool.py    # CLI: split/merge MJCF
├── urdf2mjcf/
│   ├── __init__.py            # Module exports
│   ├── convert_tool.py        # CLI: URDF→MJCF
│   └── mirror_tool.py         # CLI: MJCF mirroring
└── urdf2ros2/
    ├── __init__.py            # Module exports
    ├── mirror_tool.py         # CLI: URDF mirroring
    └── ros2_migrate.py        # CLI: ROS1→ROS2 migration
```

Each tool can be used as:
- **Standalone script**: `python utils/xml_tools/xml_modular_tool.py`
- **Module import**: `from utils.xml_tools import MJCFModularSplitter`

---

## 1. XML Modular Tool

**Purpose**: Split/merge MJCF for easier parameter editing.

### Commands

**Split** - Create modular structure:
```bash
python utils/xml_tools/xml_modular_tool.py split <input.xml> -o <output_dir>
```

**Merge** - Combine back to single file:
```bash
python utils/xml_tools/xml_modular_tool.py merge <input_dir> -o <output.xml>
```

### Example

```bash
# Split mirrored.xml into modular structure
python utils/xml_tools/xml_modular_tool.py split \
    assets/xmls/mirrored.xml \
    -o assets/xmls/models/jiyuan
```

**Output structure**:
```
assets/xmls/models/jiyuan/
├── index.xml              # Load this file
├── params/                # Frequently modified
│   ├── default_classes.xml
│   ├── materials.xml
│   ├── friction.xml       # Ground friction
│   └── tendons.xml        # Tendon stiffness/damping
├── geometry/              # Rarely modified
│   ├── meshes.xml
│   ├── world.xml
│   ├── robot.xml
│   ├── base.xml
│   └── *_leg.xml
├── actuators.xml
├── sensors.xml
├── constraints.xml
└── contact_exclude.xml
```

**Direct loading** (no merge needed):
```python
import mujoco as mj
model = mj.MjModel.from_xml_path('assets/xmls/models/jiyuan/index.xml')
```

---

## 2. URDF Conversion Tool

**Purpose**: Convert URDF to MJCF with automatic mesh path fixing.

### Commands

**Convert**:
```bash
python utils/urdf2mjcf/convert_tool.py convert <input.urdf> -o <output.xml>
```

### Example

```bash
python utils/urdf2mjcf/convert_tool.py convert \
    assets/urdf/robot.urdf \
    -o assets/mjcf/robot.xml
```

**Features**:
- Converts URDF to MJCF format
- Fixes mesh paths to relative format (`../meshes/`)
- Preserves robot structure

---

## 3. MJCF Mirror Tool

**Purpose**: Mirror right leg across YZ plane to create bilateral model.

### Commands

**Mirror**:
```bash
python utils/urdf2mjcf/mirror_tool.py mirror <input.xml> -o <output.xml>
```

### Example

```bash
python utils/urdf2mjcf/mirror_tool.py mirror \
    assets/mjcf/right_leg.xml \
    -o assets/mjcf/bilateral.xml
```

**What gets mirrored**:
- Body positions and orientations
- Joint axes
- Mesh definitions (with scale="-1.0 1.0 1.0")
- Actuators
- Contact exclusions
- All `right_*` names → `left_*`

---

## 4. URDF Mirror Tool

**Purpose**: Mirror URDF models across YZ plane for bilateral robots.

### Commands

**Mirror**:
```bash
python utils/urdf2ros2/mirror_tool.py mirror <input.urdf> -o <output.urdf>
```

### Example

```bash
python utils/urdf2ros2/mirror_tool.py mirror \
    assets/urdf/right_leg.urdf \
    -o assets/urdf/bilateral.urdf
```

**What gets mirrored**:
- Link positions (X coordinate negated)
- Joint origins (RPY → quaternion → mirror → RPY)
- Joint axes (X component negated)
- Mesh references (`right_` → `left_`)
- All `right_*` names → `left_*`

---

## 5. ROS2 Migration Tool

**Purpose**: Convert ROS1 packages to ROS2 format.

### Commands

**Convert**:
```bash
python utils/urdf2ros2/ros2_migrate.py convert <package_path>
```

### Example

```bash
python utils/urdf2ros2/ros2_migrate.py convert \
    ~/catkin_ws/src/my_robot_description
```

**What gets converted**:
- `package.xml`: catkin → ament_cmake
- `CMakeLists.txt`: Generated for ROS2
- Dependencies updated to ROS2 packages
- Launch files flagged for manual conversion

---

## Complete Workflows

### Workflow 1: URDF → Bilateral MJCF → Modular

```bash
# Step 1: Convert URDF to MJCF (right leg only)
python utils/urdf2mjcf/convert_tool.py convert \
    robot.urdf -o right_leg.xml

# Step 2: Mirror to create bilateral model
python utils/urdf2mjcf/mirror_tool.py mirror \
    right_leg.xml -o bilateral.xml

# Step 3: Split for easy parameter editing
python utils/xml_tools/xml_modular_tool.py split \
    bilateral.xml -o models/robot
```

### Workflow 2: Edit Parameters → Deploy

```bash
# Edit parameters
nano models/robot/params/friction.xml
nano models/robot/params/tendons.xml

# Load directly (no merge needed)
python scripts/train.py --xml models/robot/index.xml

# (Optional) Merge for deployment
python utils/xml_tools/xml_modular_tool.py merge \
    models/robot -o deploy.xml
```

---

## Module Usage (Python API)

### XML Tools
```python
from utils.xml_tools import MJCFModularSplitter, MJCFModularMerger
from pathlib import Path

# Split
splitter = MJCFModularSplitter(
    Path('input.xml'),
    Path('output_dir')
)
splitter.split()

# Merge
merger = MJCFModularMerger(
    Path('input_dir'),
    Path('output.xml')
)
merger.merge()
```

### URDF Conversion
```python
from utils.urdf2mjcf import convert_urdf, fix_mesh_paths
from pathlib import Path

convert_urdf(Path('robot.urdf'), Path('robot.xml'))
fix_mesh_paths(Path('robot.xml'))
```

### MJCF Mirroring
```python
from utils.urdf2mjcf import mirror_mjcf
from pathlib import Path

mirror_mjcf(Path('right_leg.xml'), Path('bilateral.xml'))
```

### URDF Mirroring
```python
from utils.urdf2ros2 import mirror_urdf
from pathlib import Path

mirror_urdf(Path('right_leg.urdf'), Path('bilateral.urdf'))
```

### ROS2 Migration
```python
from utils.urdf2ros2 import migrate_package
from pathlib import Path

migrate_package(Path('~/catkin_ws/src/my_robot'))
```

---

## Tool Design Principles

1. **Consistent Interface**: All tools use `command input -o output` pattern
2. **Module Structure**: Each tool has `__init__.py` for imports
3. **Standalone Ready**: Can run directly or import as module
4. **Clear Output**: Uses emoji prefixes (📝 📋 🔧 ⚙️ ✓) for progress
5. **No Special Terms**: Generic naming (not robot-specific)

---

## Testing

Test each tool with `--help`:
```bash
python utils/xml_tools/xml_modular_tool.py --help
python utils/urdf2mjcf/convert_tool.py --help
python utils/urdf2mjcf/mirror_tool.py --help
python utils/urdf2ros2/mirror_tool.py --help
python utils/urdf2ros2/ros2_migrate.py --help
```

Test complete workflow:
```bash
# Create test directory
mkdir -p /tmp/test_tools

# Test XML split
python utils/xml_tools/xml_modular_tool.py split \
    assets/xmls/mirrored.xml -o /tmp/test_tools/split

# Test XML merge
python utils/xml_tools/xml_modular_tool.py merge \
    /tmp/test_tools/split -o /tmp/test_tools/merged.xml

# Compare
diff assets/xmls/mirrored.xml /tmp/test_tools/merged.xml
```

---

## Requirements

- Python >= 3.11
- `urdf2mjcf` (for URDF conversion): `pip install urdf2mjcf`
- No other dependencies (uses stdlib only)

---

## Notes

- **No merge required**: MuJoCo loads modular `index.xml` directly
- **Edit freely**: Modify `params/*.xml` files as needed
- **Version control friendly**: Modular structure is easier to track
- **Portable**: All paths are relative to project root
