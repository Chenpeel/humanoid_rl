# MJCF Modular Model

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
