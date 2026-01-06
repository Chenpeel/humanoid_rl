"""检查检查点的参数形状"""

import sys

from flax import serialization

checkpoint_path = sys.argv[1]

with open(checkpoint_path, "rb") as f:
    data = serialization.msgpack_restore(f.read())


def print_shapes(d, prefix=""):
    if isinstance(d, dict):
        for k, v in d.items():
            print_shapes(v, prefix + "/" + k)
    elif hasattr(d, "shape"):
        print(f"{prefix}: {d.shape}")


print("=== 检查点信息 ===")
print(f"Step: {data.get('step', 'N/A')}")
print("\n=== 参数形状 ===")
print_shapes(data["params"])
