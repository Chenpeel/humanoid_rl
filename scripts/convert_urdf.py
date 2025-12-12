#!/usr/bin/env python3
"""
Standard URDF to Modular MJCF Conversion Pipeline

Pipeline:
1. assets/urdf/*.urdf → MJCF conversion
2. Right leg → Bilateral mirroring
3. Bilateral MJCF → assets/mjcf/
4. MJCF → Modular split → assets/xmls/models/jiyuan/

Usage:
    python scripts/convert_urdf.py <input.urdf>
    python scripts/convert_urdf.py assets/urdf/robot.urdf
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.urdf2mjcf import convert_urdf, fix_mesh_paths
from utils.urdf2mjcf import mirror_mjcf
from utils.xml_tools import MJCFModularSplitter


def run_pipeline(input_urdf: Path):
    """Execute complete conversion pipeline"""

    print("\n" + "="*60)
    print("🚀 URDF to Modular MJCF Conversion Pipeline")
    print("="*60)
    print(f"\nInput: {input_urdf}")

    if not input_urdf.exists():
        print(f"\n❌ Error: Input file not found: {input_urdf}")
        return False

    # Define paths
    temp_mjcf = project_root / "assets" / "mjcf" / "temp_converted.xml"
    bilateral_mjcf = project_root / "assets" / "mjcf" / "mirrored.xml"
    output_dir = project_root / "assets" / "xmls" / "models" / "jiyuan"

    # Ensure directories exist
    temp_mjcf.parent.mkdir(parents=True, exist_ok=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: URDF → MJCF
        print("\n" + "─"*60)
        print("📋 Step 1/3: Converting URDF to MJCF")
        print("─"*60)
        convert_urdf(input_urdf, temp_mjcf)

        if not temp_mjcf.exists():
            print(f"\n❌ Conversion failed: {temp_mjcf} not created")
            return False

        # Step 2: Mirror to bilateral
        print("\n" + "─"*60)
        print("🔄 Step 2/3: Mirroring to bilateral model")
        print("─"*60)
        mirror_mjcf(temp_mjcf, bilateral_mjcf)

        if not bilateral_mjcf.exists():
            print(f"\n❌ Mirroring failed: {bilateral_mjcf} not created")
            return False

        # Clean up temp file
        temp_mjcf.unlink()
        print(f"  ✓ Cleaned up temporary file")

        # Step 3: Split to modular
        print("\n" + "─"*60)
        print("📦 Step 3/3: Splitting to modular structure")
        print("─"*60)
        splitter = MJCFModularSplitter(bilateral_mjcf, output_dir)
        splitter.split()

        # Verify output
        index_file = output_dir / "index.xml"
        if not index_file.exists():
            print(f"\n❌ Split failed: {index_file} not created")
            return False

        # Success summary
        print("\n" + "="*60)
        print("✅ Pipeline Complete!")
        print("="*60)
        print(f"\n📁 Output locations:")
        print(f"  Bilateral MJCF: {bilateral_mjcf}")
        print(f"  Modular model:  {output_dir}/")
        print(f"  Entry point:    {index_file}")

        print(f"\n📝 Next steps:")
        print(f"  1. Review the modular structure:")
        print(f"     ls -la {output_dir}")
        print(f"  2. Edit parameters as needed:")
        print(f"     nano {output_dir}/params/friction.xml")
        print(f"     nano {output_dir}/params/tendons.xml")
        print(f"  3. Test loading in MuJoCo:")
        print(f"     python scripts/visualize_mjcf.py")

        return True

    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Standard URDF to Modular MJCF Conversion Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Steps:
  1. URDF → MJCF conversion (with mesh path fixes)
  2. Right leg → Bilateral mirroring
  3. Save bilateral to assets/mjcf/mirrored.xml
  4. Split to modular structure at assets/xmls/models/jiyuan/

Examples:
  python scripts/convert_urdf.py assets/urdf/robot.urdf
  python scripts/convert_urdf.py ~/my_robot/robot.urdf

Output:
  assets/mjcf/mirrored.xml          - Bilateral MJCF
  assets/xmls/models/jiyuan/        - Modular structure
    ├── index.xml                   - Entry point
    ├── params/                     - Editable parameters
    └── geometry/                   - Stable geometry
        """
    )

    parser.add_argument('urdf', type=str, help='Input URDF file path')
    parser.add_argument('--no-mirror', action='store_true',
                       help='Skip mirroring (URDF already bilateral)')

    args = parser.parse_args()

    input_urdf = Path(args.urdf)

    if args.no_mirror:
        print("\n⚠️  Mirroring disabled - assuming bilateral URDF")
        # TODO: Implement direct bilateral conversion
        print("❌ --no-mirror not yet implemented")
        sys.exit(1)

    if not run_pipeline(input_urdf):
        sys.exit(1)


if __name__ == '__main__':
    main()
