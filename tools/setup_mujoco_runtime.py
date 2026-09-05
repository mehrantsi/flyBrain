from __future__ import annotations

import glob
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV = PROJECT_ROOT / "work/upstream/flygym/.venv"
RUNTIME = PROJECT_ROOT / "work/mujoco/lib"


def find_library() -> Path:
    candidates = glob.glob(
        str(VENV / "lib/python*/site-packages/mujoco/libmujoco.3.9.0.dylib")
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one MuJoCo 3.9.0 library under {VENV}, found {len(candidates)}"
        )
    return Path(candidates[0]).resolve()


def find_glfw() -> Path:
    candidates = glob.glob(str(VENV / "lib/python*/site-packages/glfw/libglfw.3.dylib"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one GLFW library under {VENV}, found {len(candidates)}"
        )
    return Path(candidates[0]).resolve()


def replace_symlink(path: Path, target: Path) -> None:
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        raise FileExistsError(f"refusing to replace non-symlink: {path}")
    path.symlink_to(os.path.relpath(target, path.parent))


def main() -> int:
    library = find_library()
    glfw = find_glfw()
    framework = RUNTIME / "mujoco.framework/Versions/A"
    framework.mkdir(parents=True, exist_ok=True)
    replace_symlink(RUNTIME / "libmujoco.dylib", library)
    replace_symlink(RUNTIME / "libglfw.3.dylib", glfw)
    replace_symlink(framework / "libmujoco.3.9.0.dylib", library)
    print(RUNTIME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
