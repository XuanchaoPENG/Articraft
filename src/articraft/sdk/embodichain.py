from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from pxr import Usd, UsdPhysics

__all__ = ["export_embodichain_usdc"]


def export_embodichain_usdc(
    source: Path | str,
    destination: Path | str,
    *,
    copy_assets: bool = False,
) -> Path:
    """Publish an Articraft stage with the hierarchy expected by EmbodiChain.

    Articraft authors ``ArticulationRootAPI`` on the selected root rigid body
    and keeps ``/World`` as the default prim. DexSim instead discovers an
    articulation by traversing downward from the prim carrying that API, so
    sibling bodies and joints are invisible to it. The compatibility copy makes
    the assembly both the default prim and the articulation root while leaving
    the native USDZ untouched for Articraft's own viewer.

    Files next to the source layer are copied with it so texture references stay
    valid when this function runs inside the exporter staging directory.

    Args:
        source: Source USD, USDA, or USDC stage.
        destination: Destination ``.usdc`` path.
        copy_assets: Copy sibling texture/assets into the output directory.

    Returns:
        The resolved destination path.

    Raises:
        ValueError: If the stage does not contain exactly one articulated assembly.
        RuntimeError: If OpenUSD cannot open or publish the stage.
    """

    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    if source_path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise ValueError("EmbodiChain compatibility input must be USD, USDA, or USDC")
    if destination_path.suffix.lower() != ".usdc":
        raise ValueError("EmbodiChain compatibility output must use the .usdc suffix")

    source_stage = Usd.Stage.Open(str(source_path))
    if source_stage is None:
        raise RuntimeError(f"OpenUSD could not open {source_path}")
    assembly = _single_assembly(source_stage)
    roots = [
        prim for prim in Usd.PrimRange(assembly) if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    if len(roots) != 1:
        raise ValueError(
            "EmbodiChain compatibility export requires exactly one articulation root; "
            f"found {len(roots)}"
        )

    destination_path.parent.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_path.parent.name}.tmp-",
            dir=destination_path.parent.parent,
        )
    )
    temporary_path = temporary_dir / destination_path.name
    try:
        if copy_assets:
            _copy_layer_assets(source_path.parent, temporary_dir, source_path)

        if not source_stage.Export(str(temporary_path)):
            raise RuntimeError(f"OpenUSD could not export {destination_path}")
        compatible_stage = Usd.Stage.Open(str(temporary_path))
        if compatible_stage is None:
            raise RuntimeError(f"OpenUSD could not reopen {temporary_path}")
        compatible_assembly = compatible_stage.GetPrimAtPath(assembly.GetPath())
        compatible_root = compatible_stage.GetPrimAtPath(roots[0].GetPath())
        compatible_root.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        compatible_assembly.ApplyAPI(UsdPhysics.ArticulationRootAPI)
        compatible_stage.SetDefaultPrim(compatible_assembly)
        compatible_stage.GetRootLayer().Save()
        _validate_compatible_stage(temporary_path, assembly.GetPath().pathString)
        if destination_path.parent.exists():
            _publish_into_existing_directory(temporary_dir, destination_path.parent)
        else:
            temporary_dir.replace(destination_path.parent)
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)
    return destination_path


def _single_assembly(stage: Usd.Stage) -> Usd.Prim:
    default_prim = stage.GetDefaultPrim()
    if default_prim and default_prim.GetChild("rigid_bodies"):
        return default_prim
    assemblies = [prim for prim in default_prim.GetChildren() if prim.GetChild("rigid_bodies")]
    if len(assemblies) != 1:
        raise ValueError(
            "EmbodiChain compatibility export requires exactly one rigid-body assembly; "
            f"found {len(assemblies)}"
        )
    return assemblies[0]


def _copy_layer_assets(source_dir: Path, destination_dir: Path, source_layer: Path) -> None:
    for path in source_dir.iterdir():
        if path == source_layer:
            continue
        target = destination_dir / path.name
        if path.is_dir():
            shutil.copytree(path, target)
        elif path.is_file():
            shutil.copy2(path, target)


def _publish_into_existing_directory(source_dir: Path, destination_dir: Path) -> None:
    conflicts = [
        path.name for path in source_dir.iterdir() if (destination_dir / path.name).exists()
    ]
    if conflicts:
        raise FileExistsError(
            "EmbodiChain compatibility output would overwrite existing files: "
            + ", ".join(sorted(conflicts))
        )
    for path in source_dir.iterdir():
        path.replace(destination_dir / path.name)


def _validate_compatible_stage(path: Path, expected_root: str) -> None:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"OpenUSD could not reopen {path}")
    root = stage.GetDefaultPrim()
    if not root or root.GetPath().pathString != expected_root:
        raise RuntimeError("EmbodiChain USDC default prim does not match the assembly")
    if not root.HasAPI(UsdPhysics.ArticulationRootAPI):
        raise RuntimeError("EmbodiChain USDC assembly has no ArticulationRootAPI")
    if not root.GetChild("rigid_bodies") or not root.GetChild("joints"):
        raise RuntimeError("EmbodiChain USDC assembly has no rigid_bodies or joints scope")
