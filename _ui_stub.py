#!/usr/bin/env python
"""Lanzador local de la UI de Gradio para exploración/UX sin GPU ni modelos.

Uso rápido (desde la raíz del repo):
    pip install gradio torch pillow huggingface_hub requests
    python _ui_stub.py
    # → abre http://localhost:7860

Recarga automática al guardar app_space.py:
    pip install watchdog
    gradio _ui_stub.py      # gradio CLI detecta cambios con --reload

Qué funciona:
  - Toda la UI de Gradio (layout, sliders, dropdowns, presets, pestañas)
  - Dropdowns de slot con LoRAs custom si existen en ./models/loras/ltx23/custom/
  - El botón de generar devuelve un video negro de 1 s (stub)
  - El enhancer devuelve el prompt tal cual (stub)

Qué NO funciona (es el objetivo):
  - La generación real (sin GPU ni modelos)
  - Descargas de modelos
"""
from __future__ import annotations
import os, pathlib, sys, tempfile, time, types

# ── 1. Rutas ──────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent
APP = ROOT / "app_space.py"
assert APP.exists(), f"No se encontró {APP}"

# El stub crea directorios mínimos para que el código no truene al construir
# las rutas (MODELS / "loras" / "ltx23" / "custom" etc.).
_STUB_ROOT = pathlib.Path(tempfile.mkdtemp(prefix="ltx_ui_stub_"))
for sub in ("checkpoints", "diffusion_models", "vae", "text_encoders",
            "loras/ltx23/custom", "clip", "upscale_models", "suno"):
    (_STUB_ROOT / "ComfyUI" / "models" / sub).mkdir(parents=True, exist_ok=True)
for sub in ("input", "output"):
    (_STUB_ROOT / "ComfyUI" / sub).mkdir(parents=True, exist_ok=True)

# Si hay LoRAs custom reales en el repo, hacemos un enlace simbólico / copia
# para que los dropdowns de la UI los vean.
_REAL_CUSTOM = ROOT / "models" / "loras" / "ltx23" / "custom"
_STUB_CUSTOM = _STUB_ROOT / "ComfyUI" / "models" / "loras" / "ltx23" / "custom"
if _REAL_CUSTOM.exists():
    for f in _REAL_CUSTOM.glob("*.safetensors"):
        dst = _STUB_CUSTOM / f.name
        if not dst.exists():
            try:
                dst.symlink_to(f)
            except OSError:
                import shutil; shutil.copy2(f, dst)
    man = _REAL_CUSTOM / "manifest.json"
    if man.exists():
        import shutil; shutil.copy2(man, _STUB_CUSTOM / "manifest.json")

# ── 2. Variables de entorno que controlan el arranque ─────────────────────────
os.environ["SKIP_STARTUP_SETUP"] = "1"   # evita _ensure_comfy / _ensure_models
os.environ["DISABLE_ENHANCER"] = "1"     # no lanza llama-server
os.environ.setdefault("SKIP_SLOT_LORAS", "")  # no omite slots

# ── 3. Mock de 'spaces' (ZeroGPU → no-op) ────────────────────────────────────
_mock = types.ModuleType("spaces")
def _gpu_deco(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return lambda fn: fn
_mock.GPU = _gpu_deco
_mock.ZeroGPU = _gpu_deco
sys.modules["spaces"] = _mock

# ── 4. Mock de torch mínimo (para que el import top-level no truene en CPU) ──
# Solo necesitamos que torch.cuda.* exista; la generación no lo llama.
try:
    import torch as _torch  # noqa (si está instalado, perfecto)
except ImportError:
    _torch_mock = types.ModuleType("torch")
    class _CudaMod:
        def is_available(self): return False
        def empty_cache(self): pass
        def get_device_properties(self, *a): return type("P", (), {"total_memory": 0})()
    _torch_mock.cuda = _CudaMod()
    _torch_mock.float16 = None
    _torch_mock.bfloat16 = None
    _torch_mock.float32 = None
    sys.modules["torch"] = _torch_mock

# ── 5. Leer y parchear app_space.py en memoria ───────────────────────────────
source = APP.read_text(encoding="utf-8")

# Redirigir ROOT al directorio temporal para que MODELS / INPUT / OUTPUT
# apunten a paths que existen (necesario al construir _slot_lora_choices).
source = source.replace(
    "ROOT = pathlib.Path(__file__).resolve().parent",
    f"ROOT = pathlib.Path({str(_STUB_ROOT)!r})\n"
    f"_REAL_APP_DIR = pathlib.Path({str(ROOT)!r})",
    1,
)

# El bloque de startup usa SKIP_STARTUP_SETUP, pero _init_comfy_nodes todavía
# se invocaría en el bloque try dentro del startup. Añadir un guard corto:
source = source.replace(
    "    try:\n        _init_comfy_nodes()\n        _workflow_template()\n        _runexx_workflow_template()",
    "    try:\n        pass  # stub: skip comfy init",
    1,
)

# _prepare_runtime → no-op (se llama dentro de generate al pulsar el botón)
source = source.replace(
    "def _prepare_runtime(progress: gr.Progress | None = None) -> None:\n"
    "    _ensure_comfy()\n"
    "    _ensure_models(progress)\n"
    "    _init_comfy_nodes()",
    "def _prepare_runtime(progress: gr.Progress | None = None) -> None:\n"
    "    pass  # stub",
    1,
)

# generate → devuelve un video negro de 1 s + seed
_GENERATE_STUB = '''
def _stub_black_video(seed: int) -> str:
    """Crea un MP4 de 1 s en negro para que el componente Video tenga algo."""
    import tempfile, subprocess, shutil
    out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    if shutil.which("ffmpeg"):
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=black:size=512x512:rate=8",
             "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p", out],
            capture_output=True,
        )
    else:
        # Sin ffmpeg: un MP4 mínimo de 4 bytes (no reproducible pero no truena)
        open(out, "wb").write(b"\\x00\\x00\\x00\\x20ftypisom")
    return out

'''

source = source.replace(
    "@spaces.GPU(duration=get_gpu_duration)\ndef generate(",
    _GENERATE_STUB + "@spaces.GPU(duration=get_gpu_duration)\ndef generate(",
    1,
)

# Reemplazar el cuerpo de generate: del primer `seed_value =` hasta el último
# `return` + except block (mantenemos la firma entera, cambiamos el cuerpo).
# Estrategia: remplazamos la primera línea del cuerpo hasta el raise final.
# Reemplazamos el cuerpo completo de generate (try/except inclusive) con un stub
# que devuelve un video negro de 1 s. El anchor exacto: primera línea del cuerpo
# hasta el cierre del except (incluyendo la línea del return del except).
_GEN_BODY_OLD = (
    "    seed_value = random.randint(0, 2**32 - 1) if randomize_seed or seed < 0 else int(seed)\n"
    "    msr_enabled = input_mode == \"multi-reference (MSR)\"\n"
    "    msr_original = input_mode == \"multi-reference (original)\"\n"
    "    any_msr = msr_enabled or msr_original\n"
    "    try:"
)
_GEN_BODY_OLD_END = (
    "        return None, tb[-6000:], seed_value"
)
_GEN_BODY_NEW = (
    "    seed_value = random.randint(0, 2**32 - 1) if randomize_seed or seed < 0 else int(seed)\n"
    "    if not image_path:\n"
    "        raise gr.Error(\"[stub] sube una imagen primero\")\n"
    "    if not prompt.strip():\n"
    "        raise gr.Error(\"[stub] el prompt está vacío\")\n"
    "    time.sleep(1.5)  # simular tiempo de generación\n"
    "    _vid = _stub_black_video(seed_value)\n"
    "    return _vid, f\"[stub] generación simulada — seed {seed_value}\", seed_value"
)

_start = source.find(_GEN_BODY_OLD)
_end   = source.find(_GEN_BODY_OLD_END)
if _start != -1 and _end != -1:
    _end += len(_GEN_BODY_OLD_END)
    source = source[:_start] + _GEN_BODY_NEW + source[_end:]
else:
    print("[stub] ⚠️  No se pudo parchear el cuerpo de generate — pulsar Generar colgará")

# enhance_prompt → devuelve el prompt tal cual
source = source.replace(
    "@spaces.GPU(duration=get_enhance_duration)\ndef enhance_prompt(",
    "@spaces.GPU(duration=get_enhance_duration)\ndef _enhance_prompt_real(",
    1,
)
# Insertar la versión stub justo antes de la real (que ahora se llama _real)
source = source.replace(
    "@spaces.GPU(duration=get_enhance_duration)\ndef _enhance_prompt_real(",
    (
        "def enhance_prompt(image_path, prompt, *args, **kwargs):\n"
        "    time.sleep(0.3)\n"
        "    return prompt + \"  [stub: enhancer desactivado]\"\n"
        "\n"
        "@spaces.GPU(duration=get_enhance_duration)\ndef _enhance_prompt_real("
    ),
    1,
)

# launch: no share, puerto fijo, abre navegador
source = source.replace(
    "    demo.launch()",
    "    demo.launch(server_name='0.0.0.0', server_port=7860, share=False, inbrowser=True)",
    1,
)

# ── 6. Ejecutar ───────────────────────────────────────────────────────────────
_app_path = str(APP)
print("=" * 60)
print("  10Eros LTX — UI stub (sin GPU, sin modelos)")
print("  → http://localhost:7860")
print("  Recarga: `gradio _ui_stub.py`  (requiere pip install watchdog)")
print("=" * 60)
exec(compile(source, _app_path, "exec"), {"__file__": _app_path, "__name__": "__main__"})
