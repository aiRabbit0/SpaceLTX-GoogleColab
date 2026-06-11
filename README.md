# LTX 2.3 10Eros — Image-to-Video con audio nativo (Space de HF + Google Colab)

App Gradio de **image-to-video con audio nativo** sobre el checkpoint LTX 2.3 "10Eros", con backend ComfyUI ejecutado en proceso. Es un fork del Space de Hugging Face [`signsur4739379373/LTX-2.3-10Eros`](https://huggingface.co/spaces/signsur4739379373/LTX-2.3-10Eros) con un port a Google Colab que funciona en la **GPU T4 gratuita**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aiRabbit0/SpaceLTX-GoogleColab/blob/main/LTX_10Eros_Colab.ipynb)

## Inicio rápido (Colab)

1. Pulsa el botón **Open in Colab** de arriba (o abre `LTX_10Eros_Colab.ipynb` en Colab).
2. Activa la GPU: `Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU (T4)`.
3. Ejecuta las celdas **en orden**. La Celda 5 (LoRAs de CivitAI) es opcional.
4. La Celda 7 valida la GPU/disco y permite elegir la variante del modelo (en T4 gratuita: GGUF `Q4_K_S`).
5. La Celda 8 lanza la app y **permanece ejecutando** (spinner activo) mientras Gradio esté vivo — igual que Automatic1111. Cuando aparezca `Running on public URL: https://...gradio.live`, abre ese enlace en una pestaña nueva. Para detener: botón ■ de la celda (o `Ctrl+M I`).

La primera ejecución descarga ~45–85 GB de modelos según la variante (10–25 min). `/content` se borra al reiniciar el runtime, así que las descargas no persisten entre sesiones.

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `app_space.py` | La app (en el Space se llama `app.py`). Gradio autocontenida: al arrancar clona ComfyUI + 16 repos de custom nodes, descarga los modelos desde HF Hub y ejecuta los workflows directamente con `PromptExecutor` (sin servidor HTTP de ComfyUI). |
| `LTX_10Eros_Colab.ipynb` | Notebook de Colab. Descarga el código de **este repositorio** (`app_space.py` → `app.py`, `workflow_runexx.json` → `runexx_msr_workflow.json`), mockea el módulo `spaces` (ZeroGPU), valida el entorno, parchea `app.py` en memoria (checkpoint GGUF, enhancer opcional...) y lanza la interfaz con enlace público. |
| `workflow_default.json` | Copia de referencia del workflow visual principal de I2V (alojado en `TenStrip/LTX2.3-10Eros_Workflows`, fijado por revisión en `app_space.py`). |
| `workflow_runexx.json` | Copia del workflow visual multi-referencia (MSR), que el notebook instala como `runexx_msr_workflow.json`. |

## Selector de modelo según el entorno (Celda 7)

La celda detecta la GPU y el disco libre y desbloquea solo las variantes del checkpoint que caben en la VRAM (el disco solo avisa con ⚠️). La selección se guarda en `LTX_MODEL_CHOICE` y la Celda 8 la usa automáticamente; si se omite la celda, vale su parámetro `GGUF_QUANT`.

| Entorno | Variantes desbloqueadas (aprox.) |
|---|---|
| Colab gratuito (T4, ~15 GB VRAM) | GGUF hasta `Q4_K_S` (el por defecto) |
| Colab Pro (L4, ~22.5 GB VRAM) | GGUF hasta `Q6_K` |
| A100 (40 GB VRAM) | Todo, incluido `FP8_ORIGINAL` (checkpoint sin cuantizar, 29.2 GB — se omiten los parches GGUF) |

## LoRAs: originales y personalizadas

La app trae ~14 LoRAs opcionales con sliders de fuerza (video y audio por separado). Siete de ellas son **slots intercambiables**:

- **Slots 1-6** tienen una LoRA original (Cinematic Hardcut, Synth, Plora Sulfer, OmniNFT RL bf16, Better Motion, Physics V2). Se descargan según los **checkboxes** de la Celda 8 (por defecto, todas; desmarcar una ahorra su tamaño en disco).
- **Slot 7** es extra, sin LoRA original.

Para usar LoRAs propias de CivitAI:

1. **Celda 5**: pega las URLs (o IDs de versión) en el campo `CUSTOM_LORAS_LINKS` (varias separadas por comas) o en la lista `CUSTOM_LORAS`. La celda muestra los metadatos (nombre, base model, trigger words, tamaño), avisa si el base model no parece LTX 2.3, y descarga con verificación (descarga atómica + detección de modelos *gated*). Si da 401, configura `CIVITAI_TOKEN` en la Celda 2.
2. **En la UI**: cada slot tiene un **dropdown** con su LoRA original + todas las custom instaladas + `(none)`. Elige el archivo, sube el slider del slot y añade las trigger words al prompt. Los presets ajustan fuerzas pero nunca cambian el archivo elegido.

## Características de la app

- **I2V con audio nativo**: vídeo y audio se generan conjuntamente (LTX 2.3 AV).
- **Enhance prompt**: un servidor llama.cpp con un modelo de visión expande un concepto corto en un prompt detallado afinado para LTX. En Colab viene **desactivado por defecto** (`DISABLE_ENHANCER`): su modelo (~13 GB) no cabe en el disco del Colab gratuito.
- **Presets** que ajustan modo, sigmas y las intensidades de las LoRAs opcionales.
- **Modos multi-referencia (MSR)** cableados de extremo a extremo pero ocultos en la UI (WIP).
- Funciones extra inyectables en el workflow: prompt relay por segmentos, scene chain, condicionamiento K/V, referencia de audio con separación de stems.

## Previsualización local de la UI (sin GPU ni modelos)

`_ui_stub.py` parchea `app_space.py` en memoria y lanza la UI de Gradio en `localhost:7860` sin descargar ningún modelo — útil para explorar el layout, probar cambios de UX y ver cómo quedan los dropdowns de LoRA.

**Instalación (una sola vez):**

```bash
pip install -r requirements-dev.txt
# torch CPU (~200 MB) en vez del completo con CUDA:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Lanzar la UI:**

```bash
python _ui_stub.py
# → http://localhost:7860
```

**Con recarga automática** al guardar `app_space.py` (requiere `watchdog`):

```bash
gradio _ui_stub.py
```

Qué funciona: toda la UI (sliders, dropdowns, presets, pestañas). Si tienes LoRAs custom en `models/loras/ltx23/custom/`, los dropdowns las muestran. El botón Generar espera ~1.5 s y devuelve un video negro (usa `ffmpeg` si está disponible). El enhancer devuelve el prompt tal cual.

Qué **no** funciona: generación real (se necesita GPU + modelos).

## Desarrollo

El notebook ejecuta el código de este repositorio (rama `main`). Para modificar la app: haz un **fork**, edita `app_space.py`, haz push, y en el notebook apunta la variable `CODE_RAW` (Celda 3) a tu fork antes de re-ejecutar las Celdas 3 y 8.

No hay tests ni build: el código solo corre realmente en el Space (ZeroGPU) o en Colab. Verificación rápida:

```bash
python -m py_compile app_space.py
python -X utf8 -m json.tool workflow_runexx.json > /dev/null
```

⚠️ **Invariante principal**: el notebook parchea `app.py` por **reemplazo exacto de strings**. Cualquier cambio en `app_space.py` que toque un string objetivo de un `patch(...)` de la Celda 8 debe actualizarse también en el notebook, o el parche fallará en silencio (avisa con `⚠️ parche ... NO aplicado`). Más invariantes y arquitectura en [`CLAUDE.md`](CLAUDE.md).

---
*Cada usuario es el único responsable de todo el contenido que genere.*
