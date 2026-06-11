# LTX 2.3 10Eros — Space de HuggingFace + Google Colab

Copias de trabajo locales del Space de Hugging Face [`signsur4739379373/LTX-2.3-10Eros`](https://huggingface.co/spaces/signsur4739379373/LTX-2.3-10Eros): una app Gradio de **image-to-video con audio nativo** sobre el checkpoint LTX 2.3 "10Eros", con backend ComfyUI ejecutado en proceso, más un port a Google Colab.

## Contenido

| Archivo | Descripción |
|---|---|
| `app_space.py` | El `app.py` del Space (renombrado aquí). App Gradio autocontenida: al arrancar clona ComfyUI + 16 repos de custom nodes, descarga los modelos (~30 GB) desde HF Hub y ejecuta los workflows directamente con `PromptExecutor` (sin servidor HTTP de ComfyUI). |
| `LTX_10Eros_Colab.ipynb` | Notebook de Colab que descarga el código de **este repo** (`app_space.py` → `app.py`, `workflow_runexx.json` → `runexx_msr_workflow.json`) y lo ejecuta en GPU gratuita o de pago: mockea el módulo `spaces` (ZeroGPU), valida el entorno para desbloquear las variantes ejecutables (GGUF Q3–Q8 o fp8 original), parchea `app.py` en memoria y permite **añadir LoRAs de CivitAI** que luego se eligen por slot en la UI mediante dropdowns. |
| `workflow_default.json` | Copia local del workflow visual principal de I2V (`10Eros_10SNodes_LikenessGuideHelper_I2V_v3.2.json`, alojado en `TenStrip/LTX2.3-10Eros_Workflows` y fijado por revisión en `app_space.py`). |
| `workflow_runexx.json` | Copia local del workflow visual multi-referencia (MSR), incluido en el Space como `runexx_msr_workflow.json`. |

## Uso

### En el Space (ZeroGPU)

`app_space.py` se sube al Space como `app.py`. La duración de GPU se estima en `get_gpu_duration` (fórmula ajustada para que una generación de 4 s quepa en los 120 s/día gratuitos de ZeroGPU); se puede forzar un presupuesto manual desde la UI.

### En Colab

1. Abre `LTX_10Eros_Colab.ipynb` en Colab con runtime **GPU (T4)**.
2. Ejecuta las celdas en orden (la de LoRAs de CivitAI es opcional).
3. La celda 7 **valida el entorno** (VRAM/disco) y te deja elegir el modelo entre las variantes desbloqueadas.
4. La celda 8 aplica los parches y abre la interfaz (`gradio.live`).

La primera ejecución descarga ~45–85 GB de modelos según la variante. `/content` se borra al reiniciar el runtime.

#### Selector de modelo según el entorno (celda 7)

La celda detecta la GPU y el disco libre, y desbloquea solo las variantes del checkpoint que caben en ese runtime (las bloqueadas se muestran con 🔒 y el motivo). La selección se guarda en `LTX_MODEL_CHOICE` y la celda de lanzamiento la usa automáticamente; si se omite la celda, vale el parámetro `GGUF_QUANT` de siempre.

| Entorno | Variantes desbloqueadas (aprox.) |
|---|---|
| Colab gratuito (T4, ~15 GB VRAM) | GGUF hasta `Q4_K_S` (el por defecto) |
| Colab Pro (L4, ~22.5 GB VRAM) | GGUF hasta `Q6_K` |
| A100 (40 GB VRAM) | Todo, incluido `FP8_ORIGINAL` (checkpoint sin cuantizar, 29.2 GB — se omiten los parches GGUF) |

#### Ciclo de las LoRAs custom (celdas 5 → 8 → UI)

Las LoRAs originales de la app se descargan **siempre**; las custom solo se **añaden** al catálogo y se eligen en runtime. Hay 7 slots: los slots 1-6 corresponden a una LoRA original (Cinematic Hardcut, Synth, Plora Sulfer, OmniNFT RL bf16, Better Motion y Physics V2) y el slot 7 es extra, sin original (por defecto `(none)`). El ciclo completo:

1. **Celda 5**: pegas una lista de URLs de CivitAI (o IDs de versión). La celda consulta la API de CivitAI, muestra los **metadatos** (nombre real, versión, base model, trigger words, tamaño), avisa si el base model no parece LTX y descarga a `/content/custom_loras/` (descarga atómica vía `.part` + guard contra respuestas HTML de modelos gated). Escribe `mapping.json` keyed por archivo con los metadatos.
2. **Celda 8** (lanzamiento): el hook del instalador enlaza los archivos a `models/loras/ltx23/custom/` y copia el manifest como `manifest.json` — nada más; no se parchea ninguna constante ni label por LoRA.
3. **En la UI**: cada slot tiene un **dropdown** (`_slot_lora_choices` en `app_space.py`) con su LoRA original + todas las custom instaladas (mostradas con su nombre de CivitAI) + `(none)`. El slider del slot controla el archivo elegido; los presets ajustan fuerzas pero no tocan el dropdown.

## Características de la app

- **I2V con audio nativo**: vídeo y audio se generan conjuntamente (LTX 2.3 AV).
- **Enhance prompt**: un servidor llama.cpp con un modelo de visión expande un concepto corto en un prompt detallado afinado para LTX, a partir de la imagen de referencia.
- **Presets** que ajustan modo, sigmas y las intensidades de ~12 LoRAs opcionales (vídeo + audio por separado).
- **Modos multi-referencia (MSR)** cableados de extremo a extremo pero ocultos en la UI (WIP).
- Funciones extra inyectables en el workflow: prompt relay por segmentos, scene chain, condicionamiento K/V, referencia de audio con separación de stems.

## Desarrollo

No hay tests ni build: el código solo corre realmente en el Space o en Colab. Verificación rápida:

```bash
python -m py_compile app_space.py
python -X utf8 -m json.tool workflow_runexx.json > /dev/null
```

⚠️ **Invariante principal**: el notebook parchea `app.py` por **reemplazo exacto de strings**. Cualquier cambio en `app_space.py` que toque un string objetivo de un `patch(...)` del notebook debe actualizarse también en el notebook, o el parche fallará en silencio (avisa con `⚠️ parche ... NO aplicado`). Más invariantes y arquitectura en [`CLAUDE.md`](CLAUDE.md).

---
*Eres el único responsable de todo el contenido que generes.*