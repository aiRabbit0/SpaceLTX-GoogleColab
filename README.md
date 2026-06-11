# LTX 2.3 10Eros — Space de HuggingFace + Google Colab

Copias de trabajo locales del Space de Hugging Face [`signsur4739379373/LTX-2.3-10Eros`](https://huggingface.co/spaces/signsur4739379373/LTX-2.3-10Eros): una app Gradio de **image-to-video con audio nativo** sobre el checkpoint LTX 2.3 "10Eros", con backend ComfyUI ejecutado en proceso, más un port a Google Colab.

## Contenido

| Archivo | Descripción |
|---|---|
| `app_space.py` | El `app.py` del Space (renombrado aquí). App Gradio autocontenida: al arrancar clona ComfyUI + 16 repos de custom nodes, descarga los modelos (~30 GB) desde HF Hub y ejecuta los workflows directamente con `PromptExecutor` (sin servidor HTTP de ComfyUI). |
| `LTX_10Eros_Colab.ipynb` | Notebook de Colab que clona el Space y lo ejecuta en GPU gratuita o de pago: mockea el módulo `spaces` (ZeroGPU), valida el entorno para desbloquear las variantes ejecutables (GGUF Q3–Q8 o fp8 original), parchea `app.py` en memoria y permite reemplazar 6 slots de LoRAs por LoRAs propias de CivitAI con sus metadatos. |
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

#### Ciclo de las LoRAs custom (celdas 5 → 8)

Hay 6 slots intercambiables con nombres genéricos (`slot1`…`slot6`; sus LoRAs originales son Cinematic Hardcut, Synth, Plora Sulfer, OmniNFT RL bf16, Better Motion y Physics V2 respectivamente). El ciclo completo:

1. **Celda 5**: pegas la URL de CivitAI (o el ID de versión) en el slot. La celda consulta la API de CivitAI y muestra los **metadatos de la LoRA** (nombre real, versión, base model, trigger words, tamaño), avisa si el base model no parece LTX, descarga el archivo a `/content/custom_loras/` y escribe un manifest (`mapping.json`) con archivo + metadatos por slot.
2. **Celda 8** (lanzamiento): lee el manifest y por cada slot reemplazado parchea `app.py` en memoria para que (a) la constante `*_LORA_FILENAME` apunte a tu archivo, (b) **no** se descargue la LoRA original de ese slot, y (c) los sliders del slot (vídeo y audio) se **renombren** a `custom: <nombre de tu LoRA>` — así la UI ya no muestra el nombre genérico del slot. Las trigger words se imprimen en consola como recordatorio.
3. **En la UI**: subes el slider del slot (la mayoría arranca en 0) para activar tu LoRA.

Slot vacío (`""`) = se descarga y usa la LoRA original; no se pierde nada.

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