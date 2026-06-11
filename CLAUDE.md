# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Local working copies for the Hugging Face Space `signsur4739379373/LTX-2.3-10Eros` (a Gradio app that runs LTX 2.3 image-to-video with native audio through an in-process ComfyUI backend) plus a Google Colab port:

- `app_space.py` — the Space's `app.py`. Renamed here; in the Space repo it is `app.py`.
- `LTX_10Eros_Colab.ipynb` — Colab notebook that clones the Space and runs it on a free T4 by patching `app.py` in memory (Spanish-language; keep user-facing notebook text in Spanish).
- `workflow_default.json` — local copy of the main I2V visual workflow (hosted on HF as `TenStrip/LTX2.3-10Eros_Workflows/10Eros_10SNodes_LikenessGuideHelper_I2V_v3.2.json`, pinned by revision in `app_space.py`).
- `workflow_runexx.json` — local copy of the multi-reference (MSR) visual workflow, bundled with the Space as `runexx_msr_workflow.json`.

There is no build system, test suite, linter config, or requirements.txt here. The code only actually runs on the Space (ZeroGPU) or in Colab — it bootstraps ComfyUI, clones 16 custom-node repos, and downloads ~30+ GB of models at startup, so don't try to run `app_space.py` locally to verify changes.

## Useful commands

- Syntax-check the app: `python -m py_compile app_space.py`
- Validate JSON/notebook: `python -X utf8 -m json.tool workflow_runexx.json` (use `-X utf8` on this Windows machine — the notebook contains emoji that crash cp1252 output)
- Inspect notebook cells: load `LTX_10Eros_Colab.ipynb` with `json.load` and join each cell's `source`

## Critical cross-file invariants

1. **The Colab notebook patches `app.py` by exact string replacement.** The launch cell ("Celda 8") reads the cloned Space's `app.py` and calls `patch(old, new, label)` / `gpatch(...)` (GGUF-only variant, skipped when `FP8_ORIGINAL` is selected) with verbatim source snippets: the fp8-checkpoint entry in `DOWNLOADS`, loader node handling, the `*_LORA_FILENAME` constants and per-slot `DOWNLOADS` blocks, **and the Gradio slider `label="..."` lines for the 6 swappable LoRA slots** (video + audio, renamed to `custom: <name>` from CivitAI metadata). Any edit to `app_space.py` that touches a patch-target string must update the corresponding `old` string in the notebook, or the patch silently degrades (prints `⚠️ parche ... NO aplicado`).
2. **`get_gpu_duration` and `generate` must keep identical signatures.** `@spaces.GPU(duration=get_gpu_duration)` passes the same positional payload to both (~70 params). Hidden/disabled UI components (the MSR inputs, `input_mode`) are kept in the Gradio layout solely so these positions stay stable — do not remove them.
3. **`RUNEXX_NODE_*` constants are node ids inside `workflow_runexx.json`.** The runexx converter patches/skips/rewires nodes by these ids; changing the workflow JSON requires re-verifying every id at the top of `app_space.py`.
4. **The main workflow is pinned** by `WORKFLOW_REPO`/`WORKFLOW_REVISION`/`WORKFLOW_FILENAME`; `workflow_default.json` is the reference copy of that file. Publishing a changed workflow means uploading to the HF repo and bumping the revision hash.

## Architecture of app_space.py (~3,800 lines, single file)

**Bootstrap layer** (`_ensure_comfy`, `_ensure_models`, `_init_comfy_nodes`, `_prepare_runtime`): clones ComfyUI at a pinned commit plus the `CUSTOM_NODES` list, installs filtered requirements, downloads every entry in `DOWNLOADS` (checkpoint, gemma text encoder, ~15 LoRAs, upscaler, stem-separation model), and initializes ComfyUI's node registry in-process. ComfyUI is never run as an HTTP server — workflows execute via `execution.PromptExecutor` directly (`_execute_workflow`).

**Embedded custom nodes**: `_KV_WRAPPER_CODE` is a Python module stored as a string and written into `ComfyUI/custom_nodes/` at startup (`_install_kv_wrapper`). It defines `FunPackKVApply` (K/V reference conditioning with LTX-AV compatibility monkey-patches) and `AudioRefPrep`. Edit it as code-in-a-string; it is exercised only inside ComfyUI.

**Workflow pipeline** (the core data flow):
visual workflow JSON → `_convert_workflow` / `_convert_runexx_workflow` (visual→API format: resolves SetNode/GetNode indirection, inlines primitives, patches loader nodes to the 10Eros checkpoint) → cached template (`_workflow_template`, pre-populated at startup) → per-generation deep copy → injectors mutate the dict → `_execute_workflow`.

Injectors, each toggled by UI inputs: `_inject_params` (prompt, image, seed, dimensions, mode, likeness/anchor strengths), `_inject_optional_loras` (12 video + 12 audio strength sliders driven by `PRESET_VALUES` presets), `_inject_refine_sigmas`, `_inject_msr` (multi-subject reference), `_inject_prompt_relay`, `_inject_scene_chain`, `_inject_kv_conditioning`, `_inject_audio_reference`.

**Input modes**: `single image (i2v)` is the live path using the default workflow. The two multi-reference modes (`MSR`, `multi-reference (original)`/runexx) are wired end-to-end but hidden in the UI as WIP.

**Prompt enhancer** (`_ensure_enhancer`, `enhance_prompt`): a separate llama.cpp `llama-server` subprocess on port 18642 running a vision LLM (sulphur GGUF) that expands short concepts into LTX-tuned prompts. The binary is built from source on first run and cached to an HF dataset repo (`_pull_cached_binary`/`_push_cached_binary`).

**ZeroGPU budgeting**: `get_gpu_duration` estimates GPU seconds from frames × pixels with a "tight" formula (fits the 120 s/day free allowance for default 4 s gens) falling back to a wider one; `gen_budget > 0` is a manual override, clamped to `MIN_GPU_SECONDS`/`MAX_GPU_SECONDS` env vars.

## Colab notebook structure

Cells in order: GPU/CUDA check → tokens (HF + CivitAI, enables `hf_transfer`) → clone Space + install deps (torch 2.8.0 with cu-tag fallback) → mock the `spaces` module (no-op `@spaces.GPU`) → optional CivitAI LoRA replacement of 6 slots → VRAM/disk monitor → **environment-validated model selector** → patches + launch → disk cleanup → GPU release.

- **LoRA cell (Celda 5)**: resolves the CivitAI version, fetches metadata via `/api/v1/model-versions/{id}` (name, base model, trained words, size; warns if base model isn't LTX), downloads to `/content/custom_loras/`, and writes `mapping.json` as a manifest `slot → {file, name, version, base_model, trained_words, ...}`. The launch cell also accepts the legacy `slot → filename` string format.
- **Model selector (Celda 7)**: detects VRAM (nvidia-smi) and free disk, prints the `MODEL_CATALOG` with ✅/🔒 per variant (GGUF Q3–Q8 + `FP8_ORIGINAL`), and stores the choice in the `LTX_MODEL_CHOICE` env var via an ipywidgets dropdown limited to unlocked options (free T4 unlocks up to `Q4_K_S`). VRAM/disk thresholds in the catalog are estimates — keep `Q4_K_S` runnable at 15.0 GB (what nvidia-smi reports for a T4).
- **Launch cell (Celda 8)**: reads `LTX_MODEL_CHOICE` (falls back to its `GGUF_QUANT` param). GGUF-specific patches go through `gpatch()` and are skipped entirely for `FP8_ORIGINAL` (original checkpoint, no Kijai VAEs); custom-LoRA swap, slider renaming, and the Gradio `share=True` hook always apply.