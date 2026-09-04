"""Optional vLLM platform plugin for hosts where NVML probing is broken.

vLLM can run CUDA through its NonNvmlCudaPlatform, but the built-in platform
resolver may fail before selecting CUDA when pynvml cannot initialize. This
plugin is intentionally opt-in through VLLM_PLUGINS.
"""


def vllm_force_cuda_platform() -> str:
    return "vllm.platforms.cuda.CudaPlatform"
