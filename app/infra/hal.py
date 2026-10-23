"""Hardware Abstraction Layer (HAL) - Detects runtime environment"""

from enum import Enum
from typing import Dict, Any
import os
import platform
from app.core.subprocess_env import scrubbed_env
import logging

logger = logging.getLogger(__name__)


class HardwareProfile(Enum):
    """Hardware capability profiles"""
    CLOUD_HIGH = "cloud_high"      # High-end cloud (GPU, 32GB+ RAM)
    CLOUD_STANDARD = "cloud_std"   # Standard cloud (CPU, 8GB RAM)
    EDGE_GPU = "edge_gpu"          # Edge with GPU (Jetson, etc)
    EDGE_CPU = "edge_cpu"          # Edge CPU only
    LOCAL_HIGH = "local_high"      # Local workstation (GPU)
    LOCAL_STANDARD = "local_std"   # Local laptop
    EMBEDDED = "embedded"          # Microcontroller/Raspberry Pi


class HALBlock:
    """
    Hardware Abstraction Layer Block
    Detects hardware capabilities and runtime environment
    """
    
    name = "hal"
    version = "1.0.0"
    
    def __init__(self):
        self.profile = None
        self.capabilities = {}
        self._detect()
    
    def _detect(self):
        """Detect hardware capabilities"""
        # Check for GPU
        has_gpu = self._check_gpu()
        
        # Check memory
        memory_gb = self._get_memory()
        
        # Check environment
        env = self._detect_environment()
        
        # Determine profile
        if env == "cloud":
            if has_gpu and memory_gb >= 32:
                self.profile = HardwareProfile.CLOUD_HIGH
            else:
                self.profile = HardwareProfile.CLOUD_STANDARD
        elif env == "edge":
            if has_gpu:
                self.profile = HardwareProfile.EDGE_GPU
            else:
                self.profile = HardwareProfile.EDGE_CPU
        else:  # local
            if has_gpu and memory_gb >= 16:
                self.profile = HardwareProfile.LOCAL_HIGH
            elif memory_gb >= 4:
                self.profile = HardwareProfile.LOCAL_STANDARD
            else:
                self.profile = HardwareProfile.EMBEDDED
        
        self.capabilities = {
            "has_gpu": has_gpu,
            "memory_gb": memory_gb,
            "environment": env,
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "supports_local_llm": has_gpu or memory_gb >= 16,
            "supports_vector_db": memory_gb >= 4,
            "supports_ocr": True,  # Most systems
        }
    
    def _check_gpu(self) -> bool:
        """Check if GPU is available"""
        # Check for CUDA
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, env=scrubbed_env())
            if result.returncode == 0:
                return True
        except Exception:
            # `nvidia-smi` missing is the NORMAL case on any host without an
            # NVIDIA GPU, which includes most dev machines and the Render box.
            # DEBUG, and no traceback: at WARNING this fires on every single
            # boot with a full stack, which teaches operators that warnings
            # from this app are noise. The consequence is stated instead.
            logger.debug(
                "no nvidia-smi on this host — no CUDA GPU detected, "
                "inference will run on CPU"
            )
        
        # Check for Metal (Mac)
        if platform.system() == "Darwin":
            try:
                import subprocess
                result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], capture_output=True, text=True, env=scrubbed_env())
                if "Metal" in result.stdout:
                    return True
            except Exception:
                # Same reasoning as the CUDA probe above: a failed capability
                # probe is not an error, it is the answer "no".
                logger.debug(
                    "system_profiler probe failed — no Metal GPU detected, "
                    "inference will run on CPU", exc_info=True,
                )
        
        return False
    
    def _get_memory(self) -> int:
        """Get system memory in GB"""
        try:
            import psutil
            return psutil.virtual_memory().total // (1024**3)
        except Exception:
            # Fallback - assume 8GB
            return 8
    
    def _detect_environment(self) -> str:
        """Detect runtime environment"""
        # Check for cloud providers
        if os.getenv("RENDER") or os.getenv("RAILWAY"):
            return "cloud"
        if os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
            return "cloud"
        if os.getenv("GCP_PROJECT"):
            return "cloud"
        
        # Check for edge devices
        if os.path.exists("/etc/nv_tegra_release"):  # Jetson
            return "edge"
        if os.getenv("JETSON"):
            return "edge"
        
        return "local"
    
    def detect(self) -> HardwareProfile:
        """Return detected hardware profile"""
        return self.profile
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get hardware capabilities"""
        return self.capabilities
    
    def get_recommendations(self) -> Dict[str, Any]:
        """Get block recommendations based on hardware"""
        recs = {
            "chat_provider": "deepseek",
            "vector_backend": "pgvector",  # The Fork uses Postgres pgvector
            "use_local_embeddings": False,
            "use_local_ocr": False,
            "max_concurrent_requests": 10,
        }

        if self.profile == HardwareProfile.CLOUD_HIGH:
            recs.update({
                "chat_provider": "deepseek",
                "use_local_embeddings": True,
                "max_concurrent_requests": 100,
            })
        elif self.profile == HardwareProfile.EDGE_GPU:
            recs.update({
                "chat_provider": "openrouter",
                "use_local_embeddings": True,
                "use_local_ocr": True,
            })
        elif self.profile == HardwareProfile.EMBEDDED:
            recs.update({
                "vector_backend": "memory",  # Use memory block
                "max_concurrent_requests": 1,
            })
        
        return recs
    
    async def health(self) -> Dict[str, Any]:
        """Health check"""
        return {
            "name": self.name,
            "version": self.version,
            "profile": self.profile.value,
            "capabilities": self.capabilities,
            "recommendations": self.get_recommendations(),
            "healthy": True
        }
