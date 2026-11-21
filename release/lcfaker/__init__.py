from lcfaker.generator import Task, generate, sample_task

try:   # the main interface (bpl-v17) needs torch
    from lcfaker.torch_prior import BplTasks
except ImportError:   # pragma: no cover
    BplTasks = None

__all__ = ["BplTasks", "Task", "generate", "sample_task"]
__version__ = "0.3.1"
