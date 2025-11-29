import importlib.util
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cs336_basics.merge.merge_py import merge
else:
    use_native = os.environ.get("USE_NATIVE_MERGE", "0").lower() in ("1", "true", "yes", "on")
    native_available = importlib.util.find_spec("cs336_native") is not None

    if use_native and native_available:
        from cs336_native import merge

        log_message = "Using Rust implementation of 'merge'"
    else:
        from cs336.merge.merge_py import merge

        log_message = "Using Python implementation of 'merge'"
        if use_native and not native_available:
            logging.getLogger(__name__).warning(
                "USE_NATIVE_MERGE is set, but cs336_native extension is not available. "
                "Falling back to Python implementation."
            )

    logging.getLogger(__name__).info(log_message)


__all__ = ["merge"]
