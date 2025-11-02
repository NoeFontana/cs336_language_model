import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cs336.merge.merge_py import merge
else:
    use_native_str = os.environ.get("USE_NATIVE_MERGE", "0").lower()

    if use_native_str not in ("1", "true", "yes", "on"):
        from cs336.merge.merge_py import merge

        log_message = "Using Python implementation of 'merge'"
    else:
        from cs336_native import merge

        log_message = "Using Rust implementation of 'merge'"

    logging.getLogger(__name__).info(log_message)

__all__ = ["merge"]
