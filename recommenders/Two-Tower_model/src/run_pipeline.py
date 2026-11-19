"""Run offline stages sequentially: GPU check -> data -> train/export -> evaluate/charts -> ANN index."""
from __future__ import annotations

import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def main() -> None:
    t0 = time.perf_counter()
    from . import ann_index, check_gpu, evaluate, train
    check_gpu.main()
    prep, model, _, _ = train.main()          # runs dataset.main() internally
    evaluate.main(prep, model)
    ann_index.main(prep, model)
    print(f"\nOffline pipeline finished in {time.perf_counter() - t0:.1f}s -> outputs/")


if __name__ == "__main__":
    main()
