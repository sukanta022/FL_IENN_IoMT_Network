"""Allow `python -m fl_ienn.common` to invoke prepare_data.main()."""
from .prepare_data import main

if __name__ == "__main__":
    raise SystemExit(main())
