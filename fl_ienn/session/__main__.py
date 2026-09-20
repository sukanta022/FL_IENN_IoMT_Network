"""Allow `python -m fl_ienn.session` to invoke device_sim.main()."""
from .device_sim import main

if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
