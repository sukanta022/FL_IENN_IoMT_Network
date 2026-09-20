"""fl_ienn.session — Steps 1-3 (device simulation, auth, EKA).

Step 1 only ships Device + DeviceRegistry + StreamEnvelope. Step 2/3 will
extend this package without modifying Step 1.
"""
from .device_sim import (
    Device,
    DeviceRegistry,
    StreamEnvelope,
    assign_device,
    iter_stream,
    iter_all_datasets,
    build_registry,
    REGISTRY_PATH,
)

__all__ = [
    "Device",
    "DeviceRegistry",
    "StreamEnvelope",
    "assign_device",
    "iter_stream",
    "iter_all_datasets",
    "build_registry",
    "REGISTRY_PATH",
]
