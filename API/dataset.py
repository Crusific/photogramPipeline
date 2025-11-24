from dataclasses import dataclass, asdict
from typing import List, Optional, Iterator, Any
import json
import logging
import tempfile
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DatasetError(Exception):
    """Custom exception for dataset-related errors."""

@dataclass
class GeoLocation:
    latitude: float
    longitude: float

@dataclass
class DroneLocation:
    geoLocation: GeoLocation
    altitude: float  # in meters

@dataclass
class TargetLocation:
    xCoord: float  # in pixels from center
    yCoord: float  # in pixels from center

class DataPoint:
    def __init__(self, drone_location: DroneLocation, target_locations: List[TargetLocation]):
        self.drone_location = drone_location
        self.target_locations = target_locations
        
    def to_dict(self) -> dict:
        return {
            "droneLocation": {
                "geoLocation": asdict(self.drone_location.geoLocation),
                "altitude": float(self.drone_location.altitude),
            },
            "targetLocations": [asdict(t) for t in self.target_locations],
        }

    # Pythonic getters
    def get_drone_location(self) -> DroneLocation:
        return self.drone_location

    def get_target_locations(self) -> List[TargetLocation]:
        return self.target_locations

    # camelCase aliases to match provided requirements
    def getDroneLocation(self) -> DroneLocation:
        return self.get_drone_location()

    def getTargetLocations(self) -> List[TargetLocation]:
        return self.get_target_locations()

    def __repr__(self) -> str:
        return f"DataPoint(drone_location={self.drone_location}, target_locations={self.target_locations})"

class DataSet:
    """Loads a JSON dataset and exposes a Pythonic API.

    Features:
    - Accepts JSON `null` entries (mapped to Python `None`).
    - `cyclic` option controls whether `next()` wraps around.
    - `skip_nulls` option controls whether JSON `null` entries are omitted.
    - Iterable support (`__iter__`, `__next__`) that yields one full pass.
    - `peek()` to view current item without advancing.
    - `to_dict()` and `save()` to persist changes atomically.
    - Basic input validation with informative errors.
    """

    def __init__(self, path: str, *, cyclic: bool = True, skip_nulls: bool = False, validate: bool = False):
        self.path = str(path)
        self.cyclic = bool(cyclic)
        self.skip_nulls = bool(skip_nulls)
        self._data: List[Optional[DataPoint]] = []
        self._raw_entries: List[Any] = []
        self._idx: int = 0
        self._iter_pos: Optional[int] = None
        self._load(validate=validate)

    def _load(self, validate: bool = False) -> None:
        p = Path(self.path)
        if not p.exists():
            raise DatasetError(f"Dataset file not found: {self.path}")
        with p.open('r', encoding='utf-8') as f:
            raw = json.load(f)

        entries = raw.get('data', [])
        if validate:
            self._validate_structure(entries)

        self._raw_entries = entries
        self._data = []

        for entry in entries:
            if entry is None:
                if not self.skip_nulls:
                    self._data.append(None)
                continue

            # parse droneLocation -> GeoLocation + altitude
            try:
                dl_raw = entry.get('droneLocation', {})
                gl_raw = dl_raw.get('geoLocation', {})
                geo = GeoLocation(latitude=float(gl_raw.get('latitude', 0.0)),
                                  longitude=float(gl_raw.get('longitude', 0.0)))
                altitude = float(dl_raw.get('altitude', 0.0))
                drone_loc = DroneLocation(geo, altitude)

                targets: List[TargetLocation] = []
                for t in entry.get('targetLocations', []):
                    targets.append(TargetLocation(xCoord=float(t.get('xCoord', 0.0)),
                                                  yCoord=float(t.get('yCoord', 0.0))))
            except Exception as exc:  # keep narrow, but surface readable error
                raise DatasetError(f"Error parsing entry: {exc}") from exc

            self._data.append(DataPoint(drone_loc, targets))

    def _validate_structure(self, entries: List[Any]) -> None:
        # Basic validation: ensure each non-null entry has droneLocation and targetLocations
        if not isinstance(entries, list):
            raise DatasetError("`data` must be a JSON array")
        for i, e in enumerate(entries):
            if e is None:
                continue
            if not isinstance(e, dict):
                raise DatasetError(f"Entry {i} must be an object or null")
            if 'droneLocation' not in e:
                raise DatasetError(f"Entry {i} missing 'droneLocation'")
            dl = e.get('droneLocation')
            if not isinstance(dl, dict):
                raise DatasetError(f"Entry {i} 'droneLocation' must be an object")
            gl = dl.get('geoLocation')
            if not isinstance(gl, dict):
                raise DatasetError(f"Entry {i} 'geoLocation' must be an object")
            if 'targetLocations' in e and not isinstance(e.get('targetLocations'), list):
                raise DatasetError(f"Entry {i} 'targetLocations' must be an array if present")

    # --- iteration & indexing ---
    def next(self) -> Optional[DataPoint]:
        """Return the current item and advance the internal index.

        Returns `None` when the current slot is a JSON null (unless skipped).
        Behavior at the end depends on `self.cyclic`:
        - cyclic=True: wraps to the start
        - cyclic=False: returns None once the end is reached and keeps index at end
        """
        if not self._data:
            return None

        dp = self._data[self._idx]
        # advance
        if self.cyclic:
            self._idx = (self._idx + 1) % len(self._data)
        else:
            if self._idx < len(self._data) - 1:
                self._idx += 1
            else:
                # at end, keep index at end (subsequent next() will keep returning last element)
                pass
        return dp

    def peek(self) -> Optional[DataPoint]:
        """Return current item without advancing."""
        if not self._data:
            return None
        return self._data[self._idx]

    def reset(self, index: int = 0) -> None:
        """Reset internal pointer to `index` (default 0)."""
        if not isinstance(index, int) or index < 0:
            raise DatasetError("reset index must be a non-negative integer")
        if self._data:
            self._idx = min(index, len(self._data) - 1)
        else:
            self._idx = 0

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Optional[DataPoint]:
        return self._data[index]

    def __iter__(self) -> Iterator[Optional[DataPoint]]:
        # iterate one full pass over current data (safe for for-loops)
        return iter(self._data.copy())

    def __next__(self) -> Optional[DataPoint]:
        # support iterator protocol for manual iteration via next(iter(ds))
        val = self.next()
        if val is None and not self.cyclic:
            raise StopIteration
        return val

    @property
    def size(self) -> int:
        return len(self._data)

    # --- serialization ---
    def to_dict(self) -> dict:
        out = []
        # prefer original raw entries when available, otherwise build from objects
        if self._raw_entries:
            # convert to raw but reflect possible skip_nulls: if skip_nulls, raw_entries retains nulls but _data may not
            for e in self._raw_entries:
                out.append(e)
        else:
            for e in self._data:
                if e is None:
                    out.append(None)
                else:
                    out.append(e.to_dict())
        return {"data": out}

    def save(self, path: str) -> None:
        """Atomically write dataset to `path`."""
        data_dict = self.to_dict()
        tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)))
        try:
            with os.fdopen(tmp_fd, 'w', encoding='utf-8') as f:
                json.dump(data_dict, f, indent=2)
            shutil.move(tmp_path, path)
        except Exception as exc:
            # cleanup
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            raise DatasetError(f"Failed to save dataset: {exc}") from exc


