"""Small example showing how to use `DataSet` without embedding a demo in the module."""
from dataset import DataSet


def main():
    ds = DataSet('data.json', cyclic=False, skip_nulls=False, validate=True)
    print(f"Loaded dataset of size: {ds.size}")

    # safe iteration over one pass
    for i, dp in enumerate(ds):
        if dp is None:
            print(f"Entry {i}: <null>")
            continue
        dl = dp.getDroneLocation()
        geo = dl.geoLocation
        print(f"Entry {i}: lat={geo.latitude}, lon={geo.longitude}, alt={dl.altitude}")
        tlocs = dp.getTargetLocations()
        print(f"  {len(tlocs)} targets: {[ (t.xCoord, t.yCoord) for t in tlocs ]}")


if __name__ == '__main__':
    main()
