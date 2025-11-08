import sys

import zarr

sys.path.append(".")

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

register_codecs()
path = "data/tasks/PullTissue/PullTissue-120.zarr.zip"
store = zarr.ZipStore(path, mode="r")
root = zarr.group(store=store)


def walk(group, prefix=""):
    for name, item in group.items():
        if isinstance(item, zarr.Array):
            print(f"{prefix}{name:25s} shape={item.shape} dtype={item.dtype}")
        elif isinstance(item, zarr.Group):
            walk(item, prefix=prefix + name + "/")


print("All keys in zarr:")
walk(root)
