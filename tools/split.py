from utils.split_pickle import bulk_load_depths, bulk_load_sam3
from pathlib import Path
import numpy as np
import pickle
import json

if __name__ == "__main__":
    path = "/Users/rishabhbardia/Documents/workspace/vr-live-x/deploy/data/superbon/annotation/rigid_geometry.json"
    with open(path, "rb") as fh:
        data = json.load(fh)
    # print(data)
    frames = sorted(data)

    tracks = sorted({int(k) for s in frames for k in (data[s].get("points") or {})})

    print(tracks)
    depth_npz = Path("/Users/rishabhbardia/Documents/workspace/vr-live-x/deploy/data/superbon/output_depths.npz")
    parent = depth_npz.parent
    output = parent / "depth"
    output.mkdir(exist_ok=True)
    depth_data = bulk_load_depths(depth_npz)
    print(depth_data.shape)
    frames, H, W = depth_data.shape
    for i in range(frames):
        np.savez_compressed(
            output / f"frame_{i:05d}.npz",
            depths=depth_data[i],
        )

    segmentation_pkl = Path("/Users/rishabhbardia/Documents/workspace/vr-live-x/deploy/data/superbon/sam3/outputs.pkl")
    parent = segmentation_pkl.parent
    output = parent / "segmentation"
    output.mkdir(exist_ok=True)
    segmentation_data = bulk_load_sam3(segmentation_pkl)

    for frame_idx, frame_data in segmentation_data.items():
        with (output / f"frame_{int(frame_idx):05d}.pkl").open("wb") as f:
            pickle.dump(frame_data, f, protocol=pickle.HIGHEST_PROTOCOL)


