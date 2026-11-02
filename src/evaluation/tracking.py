import numpy as np
import pandas as pd
from trackeval.metrics import HOTA, CLEAR, Identity

def _iou(a, b):
    low = np.maximum(a[:, None, :2], b[None, :, :2])
    high = np.minimum(a[:, None, 2:], b[None, :, 2:])
    
    intersection = np.maximum(high - low, 0).prod(axis=2)
    area_a = np.maximum(a[:, 2:] - a[:, :2], 0).prod(axis=1)
    area_b = np.maximum(b[:, 2:] - b[:, :2], 0).prod(axis=1)
    union = area_a[:, None] + area_b[None, :] - intersection
    
    return np.divide(
        intersection, union,
        out=np.zeros_like(intersection), where=union > 0
    )
    

def _sequence_data(frames, gt, predictions):
    mappings = [
        {track_id: index for index, track_id in enumerate(sorted(table.TrackID.unique()))} for table in (gt, predictions)
    ]
    
    data = {
        "num_timesteps": len(frames),
        "num_gt_ids": len(mappings[0]),
        "num_tracker_ids": len(mappings[1]),
        "num_gt_dets": len(gt),
        "num_tracker_dets": len(predictions),
        "gt_ids": [], "tracker_ids": [], "similarity_scores": []
    }
    groups = [dict(tuple(table.groupby("Frame"))) for table in (gt, predictions)]
    empty = [gt.iloc[:0], predictions.iloc[:0]]
    
    for frame in frames.sort_values("Frame").Frame:
        g, p = [groups[i].get(frame, empty[i]) for i in range(2)]
        
        data["gt_ids"].append(g.TrackID.map(mappings[0]).to_numpy(dtype=int))
        data["tracker_ids"].append(p.TrackID.map(mappings[1]).to_numpy(dtype=int))
        
        similarity = _iou(
            g[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float),
            p[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float),
        )
        similarity *= (
            g.ClassID.to_numpy()[:, None] == p.ClassID.to_numpy()[None, :]
        )
        data["similarity_scores"].append(similarity)
        return data
    

def _scores(results):
    return {
        key: 100 * float(np.mean(results["HOTA"][key]))
        for key in ("HOTA", "DetA", "AssA")
    } | {
        "IDF1": 100 * float(results["Identity"]["IDF1"]),
        "MOTA": 100 * float(results["CLEAR"]["MOTA"]),
        "IDSW": int(results["CLEAR"]["IDSW"])
    }


def evaluate_tracking(frames, annotations, predictions):
    gt = annotations.merge(
        frames[["Sequence", "Frame", "Width", "Height"]],
        on=["Sequence", "Frame"], validate="many_to_one"
    )
    for name, center, size, dimension, sign in (
        ("x1", "xc", "w", "Width", -1), ("y1", "yc", "h", "Height", -1),
        ("x2", "xc", "w", "Width", 1), ("y2", "yc", "h", "Height", 1)
    ):
        gt[name] = (
            (gt[center] + sign * gt[size] / 2) * gt[dimension]
        ).clip(lower=0, upper=gt[dimension])

    metrics = {
        "HOTA": HOTA(),
        "CLEAR": CLEAR({"PRINT_CONFIG": False}),
        "Identity": Identity({"PRINT_CONFIG": False})
    }
    results = {name: {} for name in metrics}
    rows = []

    for sequence, sequence_frames in frames.groupby("Sequence", sort=True):
        data = _sequence_data(
            sequence_frames,
            gt[gt.Sequence.eq(sequence)],
            predictions[predictions.Sequence.eq(sequence)]
        )
        sequence_results = {}
        for name, metric in metrics.items():
            sequence_results[name] = metric.eval_sequence(data)
            results[name][sequence] = sequence_results[name]

        error = data["num_tracker_ids"] - data["num_gt_ids"]
        rows.append({
            "Sequence": sequence, **_scores(sequence_results),
            "GT": data["num_gt_ids"], "Predicted": data["num_tracker_ids"],
            "CountError": error, "AbsCountError": abs(error)
        })

    per_sequence = pd.DataFrame(rows).set_index("Sequence")
    combined = {
        name: metric.combine_sequences(results[name])
        for name, metric in metrics.items()
    }
    overall = pd.DataFrame([{
        **_scores(combined),
        "GT": int(per_sequence.GT.sum()),
        "Predicted": int(per_sequence.Predicted.sum()),
        "MAE": per_sequence.AbsCountError.mean(),
        "RMSE": np.sqrt(np.mean(per_sequence.CountError ** 2)),
        "SAD": int(per_sequence.AbsCountError.sum()),
        "MAPE": 100 * (
            per_sequence.AbsCountError / per_sequence.GT.replace(0, np.nan)
        ).mean()
    }])
    
    return overall, per_sequence