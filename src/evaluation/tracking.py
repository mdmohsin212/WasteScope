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
        similarity *= g.ClassID.to_numpy()[:, None] == p.ClassID.to_numpy()[None, :]
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
        gt[name] = ((gt[center] + sign * gt[size] / 2) * gt[dimension]).clip(lower=0, upper=gt[dimension])

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

        counted = predictions[predictions.Sequence.eq(sequence)]
        
        if "CountEligible" in counted.columns:
            counted = counted[counted["CountEligible"]]

        predicted_count = int(counted.TrackID.nunique())
        error = predicted_count - data["num_gt_ids"]

        rows.append({
            "Sequence": sequence,
            **_scores(sequence_results),
            "GT": data["num_gt_ids"],
            "RawTracks": data["num_tracker_ids"],
            "Predicted": predicted_count,
            "CountError": error,
            "AbsCountError": abs(error)
        })

    per_sequence = pd.DataFrame(rows).set_index("Sequence")
    combined = {
        name: metric.combine_sequences(results[name])
        for name, metric in metrics.items()
    }
    overall = pd.DataFrame([{
        **_scores(combined),
        "GT": int(per_sequence.GT.sum()),
        "RawTracks": int(per_sequence.RawTracks.sum()),
        "Predicted": int(per_sequence.Predicted.sum()),
        "MAE": per_sequence.AbsCountError.mean(),
        "RMSE": np.sqrt(np.mean(per_sequence.CountError ** 2)),
        "SAD": int(per_sequence.AbsCountError.sum()),
        "MAPE": 100 * (per_sequence.AbsCountError / per_sequence.GT.replace(0, np.nan)).mean()
    }])
    
    return overall, per_sequence


def postprocess_tracks(tracks, frames, min_frames=10, max_gap=5, max_distance=0.05):
    result = tracks.copy()
    result["OriginalTrackID"] = result["TrackID"]

    for sequence, group in tracks.groupby("Sequence", sort=True):
        shape = frames.loc[frames.Sequence.eq(sequence)].iloc[0]
        scale = np.array([shape.Width, shape.Height], dtype=float)
        segments = []

        for track_id, track in group.groupby("TrackID"):
            track = track.sort_values("Frame")
            boxes = track[["x1", "y1", "x2", "y2"]].iloc[[0, -1]].to_numpy()
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2 / scale

            segments.append({
                "id": int(track_id),
                "start": int(track.Frame.iloc[0]),
                "end": int(track.Frame.iloc[-1]),
                "class": int(track.ClassID.mode().iloc[0]),
                "first": centers[0],
                "last": centers[1]
            })

        roots = []
        for segment in sorted(segments, key=lambda x: (x["start"], x["id"])):
            candidates = []

            for index, root in enumerate(roots):
                gap = segment["start"] - root["end"] - 1
                distance = float(np.linalg.norm(segment["first"] - root["last"]))
                
                if (root["class"] == segment["class"] and 0 <= gap <= max_gap and distance <= max_distance):
                    candidates.append((distance, gap, root["id"], index))

            if candidates:
                _, _, root_id, index = min(candidates)
                mask = (result.Sequence.eq(sequence) & result.TrackID.eq(segment["id"]))
                result.loc[mask, "TrackID"] = root_id
                roots[index].update(end=segment["end"], last=segment["last"])
            else:
                roots.append(segment.copy())

    hits = result.groupby(["Sequence", "TrackID"])["Frame"].transform("nunique")
    result["CountEligible"] = hits >= min_frames
    return result