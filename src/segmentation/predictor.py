import cv2
from ..pipeline.runtime import check_image
from .masks import candidate, coco_records, expanded_rois, fuse_masks, project_mask

class OverflowSegmenter:
    def __init__(self, runtime):
        self.runtime = runtime
        self.config = runtime.config("segmentation")
    
    @staticmethod
    def _encode(result, rect, image_id, shape):
        if result.masks is None:
            return []
        
        masks = result.masks.data.byte().cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        records = []

        for mask, score in zip(masks, scores):
            rle = project_mask(mask, rect, shape)
            if rle is not None:
                records.append(candidate(image_id, rle, score))
            
        return records
    
    def predict(self, image, image_id=0, *, for_ap=False):
        check_image(image)
        height, width = image.shape[:2]
        config = self.config
        model_ids = config["models"]
        
        proposals = self.runtime.infer(
            model_ids["proposals"],
            image,
            config["proposal_inference"]
        )[0]
        
        full = self.runtime.infer(
            model_ids["full_image"],
            image,
            config["segment_inference"],
            batch=config["full_image_batch"]
        )[0]
        
        records = self._encode(full, (0, 0, width, height), image_id, (height, width))
        rects = expanded_rois(
            proposals.boxes.xyxy.cpu().numpy(),
            width, height,
            config["roi"]
        )
        
        batch_size = config["roi_batch"]
        for start in range(0, len(rects), batch_size):
            batch_rects = rects[start:start + batch_size]
            crops = []

            for left, top, right, bottom in batch_rects:
                crop = image[top:bottom, left:right]
                scale = min(1.0, config["roi"]["stored_long_side"] / max(crop.shape[:2]))
                
                if scale < 1:
                    crop = cv2.resize(crop, (round(crop.shape[1] * scale), round(crop.shape[0] * scale)), interpolation=cv2.INTER_LINEAR)

                crops.append(crop)

            results = self.runtime.infer(
                model_ids["roi"], crops,
                config["segment_inference"], batch=batch_size
            )

            for result, rect in zip(results, batch_rects):
                records.extend(self._encode(result, rect, image_id,(height, width)))

        fused = fuse_masks(records, config["fusion"]["iou"], config["fusion"]["max_det"])
        
        if not for_ap:
            confidence = config["operating_point"]["confidence"]
            fused = [
                pred for pred in fused
                if pred["score"] > confidence
            ]

        return coco_records(fused)