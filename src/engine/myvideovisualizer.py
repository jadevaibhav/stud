import numpy as np
import matplotlib as mpl
from scipy.stats import norm, chi2
import torch
from detectron2.utils.video_visualizer import VideoVisualizer
from .myvisualizer import MyVisualizer, ColorMode, _SMALL_OBJECT_AREA_THRESH
from .myvisualizer import random_color

class MyVideoVisualizer(VideoVisualizer):
    """
    Extends detectron2 Video Visualizer to draw corner covariance matrices.
    """

    def __init__(
            self,
            metadata,
            instance_mode=ColorMode.IMAGE):
        super().__init__( metadata,  instance_mode=instance_mode)



    def draw_instance_predictions_odd(self, frame, predictions,energy_threshold=None):
        """
        Draw instance-level prediction results on an image.

        Args:
            frame (ndarray): an RGB image of shape (H, W, C), in the range [0, 255].
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """
        frame_visualizer = MyVisualizer(frame, self.metadata)
        num_instances = len(predictions)
        if num_instances == 0:
            return frame_visualizer.output
        
        max_boxes = 20

        predicted_boxes = predictions.pred_boxes.tensor.cpu().numpy() if predictions.has("pred_boxes") else None
        scores = predictions.scores[0:max_boxes] if predictions.has("scores") else None
        labels = predictions.det_labels[0:max_boxes] if predictions.has("det_labels") else None
        inter_feat = predictions.inter_feat[0:max_boxes] if predictions.has("inter_feat") else None
        if energy_threshold:
            labels[(np.argwhere(
                torch.logsumexp(inter_feat[:, :-1], dim=1).cpu().data.numpy() < energy_threshold)).reshape(-1)] = 10
        '''classes = predictions.pred_classes.numpy() if predictions.has("pred_classes") else None
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None
        colors = predictions.COLOR if predictions.has("COLOR") else [None] * len(predictions)
        periods = predictions.ID_period if predictions.has("ID_period") else None
        period_threshold = self.metadata.get("period_threshold", 0)
        visibilities = (
            [True] * len(predictions)
            if periods is None
            else [x > period_threshold for x in periods]
        )

        if predictions.has("pred_masks"):
            masks = predictions.pred_masks
            # mask IOU is not yet enabled
            # masks_rles = mask_util.encode(np.asarray(masks.permute(1, 2, 0), order="F"))
            # assert len(masks_rles) == num_instances
        else:
            masks = None

        if not predictions.has("COLOR"):
            if predictions.has("ID"):
                colors = self._assign_colors_by_id(predictions)
            else:
                # ToDo: clean old assign color method and use a default tracker to assign id
                detected = [
                    _DetectedInstance(classes[i], boxes[i], mask_rle=None, color=colors[i], ttl=8)
                    for i in range(num_instances)
                ]
                colors = self._assign_colors(detected)
        '''

       # labels = frame_visualizer._create_text_labels(labels, scores, self.metadata.get("thing_classes", None))
        
        '''
        if self._instance_mode == ColorMode.IMAGE_BW:
            # any() returns uint8 tensor
            frame_visualizer.output.reset_image(
                frame_visualizer._create_grayscale_image(
                    (masks.any(dim=0) > 0).numpy() if masks is not None else None
                )
            )
            alpha = 0.3
        else:
            alpha = 0.5

        labels = (
            None
            if labels is None
            else [y[0] for y in filter(lambda x: x[1], zip(labels, visibilities))]
        )  # noqa
        assigned_colors = (
            None
            if colors is None
            else [y[0] for y in filter(lambda x: x[1], zip(colors, visibilities))]
        )  # noqa
        '''

    
    
        if len(scores) == 0 or max(scores) <= 0.0:
            return
        frame_visualizer = frame_visualizer.overlay_covariance_instances(
        labels=labels,
        scores=scores,
        boxes=predicted_boxes[0:max_boxes], covariance_matrices=None,
        score_threshold = 0.0)
        '''
        frame_visualizer.overlay_instances(
            boxes=None if masks is not None else boxes[visibilities],  # boxes are a bit distracting
            masks=None if masks is None else masks[visibilities],
            labels=labels,
            keypoints=None if keypoints is None else keypoints[visibilities],
            assigned_colors=assigned_colors,
            alpha=alpha,
        )
        '''

        return frame_visualizer.output
