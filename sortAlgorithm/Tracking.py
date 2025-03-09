"""
Bu projede nesne takibi için SORT  algoritması kullanılmıştır.
SORT, Kalman Filtresi ve IOU metriği ile çoklu nesne takibini gerçekleştiren gerçek zamanlı bir takip algoritmasıdır
Kaynak https://github.com/abewley/sort
Teşekkürler Alex Bewley
"""
import numpy as np
from .sort import Sort # . koyulmayınca hata alıyoruz


class FaceTracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.tracker = Sort(max_age, min_hits, iou_threshold)

    def update(self, detections):
        if len(detections) == 0:
            return np.empty((0, 5))

        tracked_objects = self.tracker.update(np.array(detections))
        return tracked_objects
