# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
from .nn_matching import NearestNeighborDistanceMetric

class Tracker:
    """
    This is the multi-target tracker.
    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
        用于测量跟踪关联的距离度量。

    max_age : int
        Maximum number of missed misses before a track is deleted.
        删除轨迹之前错过的最大错过次数。

    n_init : int
        Number of frames that a track remains in initialization phase.
        轨道在初始化阶段保留的帧数。

    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
        卡尔曼滤波器用于过滤图像空间中的目标轨迹。

    tracks : List[Track]
        The list of active tracks at the current time step.
        当前时间步的活动轨迹列表。

    """

    def __init__(self, max_iou_distance=0.7, max_age=30, n_init=3):
        '''
        發現問題
        n_init 是多少決定著後續deep_sort會不會報錯，which說 n_init要和圖片上的識別數量對齊。

        '''
        metric =  NearestNeighborDistanceMetric("cosine", matching_threshold=0.7)
        self.metric = metric

        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            # --------
            # print('detection_indices',detection_indices)
            # print('track_indices',track_indices)
            # 這裡做一個截斷，只要detection的長度 
            stop = len(detection_indices)
            # --------
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices[0:stop]]) # [0:stop]截斷·
            # print('features.shape = ,targets.shape = ',features.shape,targets.shape)
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices[0:stop],# [0:stop]截斷·
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [    
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
        

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade( 
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)


        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)
        
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        '''終於找到self.n_init的問題所在'''
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))        
        self._next_id += 1


# 2023.7.28 更新 尝试就用xy来获得预测值
if __name__ =="__main__":
    tracker = Tracker()
    # detections = [[(945, 363), 0.5414349], [(481, 428), 0.5572026], [(719, 264), 0.6236253], [(256, 508), 0.6116705], [(1098, 162), 0.5472754], [(939, 160), 0.71529096], [(637, 342), 0.51113856], [(691, 307), 0.56295353], [(1014, 159), 0.5815531], [(1212, 98), 0.632132], [(1095, 98), 0.45850858], [(1052, 119), 0.63280493]]
    # print(detections)
