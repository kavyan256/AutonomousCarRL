# sem_track.py
import numpy as np
from filterpy.kalman import KalmanFilter

class Track:
    free_ids = []
    next_id = 0

    def __init__(self, bbox, cls_name):
        # Assign ID
        if Track.free_ids:
            self.id = min(Track.free_ids)
            Track.free_ids.remove(self.id)
        else:
            self.id = Track.next_id
            Track.next_id += 1

        # Initialize Kalman filter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        dt = 1.0
        self.kf.F = np.array([[1,0,0,0,dt,0,0],
                              [0,1,0,0,0,dt,0],
                              [0,0,1,0,0,0,dt],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])
        self.kf.R[2:,2:] *= 10.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] = 0.01
        self.kf.Q[4:,4:] = 0.01

        x1, y1, x2, y2 = bbox
        cx = (x1+x2)/2
        cy = (y1+y2)/2
        w = x2-x1
        h = y2-y1
        self.kf.x[:4] = np.array([[cx], [cy], [w], [h]])

        self.cls_name = cls_name
        self.time_since_update = 0
        self.hit_streak = 0

        # Heading vector
        self.prev_center = np.array([cx, cy], dtype=float)
        self.heading_vector = np.array([0.0, 0.0], dtype=float)

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        return self.get_state()

    def update(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1+x2)/2
        cy = (y1+y2)/2
        w = x2-x1
        h = y2-y1
        self.kf.update(np.array([cx, cy, w, h]))

        # Compute heading vector
        new_center = np.array([cx, cy], dtype=float)
        self.heading_vector = new_center - self.prev_center
        self.prev_center = new_center

        self.time_since_update = 0
        self.hit_streak += 1

    def get_state(self):
        cx, cy, w, h = self.kf.x[:4].reshape(-1)
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        return [x1, y1, x2, y2]

class Sort:
    def __init__(self, max_age=3, min_hits=1, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []

    @staticmethod
    def iou(bb_test, bb_gt):
        xx1 = max(bb_test[0], bb_gt[0])
        yy1 = max(bb_test[1], bb_gt[1])
        xx2 = min(bb_test[2], bb_gt[2])
        yy2 = min(bb_test[3], bb_gt[3])
        w = max(0., xx2 - xx1)
        h = max(0., yy2 - yy1)
        wh = w*h
        o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) +
                  (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh + 1e-6)
        return o

    def update(self, dets):
        updated_tracks = []

        # Predict all tracks
        for t in self.tracks:
            t.predict()

        unmatched_dets = []
        unmatched_tracks = list(range(len(self.tracks)))
        matches = []

        iou_matrix = np.zeros((len(dets), len(self.tracks)), dtype=np.float32)
        for d, det in enumerate(dets):
            bbox_det = det[:4]
            for t, trk in enumerate(self.tracks):
                bbox_trk = trk.get_state()
                iou_matrix[d, t] = self.iou(bbox_det, bbox_trk)

        if iou_matrix.size > 0:
            for d in range(len(dets)):
                t = np.argmax(iou_matrix[d])
                if iou_matrix[d, t] >= self.iou_threshold:
                    matches.append((d, t))
                    if t in unmatched_tracks: unmatched_tracks.remove(t)
                else:
                    unmatched_dets.append(d)
        else:
            unmatched_dets = list(range(len(dets)))

        # Update matched tracks
        for d, t in matches:
            self.tracks[t].update(dets[d][:4])
            updated_tracks.append([*self.tracks[t].get_state(),
                                   self.tracks[t].id,
                                   self.tracks[t].cls_name,
                                   self.tracks[t].heading_vector])

        # Create new tracks
        for idx in unmatched_dets:
            det = dets[idx]
            trk = Track(det[:4], det[4])
            self.tracks.append(trk)
            updated_tracks.append([*trk.get_state(),
                                   trk.id,
                                   trk.cls_name,
                                   trk.heading_vector])

        # Remove old tracks and recycle IDs
        alive_tracks = []
        for t in self.tracks:
            if t.time_since_update <= self.max_age:
                alive_tracks.append(t)
            else:
                Track.free_ids.append(t.id)
        self.tracks = alive_tracks

        return updated_tracks
