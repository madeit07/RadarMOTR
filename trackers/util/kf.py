"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
    This is a modified version by Esslingen University of Applied Sciences.
"""
from __future__ import print_function

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

USE_LAP = True
try:
    import lap
except ImportError:
    from scipy.optimize import linear_sum_assignment
    USE_LAP = False

def linear_assignment(cost_matrix):
    if USE_LAP:
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i],i] for i in x if i >= 0])
    else:
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def convert_rv_to_z(rv):
    return np.array([rv[0], rv[1]]).reshape((2, 1))


def convert_x_to_rv(x, score=None):
    if score is None:
        return np.array([x[0], x[1]]).reshape((1, 2))
    else:
        return np.array([x[0], x[1], score]).reshape((1, 3))


def associate_detections_to_trackers(detections, trackers, dist_threshold):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0:
        return np.empty((0,2), dtype=int), np.arange(len(detections)), np.empty((0,3), dtype=int)

    dist_matrix = cdist(detections[:,:2], trackers[:,:2], 'euclidean')

    if min(dist_matrix.shape) > 0:
        matched_indices = linear_assignment(dist_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:,0]:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:,1]:
            unmatched_trackers.append(t)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if dist_matrix[m[0], m[1]] > dist_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
        matches.append(m.reshape(1,2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    def __init__(self, rv: np.ndarray, id: int, dt: float):
        """
        Initializes a tracker using initial bounding box.
        """
        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        # State: [r, v, r', v']
        # where v is doppler velocity, r is range
        # v' is change in velocity (acceleration), r' change of real velocity (not radial/doppler velocity)

        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[2:,2:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.

        self.kf.x[:2] = convert_rv_to_z(rv)
        self.time_since_update: int = 0
        self.id = id
        self.history = []
        self.hits: int = 0
        self.hit_streak: int = 0
        self.age: int = 0

        self.original_idx = rv[2]

    def update(self, rv):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.original_idx = rv[2]
        self.kf.update(convert_rv_to_z(rv))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0

        self.time_since_update += 1
        self.history.append(convert_x_to_rv(self.kf.x))

        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_rv(self.kf.x)


class RDSort:
    def __init__(self, max_age: int = 1, min_hits: int = 3, dist_threshold: float = 5, fps: int = 1):
        """
        Sets key parameters for SORT
        """
        self.max_age: int = max_age
        self.min_hits: int = min_hits
        self.dist_threshold: float = dist_threshold
        self.fps: int = fps

        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count: int = 0
        self.tracker_count: int = 0

    def update(self, dets: np.ndarray = np.empty((0, 3))):
        """
        Params:
        dets - a numpy array of detections in the format [[r,v,idx],[r,v,idx],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 3)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 3))
        to_del = []
        ret = []

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.dist_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :], self.tracker_count, dt=(1 / self.fps))
            self.tracker_count += 1
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]

            if (trk.time_since_update < 1) and ((trk.hit_streak >= self.min_hits) or (self.frame_count <= self.min_hits)):
                ret.append(np.concatenate((d, [trk.id + 1], [trk.original_idx])).reshape(1, -1)) # +1 as MOT benchmark requires positive

            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)

        return np.empty((0,4))

    def reset(self):
        self.trackers = []
        self.frame_count = 0
        self.tracker_count = 0
