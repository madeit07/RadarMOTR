# Trackers

This modules implements trackers to evaluate. The RadarMOTR model implementation itself is located in [models](../models). 

## RadarMOTR

The default tracker.

## SORT

This implements the [Simple, Online and Realtime Tracker (SORT)](https://github.com/abewley/sort) algorithm. It performs a linear Kalman Filter prediction of the future bounding boxes and matches them with the next predicted boxes using IoU and linear sum assignment problem.
