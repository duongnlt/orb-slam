# mini-orbslam

A small ORB-SLAM-style monocular SLAM pipeline in Python built for learning

## Setup

```
conda create -n slam python=3.12 -y
conda activate slam
pip install -e ".[dev]"
pytest -q
```

Datasets (TUM RGB-D, KITTI) in `data/`

## Layout

```
src/slam/    library code (camera, features, geometry, dataset, viz, ...)
tests/       pytest tests with synthetic ground truth
notebooks/   exploration and check scripts
data/        datasets
```

## Plan

- [x] 1. Camera model (`camera.py`)
- [x] 2. ORB detection and matching (`features.py`)
- [x] 3. Two-view geometry: essential matrix, relative pose, triangulation (`geometry.py`)
- [x] 3b. Check the two-view pipeline on real TUM frames (`notebooks/two_view_check.py`)
- [x] 4a. TUM dataset loader (`dataset.py`)
- [ ] 4b. Monocular initialization (parallax / inlier gate)
- [ ] 4c. Frame-to-frame visual odometry
- [ ] 5. Map points, keyframes, motion model, local map tracking
- [ ] 5b. Local bundle adjustment, map point and keyframe culling
- [ ] 6. Loop closure: place recognition, Sim(3), pose-graph optimization
- [ ] 7. Evaluation: ATE/RPE against ground truth (e.g. `evo`)
