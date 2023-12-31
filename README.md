## Intro

This is software for Minjune Song (CMU'25) and Darion Homarioon (CMU'24)'s EMG Grip Force Prediction project. This is an independent effort unaffiliated with any coursework. Currently, our project aims to:
- Find subject-specific grip force normalization variables that can serve as alternatives for Maximum Voluntary Contraction (MVC).
- Investigate how dataset size and composition affects grip force prediction accuracy.

## Roadmap

- [x] Validate the performance of predicting MVC normalized force on the putEMG dataset.
- [ ] Develop our own EMG recording platform + dynamometer to measure grip force.
- [ ] Collect grip force, MVC force, and candidate variable data from TBD participants.
- [ ] Compare performances of predicting grip force normalized by MVC and other candidate variables.

## Validate MVC Normalized Grip Force Prediction Performance

We used the [putEMG](https://biolab.put.poznan.pl/putemg-dataset/) dataset, a super high-resolution EMG dataset from 44 subjects containing MVC measurements, 24 EMG channels, and hand dynamometer force data. We resampled data from 5120hz to 1280hz and segmented into 500ms windows with 250ms overlap (347,697 segments total).

Following [TEMGNet](https://arxiv.org/pdf/2109.12379.pdf), we trained a transformer network with an 8:2 train/validation split over the segments for 50 epochs. We trained two networks for two separate tasks:
- Predicting raw (unormalized) force: average raw force from the last 10 window samples.
- Predicting MVC normalized force: average raw force from the last 10 window samples as a percent of MVC.

<br>

<p align="center">
  <img src="https://github.com/pythonlearner1025/emg/blob/main/raw_grip.png">
  <br>
  <em>Training and validation loss for raw grip force</em>
</p>

<br>

<p align="center">
  <img src="https://github.com/pythonlearner1025/emg/blob/main/norm_grip.png">
  <br>
  <em>Training and validation loss for MVC normalized grip force</em>
</p>


Validation Mean Squared Error (MSE) for raw force prediction was over 10,000, while MVC percentage prediction was between 0.05 and 0.10, highlighting the importance of MVC normalization These results will serve as benchmarks as we develop our own EMG recording platform and dynamometer, and experiment with alternative normalization variables. 

## Replication Steps

First, install the repository:

```bash
git clone https://github.com/pythonlearner1025/emg.git
cd emg
python3 -m pip install -r requirements.txt
```

Run this script to download the putEMG force dataset:

```bash
git clone https://github.com/biolab-put/putemg-downloader.git
cd putemg-downloader
./putemg_downloader.py emg_force data-hdf5
```

Run validation for raw grip force prediction:

```bash
python validate.py --load_data 1 --train 1 --visualize 1 --normalize 0
```

Run validation for MVC normalized grip force prediction:

```bash
python validate.py --load_data 1 --train 1 --visualize 1 --normalize 1
```

For just visualization, set all other arguments to 0