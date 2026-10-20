# Live-X

See combat sports in full immersion from the ring side seat that did not exist!

 ![Made for VR Headset](../results/vr180_fighters_stereo_vr_9_20_26.mp4)

LiveX explores turning existing combat-sports broadcasts into immersive VR experiences that give viewers a virtual ringside seat. The goal is to make the scale, proximity, and intensity of a fight feel tangible through carefully chosen viewpoints. Using footage already captured, LiveX aims to let you revisit your favourite fights as though you were there.

## Why a moved seat beats adding depth at the source?

Platform 2D-to-3D conversion adds depth at the broadcast camera: you still watch from far away, through a long lens, now in stereo. Live-X moves you instead, to a low seat at ringside that no camera occupied. From there the fighters fill your view and rise above your eye line: giants in a cathedral, standing in a dark space with no edges rather than inside a frame. The bet is that where you sit matters more for presence than how much depth the picture has. A cage of loosely known size is what makes it possible, because it calibrates the camera and the scale.

## Pipeline

* Calibrate the camera and sparse scene structure against loosely known geometry.
* Track landmarks from previous step to get per-frame poses.
* Use the solved scene geometry plane structure and poses to fit metric depth on monocular disparity.
* Render a novel view using the 2.5D shell of the fighters using the segmentation inputs.
* [On going] Broadcast camera motion model selector.
* [Experimental] MHR body pose to accumulate per-sequence data in single coordinate frame.

## Results

| Experiment     | Question It Answers                                                                   | Status | Result |
| -------------- | ------------------------------------------------------------------------------------- | ------ | ------ |
| Seat Volume    | With this reconstruction what is the usable<br />volume where the seat can be placed? |        |        |
| Stage Mode     | Does adding the cage/floor change the presence compared to black infinity?            |        |        |
| Window vs Seat | Does moving the viewer beat adding depth at the broadcast camera?<br />               |        |        |

### Calibration Experiments/Testing

| Experiment                        | Question It Answers                                                               | Status  | Result                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| --------------------------------- | --------------------------------------------------------------------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Calibration uncertainty           | How well is the calibration constrained, and what limits it?                      | Done    | Reprojection error median : 1.82, 1.46 and 1.15 for pillar base, pillar top and canvas points respectively in pixels.<br /><br />The uncertainty on canvas landmarks is shown in [uncertainty_Q.png](../results/uncertainty_Q.png) which is coming from mainly the camera center which is not pinned very well.<br /><br />Reduced chi-square : 0.335, so ellipses are conservatie upper bound.<br /><br />A landmark uncertainty drops from 33 cm (95% radius) to 6 cm if C is conditioned out i.e. camera center were known exactly. |
| Fixed center assumption           | How well does a per-frame free camera center explain the observations?            | Done    | Independently refined each frame's camera pose for six frames while holding the calibrated scene geometry and intrinsics fixed.<br /><br /> Estimated camera-center displacements were 3–12 mm (median 8 mm). <br /><br />Squared Mahalanobis distances (0.14–3.29) were below the 95% chi-square threshold (7.815) for 3 degrees of freedom, indicating no significant evidence of camera translation under the conditional covariance model. <br /><br />These results support the rotation-only camera assumption for this clip.  |
| Check against held-out post edges | Is the calibration right or just fitted?                                          | Planned |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Does the tracker drift?           | How far does the tracked pose wander from the truth, and how much is just jitter? |         |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Is the metric depth right?        | Does the fitted depth hold on surfaces and people it wasn't fit to?               |         |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |

## Limitations

Tested on one clip right now.


## How to run(ish)?

* Get the data from [here](../results/clip.json).
* Run Video Depth Anthing, SAM3 and optionally SAM3D-Body on the frames.
* The annotated file has 6 key frames annotated which match the frames written out as described by clip.json
* After that run :
  * `calibrate_stage`
  * `track_poses`
  * `fit_depth`
  * `render_demo`
* Use same output directory as each stage adds some artifact the next stage uses.
