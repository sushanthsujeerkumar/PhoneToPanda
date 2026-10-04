# Method

The task is to pick up one cube and place it inside an open box. The method has
two stages: convert a measured human object path into robot demonstrations, then
train a policy to imitate those demonstrations.

```text
Phone recordings + camera calibration + grasp/release labels
    → measured object carry paths
    → contact-based Panda replays in simulation
    → successful robot images, states and actions
    → SmolVLA training
    → evaluation at new simulator starts
```

## From video to robot demonstrations

OpenCV detects the printed markers. Four fixed references define a common plane,
and the calibrated camera provides the geometry needed to estimate the moving
object marker's 3D position. The labels identify the part of each recording where
the object is carried.

The measured carry path, its curvature and its timing are mapped into the robot's
workspace. Endpoint alignment, scaling and height limits adapt it to the simulated
cube and box. A controller supplies the robot-specific approach, contact-based
grasp and release. This makes the phone data useful without pretending it already
contains robot actions.

The Panda must move the cube through normal contacts and friction. Initial object
poses are assigned at reset. During the episode, the cube is neither attached to
the gripper nor moved by directly assigning its pose. Only replays that meet the
physical success check enter the training dataset.

The original dataset uses four starts per recording: **20 demonstrations from
five recordings**, with 4,471 frames in total. Demonstration seeds are 0–19.
Different starts are augmentation of those recordings, not additional human data.

## What the model sees and controls

SmolVLA is fine-tuned through LeRobot on the simulated robot observations:

| Input or output | Meaning |
| --- | --- |
| Front RGB image | 256 × 256 view of the simulated workspace |
| Wrist RGB image | 256 × 256 view from the simulated gripper camera |
| Eight state values | End-effector XYZ, three axis-angle orientation values and two gripper joint positions |
| Task text | “Pick up the cube and place it inside the box.” |
| Seven action slots | XYZ movement, three rotation slots and one gripper command |

The policy uses its XYZ and gripper outputs. The three rotation slots are ignored
and replaced by feedback that keeps the wrist pointing downward. This is a
**hybrid controller**: learned translation and grasp commands, fixed orientation
control. Training replays and evaluation use the same convention.

The policy is trained on simulator images, not directly on phone images. The phone
recordings affect training through the retargeted carry paths and timing. The
pipeline does not estimate a full human hand pose or learn unrestricted wrist
rotation.

At evaluation time the scripted demonstration controller is absent. SmolVLA
chooses the translation and gripper commands throughout the episode. It predicts
20 actions at a time and executes five before taking another observation. The
control rate is 20 Hz, and each trial has a limit of 600 steps.

## What counts as success

The cube centre must have been more than 7 cm above the table at least once. For
success, it must then be inside the box bounds, have no gripper contact and move
at less than 0.05 m/s. These checks use simulated position, contact and velocity;
a low training loss does not count as task success.

The scene uses LIBERO's tabletop framework with a custom cube, box and placement
check. It is a small custom task, so its scores should not be compared directly
with official LIBERO benchmark results.

The main limits are the five source trials, one camera setup, one object and one
fixed target. New simulator starts test a limited change in position. They do not
test new objects, independent recording sessions or a real robot.
