# FACE ID

Board-side Qt face detection, recognition, and enrollment application for a USB
camera running on the Torq runtime and Synaptics NPU.

## Setup and Run

From the repository root, activate the virtual environment and install the
repository and Face ID requirements:

```sh
python3 -m pip install -r Face_ID/requirements.txt
```

After installing the requirements, run setup once before the first launch. This
downloads or refreshes the three VMFB files from Hugging Face into the
repository-level `models/` directory:

```sh
python3 setup_demos.py face_id
```

After setup completes, start the Qt application:

```sh
python3 Face_ID/run_qt_app.py
```

`PyQt6` is provided by the board image and is intentionally not listed in the
demo `requirements.txt`.

### Registering faces one at a time

Use this sequence for each person:

1. Make sure the person's face is visible in the camera view.
2. Click `REGISTER FACES`.
3. Wait for the visible face cards to appear.
4. Click `SELECT` on the card for the person you want to enroll.
5. Enter the person's name in that card's name field.
6. Click `ENROLL SELECTED FACE`.
7. Keep that face visible and steady while the application collects the target
        number of samples.
8. Wait until the card changes to `ENROLLED`.
9. To add another person, click `ADD FACE` and repeat the sequence.

Only the selected card is enrolled. Other visible faces do not need names and
do not block the selected person's enrollment. An enrolled card is locked and
cannot be enrolled again.

The application prevents re-enrollment when the face embedding already matches
an identity in the database. It also rejects duplicate names and keeps the
existing registered identity unchanged.

## Shutdown

Press `Ctrl+C` in the terminal. The application stops the camera, closes the
Qt window, restores the NPU frequency, closes the JSON results writer, and exits.
