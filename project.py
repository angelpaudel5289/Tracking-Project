import cv2
import numpy as np

# Path to your video file (must be in same folder as this script)
VIDEO_PATH = "input.mp4"  # change this if your file name is different

# Limits on how big the display window can be (so it fits on your screen)
MAX_DISPLAY_WIDTH = 960
MAX_DISPLAY_HEIGHT = 540


def detect_fast_features(gray_frame, roi):
    """
    Detect corner-like interest points using FAST INSIDE a given ROI.

    Parameters
    ----------
    gray_frame : np.ndarray
        Single-channel (grayscale) image from the video.
    roi : tuple (x, y, w, h)
        Rectangle defining the region of interest in the original image.

    Returns
    -------
    points : np.ndarray or None
        Array of shape (N, 1, 2) of float32, containing (x, y) positions
        of detected points in the FULL image coordinate system.
        Returns None if no points were found.
    """
    x, y, w, h = roi

    # Extract the region of interest from the full frame
    roi_gray = gray_frame[y:y + h, x:x + w]

    # Create a FAST detector object.
    # - threshold: how "strong" a corner has to be
    # - nonmaxSuppression: keep only the best corners locally
    fast = cv2.FastFeatureDetector_create(
        threshold=25,
        nonmaxSuppression=True
    )

    # Run FAST on just the ROI
    keypoints = fast.detect(roi_gray, None)

    # If no corners found, we signal failure
    if len(keypoints) == 0:
        return None

    # Convert keypoints from ROI coordinates to full-image coordinates
    points = []
    for kp in keypoints:
        px = kp.pt[0] + x  # add ROI offset in x
        py = kp.pt[1] + y  # add ROI offset in y
        points.append([px, py])

    # Convert to the shape expected by calcOpticalFlowPyrLK: (N, 1, 2)
    points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    return points


def main():
    # ---------------------------------------------------------------------
    # 1. Open the video file and read the first frame
    # ---------------------------------------------------------------------
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: could not open video {VIDEO_PATH}")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("Error: could not read first frame from video")
        cap.release()
        return

    # Convert first frame to grayscale (Lucas–Kanade works on grayscale)
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # ---------------------------------------------------------------------
    # 2. Compute a scale factor so the selection window fits on your screen
    # ---------------------------------------------------------------------
    frame_h, frame_w = first_frame.shape[:2]

    # Scale factor so the frame fits inside MAX_DISPLAY_WIDTH x MAX_DISPLAY_HEIGHT
    scale = min(
        1.0,
        MAX_DISPLAY_WIDTH / float(frame_w),
        MAX_DISPLAY_HEIGHT / float(frame_h)
    )

    # Size of the displayed (resized) frame
    disp_w = int(frame_w * scale)
    disp_h = int(frame_h * scale)
    first_disp = cv2.resize(first_frame, (disp_w, disp_h))

    # ---------------------------------------------------------------------
    # 3. Ask the user to draw a bounding box around the object
    # ---------------------------------------------------------------------
    cv2.namedWindow("Select Object", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Object", disp_w, disp_h)

    print("Draw a box around the object, press ENTER or SPACE, then ESC.")
    roi_disp = cv2.selectROI(
        "Select Object",
        first_disp,
        fromCenter=False,
        showCrosshair=True
    )
    cv2.destroyWindow("Select Object")

    x_disp, y_disp, w_disp, h_disp = roi_disp

    # If the user cancels or draws an empty box, abort
    if w_disp == 0 or h_disp == 0:
        print("Error: ROI has zero width/height.")
        cap.release()
        return

    # Convert the box from display coordinates back to original frame size
    x = int(x_disp / scale)
    y = int(y_disp / scale)
    w = int(w_disp / scale)
    h = int(h_disp / scale)
    roi = (x, y, w, h)

    # ---------------------------------------------------------------------
    # 4. Detect initial feature points inside the ROI
    #    First try FAST, then fallback to goodFeaturesToTrack if needed
    # ---------------------------------------------------------------------
    points_prev = detect_fast_features(first_gray, roi)

    # If FAST finds too few points, use Shi–Tomasi "good features to track"
    if points_prev is None or len(points_prev) < 10:
        print("FAST found too few points, using goodFeaturesToTrack instead.")

        # Create a mask so we only find features inside the selected box
        mask = np.zeros_like(first_gray, dtype=np.uint8)
        mask[y:y + h, x:x + w] = 255

        points_prev = cv2.goodFeaturesToTrack(
            first_gray,
            maxCorners=200,      # upper bound on number of corners
            qualityLevel=0.01,   # minimum accepted corner quality
            minDistance=5,       # minimum distance between corners
            mask=mask
        )

    # If we still have no points, there's nothing to track
    if points_prev is None or len(points_prev) == 0:
        print("Error: could not find any points to track.")
        cap.release()
        return

    # Keep a copy of the previous grayscale frame for optical flow
    prev_gray = first_gray.copy()

    # ---------------------------------------------------------------------
    # 5. Set up Lucas–Kanade (KLT) parameters
    #    - winSize: size of the patch around each point
    #    - maxLevel: number of pyramid levels (0..maxLevel)
    # ---------------------------------------------------------------------
    win_size = (21, 21)  # typical LK/KLT window size
    max_level = 3        # 4 levels total: 0,1,2,3 (coarse-to-fine)

    lk_params = dict(
        winSize=win_size,
        maxLevel=max_level,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,   # max number of iterations
            0.01  # stop if movement is smaller than this
        ),
        flags=0,
        minEigThreshold=1e-4  # ignore points where the local structure is too weak
    )

    # Create the tracking window
    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)

    # ---------------------------------------------------------------------
    # 6. Main loop: read frames, track points, and draw bounding box
    # ---------------------------------------------------------------------
    while True:
        # Read next frame from the video
        ret, frame = cap.read()
        if not ret:
            # We have reached the end of the video
            break

        # Convert to grayscale for optical flow
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # If we have lost all points, show message and keep going
        if points_prev is None or len(points_prev) == 0:
            cv2.putText(frame, "No points to track", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            display = cv2.resize(
                frame,
                (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
            )
            cv2.imshow("Tracking", display)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

            prev_gray = frame_gray.copy()
            continue

        # -----------------------------------------------------------------
        # 6a. Track the previous points into the current frame using
        #      pyramidal Lucas–Kanade optical flow
        # -----------------------------------------------------------------
        points_next, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray,       # previous frame (I_{t-1})
            frame_gray,      # current frame (I_t)
            points_prev,     # points in previous frame
            None,            # initial guess for new positions (None = use internal)
            **lk_params
        )

        # -----------------------------------------------------------------
        # 6b. Keep only the points that were successfully tracked
        # -----------------------------------------------------------------
        good_new = np.empty((0, 1, 2), dtype=np.float32)

        if points_next is not None and status is not None:
            # status is 1 for points that tracked successfully, 0 otherwise
            good_new = points_next[status.flatten() == 1].reshape(-1, 1, 2)

        # -----------------------------------------------------------------
        # 6c. If we still have enough points, draw them and update the box
        # -----------------------------------------------------------------
        if good_new is not None and len(good_new) > 5:
            # Draw the tracked points as small green circles
            for pt in good_new:
                cx, cy = int(pt[0, 0]), int(pt[0, 1])
                cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)

            # Compute a bounding box around all the tracked points
            x_box, y_box, w_box, h_box = cv2.boundingRect(good_new)

            # Draw a red rectangle showing the tracked object region
            cv2.rectangle(
                frame,
                (x_box, y_box),
                (x_box + w_box, y_box + h_box),
                (0, 0, 255),
                2
            )

            # These tracked points become the "previous" points for next frame
            points_prev = good_new
        else:
            # We don't have enough points to trust the tracking anymore
            cv2.putText(frame, "Lost track (too few points)", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            points_prev = None

        # Current gray frame becomes "previous" frame for next iteration
        prev_gray = frame_gray.copy()

        # -----------------------------------------------------------------
        # 6d. Resize the frame for display and show it
        # -----------------------------------------------------------------
        disp_w = int(frame.shape[1] * scale)
        disp_h = int(frame.shape[0] * scale)
        display = cv2.resize(frame, (disp_w, disp_h))

        cv2.resizeWindow("Tracking", disp_w, disp_h)
        cv2.imshow("Tracking", display)

        # Press 'q' to quit early
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break

    # ---------------------------------------------------------------------
    # 7. Clean up
    # ---------------------------------------------------------------------
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
