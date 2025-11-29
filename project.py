import cv2
import numpy as np

# Path to your video file (must be in same folder as this script)
VIDEO_PATH = "input.mp4"   # change this if your file name is different

# How big the OpenCV windows are allowed to be (for your screen)
MAX_DISPLAY_WIDTH = 960
MAX_DISPLAY_HEIGHT = 540


def detect_fast_features(gray_frame, roi):
    """
    Find good corner points inside the selected box using FAST.
    Returns an array of points in the format LK optical flow expects.
    """
    x, y, w, h = roi

    # Just look at the region inside the box
    roi_gray = gray_frame[y:y+h, x:x+w]

    # FAST corner detector
    fast = cv2.FastFeatureDetector_create(
        threshold=25,
        nonmaxSuppression=True
    )

    keypoints = fast.detect(roi_gray, None)
    if len(keypoints) == 0:
        return None

    # Convert keypoints from "ROI coordinates" to full-frame coordinates
    pts = []
    for kp in keypoints:
        px = kp.pt[0] + x
        py = kp.pt[1] + y
        pts.append([px, py])

    # Shape that calcOpticalFlowPyrLK wants: (N, 1, 2)
    pts = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    return pts


def main():
    # Open the video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: could not open video {VIDEO_PATH}")
        return

    # Grab the first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Error: could not read first frame from video")
        cap.release()
        return

    # Convert first frame to grayscale for feature detection
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Figure out a scale so the window fits on your screen
    h0, w0 = first_frame.shape[:2]
    scale = min(
        1.0,
        MAX_DISPLAY_WIDTH / float(w0),
        MAX_DISPLAY_HEIGHT / float(h0)
    )

    # Show a scaled-down version of the first frame so you can draw a box
    disp_w0 = int(w0 * scale)
    disp_h0 = int(h0 * scale)
    first_disp = cv2.resize(first_frame, (disp_w0, disp_h0))

    cv2.namedWindow("Select Object", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Object", disp_w0, disp_h0)

    print("Draw a box around the object, press ENTER or SPACE, then ESC.")
    roi_disp = cv2.selectROI(
        "Select Object",
        first_disp,
        fromCenter=False,
        showCrosshair=True
    )
    cv2.destroyWindow("Select Object")

    x_disp, y_disp, w_disp, h_disp = roi_disp
    if w_disp == 0 or h_disp == 0:
        print("Error: ROI has zero width/height.")
        cap.release()
        return

    # Convert the box from "display coordinates" back to the original frame size
    x = int(x_disp / scale)
    y = int(y_disp / scale)
    w = int(w_disp / scale)
    h = int(h_disp / scale)
    roi = (x, y, w, h)

    # ---- Angel's part: get initial feature points in the box ----
    points_prev = detect_fast_features(first_gray, roi)

    # If FAST didn't find enough corners, fall back to Shi-Tomasi
    if points_prev is None or len(points_prev) < 10:
        print("FAST found too few points, using goodFeaturesToTrack instead.")
        mask = np.zeros_like(first_gray)
        mask[y:y+h, x:x+w] = 255  # only look inside the selected box
        points_prev = cv2.goodFeaturesToTrack(
            first_gray,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=5,
            mask=mask
        )

    if points_prev is None or len(points_prev) == 0:
        print("Error: could not find any points to track.")
        cap.release()
        return

    prev_gray = first_gray.copy()

    # ---- Varshney's part: Lucas-Kanade tracking ----

    # LK parameters (you can mention these in the report as “standard settings”)
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01
        )
    )

    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)

    while True:
        # Read next frame
        ret, frame = cap.read()
        if not ret:
            # End of video
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # If we somehow lost all points, just tell the user
        if points_prev is None or len(points_prev) == 0:
            cv2.putText(frame, "No points to track", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            display = cv2.resize(
                frame,
                (
                    int(frame.shape[1] * scale),
                    int(frame.shape[0] * scale)
                )
            )
            cv2.imshow("Tracking", display)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            prev_gray = frame_gray.copy()
            continue

        # Track the points from the previous frame to this frame
        points_next, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            frame_gray,
            points_prev,
            None,
            **lk_params
        )

        # Keep only the points that actually tracked successfully
        good_new = np.empty((0, 2), dtype=np.float32)
        if points_next is not None and status is not None:
            good_new = points_next[status.flatten() == 1]

        # If we still have enough good points, update the box
        if good_new is not None and len(good_new) > 5:
            # Make sure shape is (N, 2)
            good_new_flat = good_new.reshape(-1, 2)

            # Draw the tracked points as small green circles
            for (cx, cy) in good_new_flat.astype(np.int32):
                cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)

            # Make a bounding box around all the tracked points
            good_new_reshaped = good_new_flat.reshape(-1, 1, 2)
            x, y, w, h = cv2.boundingRect(good_new_reshaped)

            # Draw a red rectangle for the object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # These points become “previous” for the next frame
            points_prev = good_new_reshaped
        else:
            # Too few points left → tracking basically failed
            cv2.putText(frame, "Lost track (too few points)", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            points_prev = None

        # Store this frame as "previous" for the next loop
        prev_gray = frame_gray.copy()

        # Shrink frame for display so it fits on your screen
        h_frame, w_frame = frame.shape[:2]
        disp_w = int(w_frame * scale)
        disp_h = int(h_frame * scale)
        display = cv2.resize(frame, (disp_w, disp_h))

        cv2.resizeWindow("Tracking", disp_w, disp_h)
        cv2.imshow("Tracking", display)

        # Press 'q' to quit early
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
