import cv2 as cv
import numpy as np


class OpticalFlow:
    def __init__(self):
        self.detect_interval = 10
        self.tracks = []
        self.prev_gray = None
        self.frame_idx = 0

    @classmethod
    def get_detector(cls):
        # Setup SimpleBlobDetector parameters.
        params = cv.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 10
        params.maxThreshold = 200

        params.filterByColor = True
        params.blobColor = 0

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 32

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.5

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.96

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.5

        # Set up the detector with default parameters.
        detector = cv.SimpleBlobDetector_create(params)
        return detector

    def draw(self):
        cap = cv.VideoCapture("data/Q2_Image/opticalFlow.mp4")

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Done")
                break

            # Detect blobs.
            detector = self.get_detector()
            keypoints = detector.detect(frame)

            for i in range(1, len(keypoints)):
                x, y = np.int(keypoints[i].pt[0]), np.int(keypoints[i].pt[1])
                sz = np.int(keypoints[i].size)
                if sz > 1:
                    sz = np.int(sz / 2)
                # notice there's no boundary check for pt1 and pt2, you have to do that yourself
                frame = cv.line(frame, (x, y - sz), (x, y + sz), color=(0, 0, 255), thickness=1)
                frame = cv.line(frame, (x - sz, y), (x + sz, y), color=(0, 0, 255), thickness=1)
                frame = cv.rectangle(frame, (x - sz, y - sz), (x + sz, y + sz), color=(0, 0, 255), thickness=1)

            cv.imshow("Keypoints", frame)
            if cv.waitKey(1) == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()

    def flow(self):
        cap = cv.VideoCapture("data/Q2_Image/opticalFlow.mp4")

        while True:
            _ret, frame = cap.read()
            if not _ret:
                print("Done")
                break
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            vis = frame.copy()

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, winSize=(11, 11))
                p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, winSize=(11, 11))
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1

                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue

                    tr.append((x, y))
                    new_tracks.append(tr)
                    cv.circle(vis, (x, y), 2, (0, 0, 255), -1)

                self.tracks = new_tracks
                cv.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 0, 255), 2)

            if len(self.tracks) < 10:
                detector = self.get_detector()
                keypoints = detector.detect(frame)
                if keypoints is not None:
                    for kp in keypoints:
                        for x, y in np.float32(kp.pt).reshape(-1, 2):
                            self.tracks.append([(x, y)])

            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv.imshow('lk_track', vis)

            if cv.waitKey(1) == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    OpticalFlow().draw()
    OpticalFlow().flow()
