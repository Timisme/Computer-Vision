import numpy as np
import cv2 as cv


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv.INTER_AREA):
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv.resize(image, dim, interpolation=inter)


class PerspectiveTransform:
    def __init__(self):
        np.random.seed(42)
        self.im_src = cv.imread('data/Q3_Image/rl.jpg')
        self.cap = cv.VideoCapture("data/Q3_Image/test4perspective.mp4")

    def transform(self):
        while cv.waitKey(1) < 0:
            markerCorners, markerIds, hasFrame = None, None, None
            try:
                # get frame from the video
                hasFrame, frame = self.cap.read()
                original_frame = frame.copy()

                # Load the dictionary that was used to generate the markers.
                dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)

                # Initialize the detector parameters using default values
                parameters = cv.aruco.DetectorParameters_create()

                # Detect the markers in the image
                markerCorners, markerIds, rejectedCandidates = cv.aruco.detectMarkers(
                    frame, dictionary, parameters=parameters
                )

                index = np.squeeze(np.where(markerIds == 25))
                refPt1 = np.squeeze(markerCorners[index[0]])[1]

                index = np.squeeze(np.where(markerIds == 33))
                refPt2 = np.squeeze(markerCorners[index[0]])[2]

                distance = np.linalg.norm(refPt1 - refPt2)

                scalingFac = 0.02
                pts_dst = [
                    [refPt1[0] - round(scalingFac * distance), refPt1[1] - round(scalingFac * distance)]
                ]
                pts_dst += [
                    [refPt2[0] + round(scalingFac * distance), refPt2[1] - round(scalingFac * distance)]
                ]

                index = np.squeeze(np.where(markerIds == 30))
                refPt3 = np.squeeze(markerCorners[index[0]])[0]

                pts_dst += [
                    [refPt3[0] + round(scalingFac * distance), refPt3[1] + round(scalingFac * distance)]
                ]

                index = np.squeeze(np.where(markerIds == 23))
                refPt4 = np.squeeze(markerCorners[index[0]])[0]

                pts_dst += [
                    [refPt4[0] - round(scalingFac * distance), refPt4[1] + round(scalingFac * distance)]
                ]

                pts_src = [[0, 0], [self.im_src.shape[1], 0],
                           [self.im_src.shape[1], self.im_src.shape[0]], [0, self.im_src.shape[0]]]

                pts_src, pts_dst = np.array(pts_src), np.array(pts_dst)

                # Calculate Homography
                homography_matrix, status = cv.findHomography(pts_src, pts_dst)

                # Warp source image to destination based on homography
                im_out = cv.warpPerspective(self.im_src, homography_matrix, (frame.shape[1], frame.shape[0]))

                # Black out polygonal area in destination image.
                cv.fillConvexPoly(frame, pts_dst.astype(int), 0, 16)

                # Add warped source image to destination image.
                frame = frame + im_out

                both = np.concatenate([original_frame, frame], axis=1)
                both = ResizeWithAspectRatio(both, width=1280)  # Resize by width
                cv.imshow('frame', both)

                if cv.waitKey(1) == ord('q'):
                    break

            except:
                if hasFrame:
                    continue
                else:
                    break

        # When everything done, release the capture
        self.cap.release()
        cv.destroyAllWindows()
