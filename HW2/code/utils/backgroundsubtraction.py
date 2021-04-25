import numpy as np
import cv2 as cv

class BackgroundSubtraction:
	def __init__(self):
		np.random.seed(42)
		self.video = cv.VideoCapture("data/Q1_Image/bgSub.mp4")

	def gaussian(self,img, height, width, frame_len):
		mean = []
		std = []

		for j in range(height):
			mean_tmp = []
			std_tmp = []
			for i in range(width):

				sample = [img[n][j][i] for n in range(frame_len)]

				gaussian = np.random.normal(np.mean(sample), np.maximum(np.std(sample), 5.0), frame_len)
				mean_tmp.append(np.mean(gaussian))
				std_tmp.append(np.std(gaussian))
			mean.append(mean_tmp)
			std.append(std_tmp)

		return np.array(mean), np.array(std)

	def subtract(self):
		img_frame = []
		if not self.video.isOpened():
			print("cannot load video")
			exit(1)
		while len(img_frame) < 50:
			# Capture frame-by-frame
			ret, frame = self.video.read()
			# if frame is read correctly ret is True
			if not ret:
				print("cannot load image")
				exit(1)
			gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
			img_frame.append(gray)

		img_height, img_width = img_frame[0].shape
		mean, std = self.gaussian(img_frame, img_height, img_width, 50)

		fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
		fps = self.video.get(cv.CAP_PROP_FPS)
		# out = cv.VideoWriter('output.mp4', fourcc, fps, (img_width, img_height))

		while True:
			# Capture frame-by-frame
			ret, frame = self.video.read()
			# if frame is read correctly ret is True
			if not ret:
				print("Done")
				break

			gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
			new = np.empty([img_height, img_width], dtype="uint8")  # 一定要uint8合併時的影片才能正常呈現

			for j in range(img_height):
				for i in range(img_width):
					if abs(gray[j][i] - mean[j][i]) > 5.0 * std[j][i]:
						new[j][i] = 255.0
					else:
						new[j][i] = 0.0

			new = np.expand_dims(new, axis=-1)
			new = cv.cvtColor(new, cv.COLOR_GRAY2BGR)

			both = np.concatenate([frame, new], axis=1)
			cv.imshow('frame', both)
			# out.write(both)  # output file always fail to open

			if cv.waitKey(1) == ord('q'):
				break

		# When everything done, release the capture
		self.video.release()
		cv.destroyAllWindows()
