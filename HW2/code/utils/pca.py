import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob

# 只有一張圖片時
# from sklearn.decomposition import PCA
# pca_r, pca_g, pca_b = PCA(0.99), PCA(0.99), PCA(0.99)
# img = cv.imread("data/Q4_Image/1.jpg")
# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# rm, gm, bm = img[:, :, 0], img[:, :, 1], img[:, :, 2]
# rmm, gmm, bmm = pca_r.fit_transform(rm), pca_g.fit_transform(gm), pca_b.fit_transform(bm)
# rmm, gmm, bmm = pca_r.inverse_transform(rmm), pca_g.inverse_transform(gmm), pca_b.inverse_transform(bmm)
# rmm, gmm, bmm = np.uint8(np.absolute(rmm)), np.uint8(np.absolute(gmm)), np.uint8(np.absolute(bmm))
# recon = np.dstack((rmm, gmm, bmm))
# imgplot = plt.imshow(recon)
# plt.show()


def pca_reconstruction(image, n_features=34):

    # This function is equivalent to:
    from sklearn.decomposition import PCA
    

    img_avg = np.expand_dims(np.mean(image, axis=1), axis=1)
    X = image - img_avg

    pca = PCA(n_features)
    recon = pca.fit_transform(image)
    recon = pca.inverse_transform(recon)
    # U, S, VT = np.linalg.svd(X, full_matrices=False)
    # recon = img_avg + np.matmul(np.matmul(U[:, :n_features], U[:, :n_features].T), X)
    return np.uint8(np.absolute(recon))


def reconstruction_error(gray):
    gray = gray.reshape(gray.shape[0], -1)
    gray_recon = pca_reconstruction(gray)
    re = np.sum(np.abs(gray - gray_recon), axis=1)
    return np.mean(re)


class PCA:
    def __init__(self):
        np.random.seed(42)
        self.img_path = sorted(glob.glob("data/Q4_Image/*.jpg"), key=lambda x: int(x.split("/")[1].split(".")[0].split('\\')[1]))
        self.gray_img = None
        self.original_img = None

    def reconstruction(self):
        for img_path in self.img_path:
            if self.gray_img is None:
                img = cv.imread(img_path)
                self.original_img = np.expand_dims(cv.cvtColor(img, cv.COLOR_BGR2RGB), axis=0)
                self.gray_img = np.expand_dims(cv.cvtColor(img, cv.COLOR_BGR2GRAY), axis=0)
            else:
                img = cv.imread(img_path)
                img1 = np.expand_dims(cv.cvtColor(img, cv.COLOR_BGR2RGB), axis=0)
                img2 = np.expand_dims(cv.cvtColor(img, cv.COLOR_BGR2GRAY), axis=0)
                self.original_img = np.concatenate((self.original_img, img1), axis=0)
                self.gray_img = np.concatenate((self.gray_img, img2), axis=0)

        img_shape = self.original_img.shape
        r = self.original_img[:, :, :, 0].reshape(img_shape[0], -1)
        g = self.original_img[:, :, :, 1].reshape(img_shape[0], -1)
        b = self.original_img[:, :, :, 2].reshape(img_shape[0], -1)
        r_r, r_g, r_b = pca_reconstruction(r), pca_reconstruction(g), pca_reconstruction(b)
        recon_img = np.dstack((r_r, r_g, r_b))
        recon_img = np.reshape(recon_img, img_shape)

        # Setup a figure 6 inches by 6 inches
        fig = plt.figure(figsize=(17, 4))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

        length = 17
        for i in range(length):
            ax1 = fig.add_subplot(4, length, i + 1, xticks=[], yticks=[])
            ax1.imshow(self.original_img[i], cmap=plt.cm.bone, interpolation='nearest')
            ax2 = fig.add_subplot(4, length, i + length + 1, xticks=[], yticks=[])
            ax2.imshow(recon_img[i], cmap=plt.cm.bone, interpolation='nearest')
            ax3 = fig.add_subplot(4, length, i + length * 2 + 1, xticks=[], yticks=[])
            ax3.imshow(self.original_img[i + length], cmap=plt.cm.bone, interpolation='nearest')
            ax4 = fig.add_subplot(4, length, i + length * 3 + 1, xticks=[], yticks=[])
            ax4.imshow(recon_img[i + length], cmap=plt.cm.bone, interpolation='nearest')

        plt.show()

    def get_error(self):
        if self.gray_img is None:
            print("Please use pca to reconstruct images first")
            return

        total_error = []
        for img in self.gray_img:
            error = reconstruction_error(img)
            total_error.append(error)

        print("Reconstruction Error:", total_error)


if __name__ == "__main__":
    pca = PCA()
    pca.reconstruction()
    pca.get_error()

    # print(glob.glob("../data/Q4_Image/*.jpg")[0].split('/')[2].split('.')[0].split('\\')[1])
    # print(sorted(glob.glob("../data/Q4_Image/*.jpg"), key=lambda x: int(x.split("/")[2].split(".")[0].split('\\')[1])))

