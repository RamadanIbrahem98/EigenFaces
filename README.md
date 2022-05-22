# EigenFaces

Use the PCA Eigen Faces Algorithm to Create a Face Detection Classifier

## Downloading the Yale Image Dataset

We Used the Extended Yale Face Database B. Which is already available in the [folowing link](http://vision.ucsd.edu/content/extended-yale-face-database-b-b)

The Cropped Images are already cropped to the size of 192x168. So This Helped us to reduce the steps by not having to do face detection.

in order to download and extracting the dataset, we used the following command in the linux shell:

```sh
wget -co CroppedYale.zip http://vision.ucsd.edu/extyaleb/CroppedYaleBZip/CroppedYale.zip
unzip CroppedYale.zip
rm CroppedYale.zip
```

or simply downloading the zip file and extracting it using the browser.
