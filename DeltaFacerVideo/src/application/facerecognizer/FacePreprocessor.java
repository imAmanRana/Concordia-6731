package application.facerecognizer;

import static org.bytedeco.javacpp.opencv_core.CV_32F;
import static org.bytedeco.javacpp.opencv_imgcodecs.IMREAD_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;
import static org.bytedeco.javacpp.opencv_imgproc.getRotationMatrix2D;
import static org.bytedeco.javacpp.opencv_imgproc.resize;
import static org.bytedeco.javacpp.opencv_imgproc.warpAffine;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.RectVector;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier;
import org.bytedeco.javacpp.indexer.FloatRawIndexer;

import application.facedetection.FaceRecognizer;

public class FacePreprocessor {

	CascadeClassifier eyeClassifier;
	CascadeClassifier faceClassifier;

	public FacePreprocessor(CascadeClassifier faceClassifier, CascadeClassifier eyeClassifier) {
		this.faceClassifier = faceClassifier;
		this.eyeClassifier = eyeClassifier;
	}

	public FacePreprocessor(String faceClassifier, String eyeClassifier) {
		this.faceClassifier = new CascadeClassifier(faceClassifier);
		this.eyeClassifier = new CascadeClassifier(eyeClassifier);
	}

	public List<Mat> facepreprocessing(Mat image, RectVector rectVector, Size targetSize, double paddingAmount) {
		List<Mat> faces = null;
		try {

			faces = extractFaces(rectVector, image, targetSize, paddingAmount);
		} catch (Exception e) {
			System.out.println(e.getMessage());
		}

		if (faces == null) {
			return Collections.<Mat>emptyList();
		}
		List<Mat> alignedFaces = new ArrayList<>();
		for (int i = 0; i < faces.size(); i++) {
			try {
				alignedFaces.add(alignFace(faces.get(i)));
			} catch (Exception e) {
			}

		}

		return faces;
	}

	public static void main(String[] args) {

		FaceRecognizer.loadApplicationProperties();
		String faceClassifier = FaceRecognizer.properties.getProperty("FACE_CLASSIFIER");
		String eyeClassifier = FaceRecognizer.properties.getProperty("EYE_CLASSIFIER");

		FacePreprocessor f = new FacePreprocessor(faceClassifier, eyeClassifier);

		Mat image = imread("A:\\CodingStuff\\Eclipse_Workspaces\\Pattern6731\\FaceDetection\\src\\main\\resources\\image.jpg", IMREAD_GRAYSCALE);
		List<Mat> faces = null;

		try (RectVector rv = new RectVector()) {
			Size targetSize = new Size(200, 200);
			faces = f.extractFaces(rv, image, targetSize, 0.2);
		} catch (Exception e) {
			System.out.println(e.getMessage());
		}

		if (faces == null) {
			return;
		}

		for (int i = 0; i < faces.size(); i++) {
			try {
				imwrite("facebefore" + i + ".jpg", faces.get(i));
				imwrite("face" + i + ".jpg", f.alignFace(faces.get(i)));
			} catch (Exception e) {
				System.out.println("error on " + i);
			}

		}
	}

	public Mat alignFace(Mat image) throws Exception {

		// detect eyes
		RectVector rectVector = new RectVector();
		this.eyeClassifier.detectMultiScale(image, rectVector);

		// ensure minimum of 2 eyes detected
		if (rectVector.size() < 2) {
			throw new Exception("Must have at least 2 eyes.");
		}

		// choose 2 eyes from the results such that the angle between them is minimized
		double minAngle = 7.0; // max possible angle is 2π < 7.0
		int mini1 = 1, mini2 = 1;
		for (int i = 0; i < rectVector.size(); i++) {
			for (int j = 0; j < rectVector.size(); j++) {
				double newAngle = Point.angle1(Point.centre(rectVector.get(i)), Point.centre(rectVector.get(j)));
				if (newAngle < minAngle) {
					minAngle = newAngle;
					mini1 = i;
					mini2 = j;
				}
			}
		}

		// calculate eye locations
		Point eye1 = Point.centre(rectVector.get(mini1));
		Point eye2 = Point.centre(rectVector.get(mini2));

		// ensure that eye1 is the left eye
		if (eye1.x > eye2.x) {
			Point tmp = eye1;
			eye1 = eye2;
			eye2 = tmp;
		}

		// calculate rotation matrix
		Point centreRotation = Point.centre(eye1, eye2);
		double angleRotation = Point.angle2(eye1, eye2);

		Mat rotationMatrix = getRotationMatrix2D(centreRotation.toPoint2f(), Math.toDegrees(angleRotation), 1.0);

		// rotate image
		Mat rotatedImage = new Mat();
		warpAffine(image, rotatedImage, rotationMatrix, image.size());

		// calculate rotated eye positions
		double angleRotation2 = -Point.angle2(eye1, eye2);
		eye1.rotate(angleRotation2, centreRotation);
		eye2.rotate(angleRotation2, centreRotation);

		// calculate x translation
		int eyeMid = Point.x_midpoint(eye1, eye2);
		int imageMid = Point.x_midpoint(new Point(0, 0), new Point(image.cols(), image.rows()));
		int xTrans = imageMid - eyeMid;

		// calculate y translation
		// int desiredY = 11 * image.rows() / 32; //a little higher than the middle of
		// the image
		// int yTrans = desiredY - eye1.y;

		// construct translation matrix
		Mat xTranslationMatrix = new Mat(2, 3, CV_32F);
		FloatRawIndexer i = xTranslationMatrix.createIndexer();
		i.put(0, 0, 1);
		i.put(0, 1, 0);
		i.put(0, 2, xTrans);
		i.put(1, 0, 0);
		i.put(1, 1, 1);
		i.put(1, 2, 0); // y_trans

		// translate image
		Mat translatedImage = new Mat();
		warpAffine(rotatedImage, translatedImage, xTranslationMatrix, image.size());

		return translatedImage;
	}

	public List<Mat> extractFaces(RectVector faceRects, Mat image, Size targetSize, double proportionPadding) {

		// minimum size of face to grab is 2% of the source image height
		int absoluteFaceSize = Math.round(image.rows() * 0.02f);

		this.faceClassifier.detectMultiScale(image, faceRects, 1.04, 2, 0, new Size(absoluteFaceSize, absoluteFaceSize),
				new Size());

		// get image dimensions
		int imageHeight = image.rows();
		int imageWidth = image.cols();

		// extract each face
		ArrayList<Mat> faces = new ArrayList<>();
		for (int i = 0; i < faceRects.size(); i++) {
			opencv_core.Rect currentRect = faceRects.get(i);

			// set padding to be either the prescribed padding or the smallest amount to the
			// image bounds
			int padding = (int) (proportionPadding * currentRect.height());
			int maxPadding = Math.min(currentRect.x(), currentRect.y());
			maxPadding = Math.min(maxPadding, imageHeight - (currentRect.y() + currentRect.height()));
			maxPadding = Math.min(maxPadding, imageWidth - (currentRect.x() + currentRect.width()));
			maxPadding = Math.min(maxPadding, padding);

			// apply the padding
			currentRect.x(currentRect.x() - maxPadding);
			currentRect.y(currentRect.y() - maxPadding);
			currentRect.width(currentRect.width() + 2 * maxPadding);
			currentRect.height(currentRect.height() + 2 * maxPadding);

			// add to final list
			Mat face = new Mat(image, currentRect);
			// resize faces to standard size
			resize(face, face, targetSize);
			faces.add(face);
		}

		return faces;
	}
}
