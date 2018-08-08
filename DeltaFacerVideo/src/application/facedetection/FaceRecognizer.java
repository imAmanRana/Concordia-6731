package application.facedetection;

import static org.bytedeco.javacpp.opencv_core.FONT_HERSHEY_PLAIN;
import static org.bytedeco.javacpp.opencv_highgui.destroyAllWindows;
import static org.bytedeco.javacpp.opencv_highgui.imshow;
import static org.bytedeco.javacpp.opencv_highgui.waitKey;
import static org.bytedeco.javacpp.opencv_imgproc.COLOR_BGRA2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.equalizeHist;
import static org.bytedeco.javacpp.opencv_imgproc.putText;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Properties;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.RectVector;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber.Exception;
import org.bytedeco.javacv.OpenCVFrameConverter;

import application.facerecognizer.DeltaFacer;
import application.facerecognizer.FacePreprocessor;
import application.facerecognizer.RecognizerType;

public class FaceRecognizer {

	public static Properties properties;
	static CascadeClassifier faceClassifier;
	static CascadeClassifier eyeClassifier;
	static OpenCVFrameConverter.ToMat converterToMat = new OpenCVFrameConverter.ToMat();
	static FacePreprocessor preprocessor;

	public static DeltaFacer mainRecognizer;

	public static void main(String[] args) throws Exception {

		loadApplicationProperties();

		boolean load = Boolean.parseBoolean(properties.getProperty("LOAD_FROM_FILE"));
		RecognizerType recognizer = getRecognizerType();

		boolean toSave = setupRecognizer(recognizer, load);

		if (toSave) {
			mainRecognizer.save("resources/models/" + mainRecognizer.getRecognizerType().getName() + "_Trained");
			System.out.println("Saved the Trained Data");
		}

		FFmpegFrameGrabber grabber = initializeFrameGrabber();
		Frame videoFrame = null;
		Mat videoMat;
		double paddingAmount = Double.parseDouble(properties.getProperty("PADDING_AMOUNT"));
		String name = null;
		while (true) {
			videoFrame = grabber.grabImage();
			try (Mat videoMatGray = new Mat(); RectVector rectVector = new RectVector()) {
				if (videoFrame == null || videoFrame.image == null)
					continue;

				videoMat = converterToMat.convert(videoFrame);

				// Convert the current frame to grayscale:
				cvtColor(videoMat, videoMatGray, COLOR_BGRA2GRAY);
				equalizeHist(videoMatGray, videoMatGray);

				Size targetSize = new Size(257, 300);
				List<Mat> imageList = preprocessor.facepreprocessing(videoMatGray, rectVector, targetSize,
						paddingAmount);

				for (int i = 0; i < imageList.size(); i++) {
					name = mainRecognizer.predictLabel(imageList.get(i));

					rectangle(videoMat, rectVector.get(i), new Scalar(0, 255, 0, 1));
					int pos_x = Math.max(rectVector.get(i).tl().x() - 10, 0);
					int pos_y = Math.max(rectVector.get(i).tl().y() - 10, 0);

					// And now put it into the image:
					putText(videoMat, "Person: "+name, new Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0,
							new Scalar(0, 255, 0, 2.0));
				}

			}
			imshow("face_recognizer", videoMat);

			char key = (char) waitKey(20);
			// Exit this loop on escape:
			if (key == 27) {
				destroyAllWindows();
				break;
			}
		}
		grabber.flush();
		grabber.release();
		grabber.close();
	}

	private static RecognizerType getRecognizerType() {
		String tr = properties.getProperty("TRAINING_RECOGNIZER");
		for (RecognizerType name : RecognizerType.values()) {
			if (name.getName().equals(tr)) {
				return name;
			}
		}
		return null;
	}

	private static FFmpegFrameGrabber initializeFrameGrabber() {
		FFmpegFrameGrabber grabber = null;
		try {
			grabber = new FFmpegFrameGrabber(new File(properties.getProperty("VIDEO_FILE")));
			grabber.setImageHeight(700);
			grabber.setImageWidth(900);
			grabber.start();
		} catch (Exception e) {
			System.err.println("Failed start the grabber.");
		}
		return grabber;
	}

	private static boolean setupRecognizer(RecognizerType type, boolean loadFromFile) {
		mainRecognizer = new DeltaFacer(type);
		String trainingData = properties.getProperty("TRAINING_DATA");
		String fileName = "resources/models/" + type.getName() + "_Trained";

		File f = new File(fileName);

		if (loadFromFile && f.exists()) {
			System.out.println("Loading " + mainRecognizer.getRecognizerType().getName() + " recognizer with file "
					+ loadFromFile);
			mainRecognizer.load(fileName, trainingData);
		} else {
			if(!f.exists()) {
				System.out.println("Traning data not found, building the one");
			}
			System.out.println("Training " + mainRecognizer.getRecognizerType().getName()
					+ " recognizer with data at \"" + trainingData + "\"");
			mainRecognizer.train(trainingData);
		}

		// print the confusion matrix
		mainRecognizer.printConfusionMat();
		return !(loadFromFile && f.exists());
	}

	public static void loadApplicationProperties() {
		InputStream inStream = FaceRecognizer.class.getResourceAsStream("/application.properties");
		properties = new Properties();
		try {
			properties.load(inStream);
			faceClassifier = new CascadeClassifier(properties.getProperty("FACE_CLASSIFIER"));
			eyeClassifier = new CascadeClassifier(properties.getProperty("EYE_CLASSIFIER"));
			preprocessor = new FacePreprocessor(faceClassifier, eyeClassifier);
		} catch (IOException e) {
			System.out.println("Error while loading the system properties");
			e.printStackTrace();
		}
	}

}
