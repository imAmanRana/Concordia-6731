package application.facerecognizer;
import org.bytedeco.javacpp.*;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;
import static org.bytedeco.javacpp.opencv_face.*;


public class DeltaFacer {
	private List<String> classLabels = new ArrayList<String>();
	// Training set
	private List<Mat> trainingMats = new ArrayList<Mat>();;
	private List<Integer> trainingLabels = new ArrayList<Integer>();
	// Test set
	private List<Mat> testMats = new ArrayList<Mat>();
	private List<Integer> testLabels = new ArrayList<Integer>();
	
	private RecognizerType _recognizerType;
	public RecognizerType getRecognizerType() { return _recognizerType; }
	
	private String _loadFile = "";
	//public void setLoadFile(String loadFile) { _loadFile = loadFile; }
	public String getLoadFile(String loadFile) { return _loadFile; }
	
	private boolean _isTrained = false;
	public boolean IsTrained() { return _isTrained; }
	
	private opencv_face.FaceRecognizer _model = null;

	public DeltaFacer(RecognizerType recognizerType) {
		// Setup new face recognizer
		_recognizerType = recognizerType;
		switch (_recognizerType)
		{
		case PCA:
			_model = createEigenFaceRecognizer();
			break;
		case LDA:
			_model = createFisherFaceRecognizer();
			break;
		case BPH:
			_model = createLBPHFaceRecognizer();
			break;
		default:
			System.err.println("Unrecognized Recognizer Type");
		}
	}
	
	// TODO: doc arguments
	public DeltaFacer(RecognizerType recognizerType, int arg0, double arg1) {
		// Setup new face recognizer with specified input arguments
		_recognizerType = recognizerType;
		switch (_recognizerType)
		{
		case PCA:
			_model = createEigenFaceRecognizer(arg0, arg1);
			break;
		case LDA:
			_model = createFisherFaceRecognizer(arg0, arg1);
			break;
		case BPH:
			//_model = createLBPHFaceRecognizer(0, 0, 0, 0, 0);
			//break;
		default:
			System.err.println("Unrecognized Recognizer Type");
		}
	}
	
    private int[] toIntArray(List<Integer> list){
        int[] ret = new int[list.size()];
        for(int i = 0;i < ret.length;i++)
            ret[i] = list.get(i);
        return ret;
    }

    public void save(String saveFile) {
    	if (!_isTrained) {
    		System.err.println("Recognizer untrained; nothing to save."); 
    		return;
    	}
    	
    	_model.save(saveFile);
    }
    
    public void load(String loadFile, String trainingData) {
    	_model.load(loadFile);
    	
    	//Todo: hack, use training data to load in class labels
    	//model should be able to provide us this info?
        try {
            readFaces(trainingData);
        } catch (Exception e) {
            System.out.println("Failed to load images: " + e.getMessage());
            System.exit(1);
        }
        
    	_isTrained = true;
    }
    
    private void readFaces(String faceDir) throws IOException {
        if (!Files.isDirectory(Paths.get(faceDir))) {
            throw new FileNotFoundException(faceDir + "not found.");
        }
     
        // In our Face folder, each folder represents a class
        File[] faceClasses = new File(faceDir).listFiles(File::isDirectory);

        Integer classIndex = 0;
        for (File faceClass : faceClasses) {
        	// OpenCV uses integers as labels, so we use the integer as the index in this list
        	// to get the actual name of the class.
        	classLabels.add(classIndex, faceClass.getName());
        	
        	File[] faceImgs = faceClass.listFiles(File::isFile);
        	
        	Integer numTrainingFaces = faceClass.listFiles(File::isFile).length;
        	
        	for (int i = 0; i < numTrainingFaces; ++i) {
            	// Only half the set of images in each class will be used to train -- the rest to test.
        		if (i < numTrainingFaces / 2) {
        			trainingMats.add(imread(faceImgs[i].getAbsolutePath(), 0));
        			trainingLabels.add(classIndex);
        		}
        		else {
        			testMats.add(imread(faceImgs[i].getAbsolutePath(), 0));
        			testLabels.add(classIndex);
        		}
        	}
        	
        	classIndex++;
        }
    }

    public String predictLabel(Mat img) {
    	 int predictedLabel = _model.predict_label(img);
    	 return classLabels.get(predictedLabel);
    }
    
    public String predictLabelAndConfidence(Mat img) {
    	int[] values = {0};
    	double[] dbls = {0.0};
    	IntPointer predictedLabel = new IntPointer(values);
        DoublePointer confidence = new DoublePointer(dbls);
        _model.predict(img, predictedLabel, confidence);
        return classLabels.get(predictedLabel.get()) + " (" + confidence.get() + ")";
    }
    
    public void printConfusionMat() {
    	if (_model == null)
    		System.out.println("Please train the classifier first.");
    	
        int testCount = 0;
        // Initialize confusion matrix
        int[][] confusion_matrix = new int[classLabels.size()][];
        for (int i = 0; i < classLabels.size(); ++i) {
        	confusion_matrix[i] = new int[classLabels.size()];
        	
        	for (int j = 0; j < classLabels.size(); ++j)
        		confusion_matrix[i][j] = 0;
        }
        
        for (int i = 0; i < testMats.size(); ++i) {
        	 int predictedLabel = _model.predict_label(testMats.get(i));
        	 System.out.println(
        			 String.format("Predicted class = %s / Actual class = %s.", 
        					 classLabels.get(predictedLabel), 
        					 classLabels.get(testLabels.get(i))
        			 )
        	);
        	confusion_matrix[testLabels.get(i)][predictedLabel]++;
        }
        
        // Output confusion matrix
        System.out.println("\nClasses:");
        for (int i = 0; i < classLabels.size(); ++i)
        	System.out.println(String.format("%d = %s", i, classLabels.get(i)));
        for (int i = 0; i < classLabels.size(); ++i)
        	System.out.format("%5d", i);
        System.out.println();
        for (int i = 0; i < classLabels.size(); ++i) {
        	System.out.format("%d", i);
        	for (int j = 0; j < classLabels.size(); ++j) {
        		System.out.format("%5d", confusion_matrix[i][j]);
        	}
        	System.out.println();
        }
    }
    
    public void train(String faceFolder) {
        // <ARGUMENT: path to faces folder>
        // the Faces folder should have a folder for each class (i.e. CELEBRITY NAME)
        // inside each of these folders, there should be cropped images of the celebrity belonging to that class
        // half of the faces in the folder will be used for training, the other half for testing
        // NOTE: all images must be of same dimensions
        //String faceFolder = args[0];
    	
    	//String faceFolder = "C:/Users/Aman Rana/Downloads/Faces";//args[0];
        
        try {
            readFaces(faceFolder);
        } catch (Exception e) {
            System.out.println("Failed to load images: " + e.getMessage());
            System.exit(1);
        }
        
        // Quit if there are not enough images for this demo.
        if(trainingMats.size() <= 1) {
            throw new RuntimeException("This demo needs at least 2 images to work. Please add more images to your data set!");
        }

        Mat[] matArray = new Mat[trainingMats.size()];
        matArray = trainingMats.toArray(matArray);
        int[] array = toIntArray(trainingLabels);
        Mat matlabels = new Mat(new IntPointer(array));
        MatVector matVector = new MatVector(matArray);
        _model.train(matVector, matlabels);
        
        _isTrained = true;
    }
}

