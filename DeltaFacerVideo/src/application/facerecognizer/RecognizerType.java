package application.facerecognizer;

public enum RecognizerType {
	PCA("PCA"), //principle component analysis (eigen recognizer)
	LDA("LDA"), //linear discriminant analysis (fisher recognizer),
	BPH("BPH");  //Binary Pattern Histogram

	private String name;
	
	RecognizerType(String value){
		this.name = value;
	}

	/**
	 * @return the value
	 */
	public String getName() {
		return name;
	}
	
}
