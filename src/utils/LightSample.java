package utils;

public class LightSample{
	public int word_ids[];
	public int label;
	public int text_id;
	public String type;
	
	public LightSample(Sample p, String type) {
		this.word_ids = p.getWordIds();
		this.label = p.label;
		this.text_id = p.text_id;
		this.type = type;
	}
}
