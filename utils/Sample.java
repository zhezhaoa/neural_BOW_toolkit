package utils;

import java.io.StringReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class Sample {

	public static Random random = new Random();
	public static double powFactor = 0.75;

	public static Map<String, Integer> word2id = new HashMap<String, Integer>(); 
	public static ArrayList<Integer> id2count = new ArrayList<Integer>();
	
	public static ArrayList<Integer> id2count_pos = new ArrayList<Integer>();// word occurrences in positive texts
	public static ArrayList<Integer> id2count_neg = new ArrayList<Integer>();// word occurrences in negative texts
	public static ArrayList<Float> ratio = new ArrayList<Float>();
	public static int[] uni_table = null;
	
	public int word_ids[]; // word ids in the text
	public String tokens[]; // tokens in the text
	public int length; // text length
	public int label;  // text label
	public int text_id;
	public String type; // train/test or validation
	
	public Sample(String text, int label, int text_id, String type) {
		this.tokens = text.split(" ");
		this.length = tokens.length;
		this.type = type;
		this.label = label;
		this.text_id = text_id;
		setWordIds(this.tokens);	
	}	
	
	public int[] getWordIds() {
		return word_ids;
	}

	public void setWordIds(String tokens[]) {
		int id;
		word_ids = new int[tokens.length];
		for (int i = 0; i < word_ids.length; i++) {
			if (!word2id.containsKey(tokens[i])) {
				word2id.put(tokens[i], id2count.size());
				id2count.add(0);
				id2count_pos.add(0);
				id2count_neg.add(0);
			}
			id = word2id.get(tokens[i]);
			id2count.set(id, id2count.get(id) + 1);
            if (type.equals("train")){
			    if (label == 1){
				    id2count_pos.set(id, id2count_pos.get(id) + 1);
			    }else if (label == 0){
				    id2count_neg.set(id, id2count_neg.get(id) + 1);
			    }
			}
			word_ids[i] = id;
		}
	}
	
	static public void calRatio(){
		ratio.clear();
		for (int i=0; i<id2count.size();i++){
			ratio.add((float) 1.0);
		}
		for (int i=0; i<id2count.size(); i++){
			if (id2count_pos.get(i)>id2count_neg.get(i)){
				ratio.set(i, (float) Math.sqrt(((float)id2count_pos.get(i)+1)/(id2count_neg.get(i)+1)));
			}else{
				ratio.set(i, (float) Math.sqrt(((float)id2count_neg.get(i)+1)/(id2count_pos.get(i)+1)));
			}
		}
	}
	

	public static int getRandomWordId() {
		int i = random.nextInt(Sample.uni_table[Sample.uni_table.length - 1]) + 1;
		int l = 0, r = uni_table.length - 1;
		while (l != r) {
			int m = (l + r ) / 2;
			if (i <= uni_table[m])
				r = m;
			else
				l = m + 1;
		}
		return l;
	}

	public static void initUniTable() {
		uni_table = new int[id2count.size()];
		uni_table[0] = (int) Math.round(Math.pow(id2count.get(0), powFactor));
		for (int t = 1; t < id2count.size(); t++)
			uni_table[t] = (int) Math.round(Math.pow(id2count.get(t), powFactor) + uni_table[t - 1]);
	}
	
	public static void clear() {
		word2id.clear();
		id2count.clear();
		id2count_pos.clear();
		id2count_neg.clear();
		uni_table = null;
	}
}
