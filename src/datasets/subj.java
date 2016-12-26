package datasets;

import java.io.*;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Map.Entry;

import utils.Sample;

public class subj {
	public static int ngram = 2;
	public static int nb = 1;
	public static int iter_num = 30;
	public static int n = 300;
	public static int neg_size = 15;
	public static String data_file_path = "./datasets/subj/" + ngram + "gram.txt";
	public static int use_w2v = 1;
	public static String w2v_file = "./datasets/subj/w2v.txt";
	public static String train_test_split = null;
	public static int [] train_test_split_index;
	public static Map<String, ArrayList<Sample>> getDataset(String filePath, int cv) {
		Map<String, ArrayList<Sample>> dataset = new HashMap<String, ArrayList<Sample>>();
		dataset.put("train", new ArrayList<Sample>());
		dataset.put("test", new ArrayList<Sample>());
		dataset.put("unlabeled", new ArrayList<Sample>());
		Sample.clear();
		try {
			System.out.println("Loading dataset...");
			
			train_test_split = "./datasets/subj/cv_" + cv + ".txt";
			File split = new File(train_test_split);
			BufferedReader reader = new BufferedReader(new FileReader(split));
			String line = null;
			train_test_split_index = new int [10000];
			int counter = 0;
			while ((line = reader.readLine()) != null) {
				train_test_split_index[counter] = Integer.parseInt(line);
			    counter++;
			}
			reader.close();
			
			File file = new File(filePath);
			reader = new BufferedReader(new FileReader(file));
			line = null;
			int text_id;
			int label; // 1:positive 0:negative -1:unlabeled 
			String type = null;
			while ((line = reader.readLine()) != null) {
				text_id = Integer.parseInt(line.split(" ")[0].substring(2));
				/*
				type = "train";
				if ((cv*500 <= text_id && text_id < (cv+1)*500) || ((cv*500+5000) <= text_id && text_id < ((cv+1)*500+5000)))
					type = "test";
				*/
				type = "train";
				if (train_test_split_index[text_id] == 0)
					type = "test";
				label = 0;
				if (text_id < 5000)
					label = 1;
				Sample p = new Sample(line.substring(line.split(" ")[0].length()).trim(), label, text_id, type);
				dataset.get(type).add(p);
			}
			reader.close();
			System.out.println("train"+dataset.get("train").size());
			System.out.println("test"+dataset.get("test").size());
			System.out.println("loading finished");
		} catch (Exception e) {
			e.printStackTrace();
		}
		return dataset;
	}
}
