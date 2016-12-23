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

public class RTs {
	public static int ngram = 1;
	public static int nb = 0;
	public static int iter_num = 150;
	public static int n = 300;
	public static int neg_size = 5;
	public static int use_w2v = 0;
	public static int use_unlabelled = 0;
	public static String data_file_path = "./datasets/RTs/" + ngram + "gram.txt";
	public static String w2v_file = "./datasets/RTs/w2v.txt";
	public static String train_test_split = null;
	public static int [] train_test_split_index;
	public static Map<String, ArrayList<Sample>> getDataset(int cv) {
		Map<String, ArrayList<Sample>> dataset = new HashMap<String, ArrayList<Sample>>();
		dataset.put("train", new ArrayList<Sample>());
		dataset.put("test", new ArrayList<Sample>());
		dataset.put("unlabeled", new ArrayList<Sample>());
		Sample.clear();
		try {
			System.out.println("Loading dataset...");
			
			train_test_split = "./datasets/RTs/cv_" + cv + ".txt";
			File split = new File(train_test_split);
			BufferedReader reader = new BufferedReader(new FileReader(split));
			String line = null;
			train_test_split_index = new int [10662];
			int counter = 0;
			while ((line = reader.readLine()) != null) {
				train_test_split_index[counter] = Integer.parseInt(line);
			    counter++;
			}
			reader.close();
			
			File file = new File(data_file_path);
			reader = new BufferedReader(new FileReader(file));
			line = null;
			int text_id;
			int label; // 1:positive 0:negative -1:unlabeled 
			String type = null;
			while ((line = reader.readLine()) != null) {
				text_id = Integer.parseInt(line.split(" ")[0].substring(2));
				
				if (text_id < 10662){
					type = "train";
					if (train_test_split_index[text_id] == 0)
						type = "test";
				}
				/*
				type = "train";
				if ( (cv*533<=text_id&&text_id<(cv+1)*533) || ((cv*533+5331)<=text_id && text_id<((cv+1)*533+5331)) )
					type = "test";
				*/
				
				if (use_unlabelled == 0 && text_id >=10662)
					continue;
				if (use_unlabelled == 1 && text_id >=10662)
					type = "unlabeled";
				if (text_id < 5331)
					label = 1;
				else if (text_id < 10662)
					label = 0;
				else
					label = -1;
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
