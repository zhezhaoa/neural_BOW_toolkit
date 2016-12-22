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

public class MPQA {
	public static int gram = 2;
	public static int nb = 1;
	public static int iter_num = 10;
	public static int n = 100;
	public static int use_unlabelled = 1;
	public static String data_file_path = "./datasets/MPQA/" + gram + "gram.txt";
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
			
			train_test_split = "./datasets/MPQA/cv_" + cv + ".txt";
			File split = new File(train_test_split);
			BufferedReader reader = new BufferedReader(new FileReader(split));
			String line = null;
			train_test_split_index = new int [10606];
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
				
				type = "train";
				if (train_test_split_index[text_id] == 0)
					type = "test";
				
				if (text_id < 3312)
					label = 1;
				else
					label = 0;

				Sample p = new Sample(line.substring(line.split(" ")[0].length()).trim(), label, text_id, type);
				dataset.get(type).add(p);
			}
			reader.close();
			System.out.println("loading finished");
		} catch (Exception e) {
			e.printStackTrace();
		}
		return dataset;
	}
}
