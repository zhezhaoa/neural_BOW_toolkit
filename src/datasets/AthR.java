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

public class AthR {
	public static int ngram = 2;
	public static int nb = 1;
	public static int iter_num = 20;
	public static int n = 300;
	public static int neg_size = 30;
	public static int use_w2v = 1;
	public static String data_file_path = "./datasets/AthR/"+ ngram + "gram.txt";
	public static String w2v_file = "./datasets/AthR/w2v.txt";
	public static String train_test_split = "./datasets/AthR/train_test_split.txt";
	public static int [] train_test_split_index;
	public static Map<String, ArrayList<Sample>> getDataset() {
		Map<String, ArrayList<Sample>> dataset = new HashMap<String, ArrayList<Sample>>();
		dataset.put("train", new ArrayList<Sample>());
		dataset.put("test", new ArrayList<Sample>());
		dataset.put("unlabeled", new ArrayList<Sample>());
		try {
			System.out.println("Loading dataset...");
			
			
			File split = new File(train_test_split);
			BufferedReader reader = new BufferedReader(new FileReader(split));
			String line = null;
			train_test_split_index = new int [1427];
			int counter = 0;
			while ((line = reader.readLine()) != null) {
				train_test_split_index[counter] = Integer.parseInt(line);
			    counter++;
			}
			reader.close();
			
			
			File file = new File(data_file_path);
			reader = new BufferedReader(new FileReader(file));

			int text_id;
			int label; // 1:positive 0:negative -1:unlabeled 
			String type = null;
			while ((line = reader.readLine()) != null) {
				text_id = Integer.parseInt(line.split(" ")[0].substring(2));
				type = "train";
				if (train_test_split_index[text_id] == 0)
					type = "test";				
				label = -1;
				if (text_id < 799)
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
