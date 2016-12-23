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

public class IMDB {
	public static int ngram = 1;
	public static int nb = 0;
	public static int iter_num = 20;
	public static int n = 300;
	public static int neg_size = 5;
	public static int use_w2v = 0;
	public static int use_unlabelled = 0;
	public static String data_file_path = "./datasets/IMDB/alldata-id_p" + ngram + "gram.txt";
	public static String w2v_file = "./datasets/IMDB/w2v.txt";
	public static Map<String, ArrayList<Sample>> getDataset() {
		Map<String, ArrayList<Sample>> dataset = new HashMap<String, ArrayList<Sample>>();
		dataset.put("train", new ArrayList<Sample>());
		dataset.put("test", new ArrayList<Sample>());
		dataset.put("unlabeled", new ArrayList<Sample>());
		try {
			System.out.println("Loading dataset...");
			File file = new File(data_file_path);
			BufferedReader reader = new BufferedReader(new FileReader(file));
			String line = null;
			int text_id;
			int label; // 1:positive 0:negative -1:unlabeled 
			String type = null;
			while ((line = reader.readLine()) != null) {
				text_id = Integer.parseInt(line.split(" ")[0].substring(2));
				type = "train";
				if (25000 <= text_id && text_id < 50000)
					type = "test";
				if (text_id >= 50000)
					if (use_unlabelled == 1)
						type = "unlabeled";
					else
					    continue;			
				label = -1;
				if (text_id < 12500)
					label = 1;
				if (12500 <= text_id && text_id < 25000)
					label = 0;
				if (25000 <= text_id && text_id < 37500)
					label = 1;
				if (37500 <= text_id && text_id < 50000)
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
