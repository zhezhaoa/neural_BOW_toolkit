package model;

import java.io.File;
import java.io.FileWriter;
import java.util.*;
import java.util.Map.Entry;

import datasets.*;
import utils.*;

//our model
public class NBOW {
	public static int use_imdb = 0;
	public static int use_rt2k = 0;
	public static int use_rts = 1;
	public static int use_subj = 0;
	public static int use_athr = 0;
	public static int use_xgraph = 0;
	public static int use_bbcrypt = 0;
	public static int use_cr = 0;
	public static int use_mpqa = 0;
	
	public static Random random = new Random();
	public static float lr = 0.025f; // learning rate
	public static float original_lr = lr; // initialized learning rate
	public static int neg_size = 5; // negative sampling size
	public static int gram = 1;
	public static int iter_num = 20; // iteration number
	public static int batch_size = 100;
	public static int n = 100; // vector size for both words and documents
	public static int thread_num = 8;
	public static String data_file_path = null;
	public static int vocab_size; // vocabulary size
	public static int texts_num; // the number of texts 
	public static int cv_num = 10;
	public static int nb = 1;

	public static float WE[][]; // word embeddings
	public static float TE[][]; // text embeddings
	public static Map<String, ArrayList<Sample>> dataset = null;
	public static List<LightSample> dataset_light = new ArrayList<LightSample>();

	public static void initNet() {
		vocab_size = Sample.id2count.size();
		texts_num = dataset_light.size();
		System.out.println("Vocabulary size: " + vocab_size);
		System.out.println("The number of texts: " + texts_num);
		WE = new float[vocab_size][n];
		for (int i = 0; i < vocab_size; i++)
			for (int j = 0; j < n; j++)
				WE[i][j] = (random.nextFloat() - 0.5f) / n;

		TE = new float[texts_num][n];
		for (int i = 0; i < texts_num; i++)
			for (int j = 0; j < n; j++)
				TE[i][j] = (random.nextFloat() - 0.5f) / n;
	}
    

	
	public static class TrainThread extends Thread {
		public List<LightSample> ls;
		public TrainThread(List<LightSample> ls) {
			this.ls = ls;
		}
		public void run() {
			train();
		}
		public void backprop(float a[], float b[], float t, float ratio) {
			if (a == null || b == null)
				return;
			float y = 0;
			for (int i = 0; i < n; i++)
				y += a[i] * b[i];
			y = (float) (1.0 / (1 + Math.exp(-y)));
			for (int i = 0; i < n; i++) {
				float wv = a[i];
				if (nb == 0)
					ratio = 1;
				a[i] += -(y - t) * b[i] * lr * ratio;
				b[i] += -(y - t) * wv * lr * ratio;
			}
		}		
		public void train() {
			int randId = 0;
			for (LightSample l : ls) {
				int text_id = l.text_id;
				int ids[] = l.word_ids;
				for (int i=0; i<ids.length; i++) {
					backprop(TE[text_id], WE[ids[i]], 1, Sample.ratio.get(ids[i]));
					for (int index_neg = 0; index_neg < neg_size; index_neg++) {
						randId = Sample.getRandomWordId();
						backprop(TE[text_id], WE[randId], 0, Sample.ratio.get(randId));
					}
				}
			}
		}
	}
    
	public static double trainModel() {
		double accuracy = 0;
		TrainThread threads[] = new TrainThread[thread_num];
		for (int iter = 0; iter < iter_num; iter++) {
			long startTime = System.currentTimeMillis();
			System.out.print("Iter: "+iter);
			int batch_index = 0;
			Collections.shuffle(dataset_light);
			while (true) {
				boolean over = false;
				for (int i = 0; i < threads.length; i++) {
					if (threads[i] == null || !threads[i].isAlive()) {
						if (batch_index < dataset_light.size() / batch_size) {
							int start = batch_size * batch_index;
							int end = batch_size * batch_index + batch_size;
							if (dataset_light.size() < end)
								end = dataset_light.size();
							threads[i] = new TrainThread(dataset_light.subList(start, end));
							threads[i].start();
							batch_index++;
							lr = original_lr
									* (1 - (iter * dataset_light.size() / batch_size + batch_index) * 1.0f
											/ (iter_num * dataset_light.size() / batch_size));
						} else {
							over = true;
							break;
						}
					} else {
					}
				}
				if (over) {
					//break;
					over = false;
					while (true) {
				        if (over == true){
				    	    break;
				        } else {
				    	    over = true;
				    	    for (int i = 0; i < threads.length; i++) {
				    		    if (threads[i].isAlive())
				    			    over = false;
				    	    }
				        }
				    }
					if (over == true)
						break;
				}
			}
			System.out.println(" ||" + " training time:" + (System.currentTimeMillis() - startTime)/1000 + "s");
			if (iter <= iter_num - 1) {
			    try {
				    FileWriter fw_train = new FileWriter("./liblinear/" + "train.txt");
				    FileWriter fw_test = new FileWriter("./liblinear/" + "test.txt");
				    for (LightSample ls : dataset_light) {
					    if (ls.label == -1)
						    continue;
					    FileWriter fw = fw_test;
					    if (ls.type.equals("train"))
						    fw = fw_train;
					    fw.write(ls.label + "\t");
					    for (int i = 0; i < n; i++)
						    fw.write((i + 1) + ":" + TE[ls.text_id][i] + "\t");
					    fw.write("\r\n");
				    }
				    fw_train.close();
				    fw_test.close();
				    accuracy = Classifier.runAndPrint_liblinear(false);
				    System.out.println("accuacy:"+accuracy);
			    } catch (Exception e) {
				    e.printStackTrace();
			    }
			}
		}
		return accuracy;
		
	}
	
	public static void trainAndTest() {
		dataset_light.clear();
		for (Entry<String, ArrayList<Sample>> entry : dataset.entrySet()) {
			if (entry.getKey().equals("train") || entry.getKey().equals("test") || entry.getKey().equals("unlabeled")) {
				for (Sample sample : entry.getValue()) {
				    dataset_light.add(new LightSample(sample, entry.getKey()));
				}
			}
		}
		Sample.initUniTable();
		Sample.calRatio();  
		System.out.println("Parameters initialization");
		initNet();
		System.out.println("Accuracy: " + trainModel());
	}
	
	public static double crossValidation() {
		double accuracy = 0;
		for (Entry<String, ArrayList<Sample>> entry : dataset.entrySet()) {
		    if (entry.getKey().equals("train") || entry.getKey().equals("test") || entry.getKey().equals("unlabeled")) {
			    for (Sample sample : entry.getValue()) {
			        dataset_light.add(new LightSample(sample, entry.getKey()));
			    }
		    }
	    }
	    Sample.initUniTable();
	    Sample.calRatio();
	    System.out.println("Parameters initialization");
	    initNet();
	    accuracy = trainModel();
	    return accuracy;
	}

	public static void printHyperParameter(){
		System.out.println("Hyper-parameter setting");
		System.out.println("N-gram: from 1 to " + gram);
		System.out.println("Embedding size (dimension): " + n);
		System.out.println("Iteration number: " + iter_num);
		System.out.println("The number of negative samples: " + neg_size);
		System.out.println("The number of threads: " + thread_num);
		System.out.println();
	}
	public static void main(String args[]) {
		if (use_imdb == 1){
			gram = IMDB.gram;
			iter_num = IMDB.iter_num;
			nb = IMDB.nb;
			n = IMDB.n;
			dataset = IMDB.getDataset();
			printHyperParameter();
			trainAndTest();
		}
		if (use_rt2k == 1){
			gram = RT2k.gram;
			nb = RT2k.nb;
			iter_num = RT2k.iter_num;
			n = RT2k.n;
			printHyperParameter();
			double accuracy = 0;
			double cv_accuracy = 0;
			for (int cv = 0; cv < cv_num; cv++){
				dataset_light.clear();
				dataset = RT2k.getDataset(cv);
			    accuracy = crossValidation();
			    System.out.println("cv:"+cv+"    accuracy:"+accuracy);
			    cv_accuracy += accuracy;
			}
			System.out.println(cv_num+" fold cv accuracy:"+cv_accuracy/10);
			System.out.println("Training finished");
			
		}
		if (use_rts == 1){
			gram = RTs.gram;
			iter_num = RTs.iter_num;
			nb = RTs.nb;
			n = RTs.n;
			if (RTs.use_unlabelled == 1)
				iter_num = 15;
			printHyperParameter();
			double accuracy = 0;
			double cv_accuracy = 0;
			for (int cv = 0; cv < cv_num; cv++){
				dataset_light.clear();
				dataset = RTs.getDataset(cv);
			    accuracy = crossValidation();
			    System.out.println("cv:"+cv+"    accuracy:"+accuracy);
			    cv_accuracy += accuracy;
			}
			System.out.println(cv_num+" fold cv accuracy:"+cv_accuracy/10);
			System.out.println("Training finished");
		}
		if (use_subj == 1){
			data_file_path = subj.data_file_path;
			gram = subj.gram;
			iter_num = subj.iter_num;
			nb = subj.nb;
			n = subj.n;
			printHyperParameter();
			double accuracy = 0;
			double cv_accuracy = 0;
			for (int cv = 0; cv < cv_num; cv++){
				dataset_light.clear();
				dataset = subj.getDataset(data_file_path, cv);
			    accuracy = crossValidation();
			    System.out.println("cv:"+cv+"    accuracy:"+accuracy);
			    cv_accuracy += accuracy;
			}
			System.out.println(cv_num+" fold cv accuracy:"+cv_accuracy/10);
			System.out.println("Training finished");
		}
		if (use_athr == 1){
			data_file_path = AthR.data_file_path;
			gram = AthR.gram;
			iter_num = AthR.iter_num;
			nb = AthR.nb;
			n = AthR.n;
			dataset = AthR.getDataset(data_file_path);
			printHyperParameter();
			trainAndTest();
		}
		if (use_xgraph == 1){
			data_file_path = XGraph.data_file_path;
			gram = XGraph.gram;
			iter_num = XGraph.iter_num;
			nb = XGraph.nb;
			n = XGraph.n;
			dataset = XGraph.getDataset(data_file_path);
			printHyperParameter();
			trainAndTest();
		}
		if (use_bbcrypt == 1){
			data_file_path = BbCrypt.data_file_path;
			gram = BbCrypt.gram;
			iter_num = BbCrypt.iter_num;
			nb = BbCrypt.nb;
			n = BbCrypt.n;
			dataset = BbCrypt.getDataset(data_file_path);
			printHyperParameter();
			trainAndTest();
		}
		if (use_cr == 1){
			gram = CR.gram;
			nb = CR.nb;
			iter_num = CR.iter_num;
			n = CR.n;
			printHyperParameter();
			double accuracy = 0;
			double cv_accuracy = 0;
			for (int cv = 0; cv < cv_num; cv++){
				dataset_light.clear();
				dataset = CR.getDataset(cv);
			    accuracy = crossValidation();
			    System.out.println("cv:"+cv+"    accuracy:"+accuracy);
			    cv_accuracy += accuracy;
			}
			System.out.println(cv_num+" fold cv accuracy:"+cv_accuracy/10);
			System.out.println("Training finished");
		}
		if (use_mpqa == 1){
			gram = MPQA.gram;
			nb = MPQA.nb;
			iter_num = MPQA.iter_num;
			n = MPQA.n;
			printHyperParameter();
			double accuracy = 0;
			double cv_accuracy = 0;
			for (int cv = 0; cv < cv_num; cv++){
				dataset_light.clear();
				dataset = MPQA.getDataset(cv);
			    accuracy = crossValidation();
			    System.out.println("cv:"+cv+"    accuracy:"+accuracy);
			    cv_accuracy += accuracy;
			}
			System.out.println(cv_num+" fold cv accuracy:"+cv_accuracy/10);
			System.out.println("Training finished");
		}
	}
}
