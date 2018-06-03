package librec.fajie;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.math.BigDecimal;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;


import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;

/**
 * * -Xms1G multicenter inverse distance prob
 * 
 * @author Administrator
 *
 *
 */
public class MProUtil {
	// private static String path="uid.csv";
	// private static String path="uid_Phoenix_20_5.csv";
	// private static String path="uid_90.csv";
	// private static String path="uid_basedonall.csv";
	// private static String path="uifake";
	// private static String path="data//sigir2015//ratingssmall.dat";

	public static String slash = "-";
	public static Map<String, Double> uidmap;
	public static Map<String, Integer> uid_fakemap;
	public static Map<Integer, Integer> poi_statemap;// business and state
	public static BiMap<String, Double> uidmap_;
	private static String comma = ",";
	public static Map<Integer, String> nnmap;
	public static Map<String, String> mnnmap;
	public static Map<Integer, Double> reviewmap;
	public static Map<Integer, Integer> music_artist_map;
	public static Map<Integer, List<Integer>> user_tag_map;
	public static Map<Integer, List<Integer>> item_tag_map;
	public static Map<Integer, Integer> movie_director_map;
	public static Map<Integer, List<Integer>> user_trust_map;
	public static Map<Integer, Integer> libimseti_gender;

	public static Map<Integer, int[]> movielens_userfeature;
	public static Map<Integer, int[]> yahoo_musicfeature;
	public static Map<Integer, String> movielens_itemfeature;
	public static Map<Integer, Integer> reviewmap_discrete;
	private static String nn_path = "nn";
	private static String mnn_path = "o_mul_dis";
	private static String review_path = "review";
	private static int maxreview_nim = 1512;
	private static double distance_threshold = 20;

	public static void main(String[] args) throws SQLException, IOException {

		// getMap();
		// getNeiborMap_File();
		// getNeiborMap();
		// getReview_File();
		// getReviewCount();
		// getNeiborMap();
		// getMuti_NeiborMap();
	}


	public static void getReviewCount() throws SQLException, IOException {
		// TODO Auto-generated method stub

		reviewmap = new HashMap<Integer, Double>();
		BufferedReader br = new BufferedReader(new InputStreamReader(
				new FileInputStream(review_path)));
		for (String line = br.readLine(); line != null; line = br.readLine()) {
			String bid = line.split(",")[0];
			String recount = line.split(",")[1]; // nn6
			BigDecimal b = new BigDecimal(Double.parseDouble(recount)
					/ maxreview_nim);
			double num = b.setScale(3, BigDecimal.ROUND_HALF_UP).doubleValue();
			reviewmap.put(Integer.parseInt(bid), num);
		}
		br.close();
	}

	public static void getReview_discrete() throws SQLException, IOException {
		// TODO Auto-generated method stub

		reviewmap_discrete = new HashMap<Integer, Integer>();
		BufferedReader br = new BufferedReader(new InputStreamReader(
				new FileInputStream(review_path)));
		for (String line = br.readLine(); line != null; line = br.readLine()) {
			String bid = line.split(",")[0];
			String recount = line.split(",")[1];
			reviewmap_discrete.put(Integer.parseInt(bid),
					Integer.parseInt(recount));
		}
		br.close();
	}

	

	public static void getNeiborMap() throws SQLException, IOException {
		// TODO Auto-generated method stub
		nnmap = new HashMap<Integer, String>();
		BufferedReader br = new BufferedReader(new InputStreamReader(
				new FileInputStream(nn_path)));
		for (String line = br.readLine(); line != null; line = br.readLine()) {
			String bid = line.split(",")[0];
			String nn6 = line.split(",")[1]; // nn6
			nnmap.put(Integer.parseInt(bid), nn6);
		}
		br.close();

	}

	
	


	public static int Norm(double distance) {
		int d = 20;
		if (distance < 20) {
			d = (int) (distance / 1);
		} else {
			d = 20;
		}
		return d;
	}

	public static int Norm_review(int review_n) {
		return review_n;
	}


	
	public static void getMap_movie_directorFeature(String path_)
			throws IOException, SQLException {

		// TODO Auto-generated method stub
		BufferedReader br = new BufferedReader(new InputStreamReader(
				new FileInputStream(path_)));
		movie_director_map = new HashMap<Integer, Integer>();
		for (String line = br.readLine(); line != null; line = br.readLine()) {
			int mid = Integer.parseInt(line.split(",")[0]);// music
			int did = Integer.parseInt(line.split(",")[1]);// artist
			movie_director_map.put(mid, did);
		}
		br.close();
	}
	
	public static void getMap_music_artistFeature(String path_)
			throws IOException, SQLException {

		// TODO Auto-generated method stub
		BufferedReader br = new BufferedReader(new InputStreamReader(
				new FileInputStream(path_)));
		music_artist_map = new HashMap<Integer, Integer>();
		for (String line = br.readLine(); line != null; line = br.readLine()) {
			int mid = Integer.parseInt(line.split(",")[0]);// music
			int aid = Integer.parseInt(line.split(",")[1]);// artist
			music_artist_map.put(mid, aid);
		}
		br.close();
	}
	//For tag recommendation, testing the algorithm without item information, non-context
	public static void getMap_user_tag_Feature(String path_)
			throws IOException, SQLException {

		// TODO Auto-generated method stub
		BufferedReader br = new BufferedReader(new InputStreamReader(
				new FileInputStream(path_)));
		user_tag_map = new HashMap<Integer, List<Integer>>();
		item_tag_map= new HashMap<Integer, List<Integer>>();
		for (String line = br.readLine(); line != null; line = br.readLine()) {
			int uid = Integer
					.parseInt(line.split("[ \t,::]+")[0].split("-")[0]);// user
			int iid = Integer
					.parseInt(line.split("[ \t,::]+")[0].split("-")[1]);// item
			int tid = Integer.parseInt(line.split("[ \t,::]+")[1]);// tag
			if (user_tag_map.containsKey(uid))
				user_tag_map.get(uid).add(tid);
			else {
				List<Integer> tags = new ArrayList<Integer>();
				tags.add(tid);
				user_tag_map.put(uid, tags);
			}
			if (item_tag_map.containsKey(iid))
				item_tag_map.get(iid).add(tid);
			else {
				List<Integer> tags = new ArrayList<Integer>();
				tags.add(tid);
				item_tag_map.put(iid, tags);
			}
		}
		br.close();
	}

	public static void getMap_user_trustFeature(String path_)
			throws IOException, SQLException {

		// TODO Auto-generated method stub
		BufferedReader br = new BufferedReader(new InputStreamReader(
				new FileInputStream(path_)));
		user_trust_map = new HashMap<Integer, List<Integer>>();
		List<Integer> ulist;
		for (String line = br.readLine(); line != null; line = br.readLine()) {
			int uid = Integer.parseInt(line.split("[ \t,::]+")[0]);// user       movie
			int tid = Integer.parseInt(line.split("[ \t,::]+")[1]);// trust      tags or actors
			if (user_trust_map.get(uid) != null) {
				user_trust_map.get(uid).add(tid);
			} else {
				ulist = new ArrayList<Integer>();
				ulist.add(tid);
				user_trust_map.put(uid, ulist);
			}
		}
		br.close();
	}

	public static void getMap_libimseti_genderFeature(String path_)
			throws IOException, SQLException {

		// TODO Auto-generated method stub
		BufferedReader br = new BufferedReader(new InputStreamReader(
				new FileInputStream(path_)));
		libimseti_gender = new HashMap<Integer, Integer>();
		for (String line = br.readLine(); line != null; line = br.readLine()) {
			int iid = Integer.parseInt(line.split(",")[0]);// person
			int gid = 0;
			String gender = line.split(",")[1];// gender
			if (gender.equals("M")) {
				gid = 0;
			} else if (gender.equals("F")) {
				gid = 1;
			} else {
				gid = 2;
			}

			libimseti_gender.put(iid, gid);
		}
		br.close();
	}

	public static void getMap_movielens_1m_userfeature(String path_)
			throws IOException, SQLException {

		// TODO Auto-generated method stub
		BufferedReader br = new BufferedReader(new InputStreamReader(
				new FileInputStream(path_)));
		movielens_userfeature = new LinkedHashMap<Integer, int[]>();
		Map<String, String> a = new LinkedHashMap<String, String>();
		for (String line = br.readLine(); line != null; line = br.readLine()) {
			// 1::F::1::10::48067
			int[] context = new int[3];
			int uid = Integer.parseInt(line.split("::")[0]);// music
			context[0] = line.split("::")[1].equals("M") ? 0 : 1;//
			context[1] = getAgebyMapping(Integer.parseInt(line.split("::")[2]));
			context[2] = Integer.parseInt(line.split("::")[3]);
			movielens_userfeature.put(uid, context);
		}
		br.close();
	}

	public static void getMap_movielens_1m_itemfeature(String path_)
			throws IOException, SQLException {

		// TODO Auto-generated method stub
		BufferedReader br = new BufferedReader(new InputStreamReader(
				new FileInputStream(path_)));
		movielens_itemfeature = new LinkedHashMap<Integer, String>();
		for (String line = br.readLine(); line != null; line = br.readLine()) {
			// 1::F::1::10::48067
			int uid = Integer.parseInt(line.split("::::")[0]);
			String iids = line.split("::::")[1];
			movielens_itemfeature.put(uid, iids);
		}
		br.close();
	}

	public static void getMap_yahoo_musicfeature(String path_)
			throws IOException, SQLException {

		// TODO Auto-generated method stub
		BufferedReader br = new BufferedReader(new InputStreamReader(
				new FileInputStream(path_)));
		yahoo_musicfeature = new LinkedHashMap<Integer, int[]>();
		for (String line = br.readLine(); line != null; line = br.readLine()) {
			// 1::F::1::10::48067
			int[] context = new int[2];
			int mid = Integer.parseInt(line.split(",")[0]);// music
			context[0] = Integer.parseInt(line.split(",")[1]);
			context[1] = Integer.parseInt(line.split(",")[2]);
			yahoo_musicfeature.put(mid, context);
		}
		br.close();
	}

	private static int getAgebyMapping(int age) {
		// TODO Auto-generated method stub
		int age_ = 0;
		switch (age) {
		case 1:
			age_ = 0;
			break;
		case 18:
			age_ = 1;
			break;
		case 25:
			age_ = 2;
			break;
		case 35:
			age_ = 3;
			break;
		case 45:
			age_ = 4;
			break;
		case 50:
			age_ = 5;
			break;
		case 56:
			age_ = 6;
			break;
		}
		return age_;
	}

	


}
