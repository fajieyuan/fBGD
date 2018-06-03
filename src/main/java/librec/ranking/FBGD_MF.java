// Copyright (C) 2014 Guibing Guo
//
// This file is part of LibRec.
//
// LibRec is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// LibRec is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with LibRec. If not, see <http://www.gnu.org/licenses/>.
//

package librec.ranking;

import java.util.ArrayList;
import java.util.List;


import happy.coding.io.Strings;

import librec.data.DenseMatrix;

import librec.data.SparseMatrix;
import librec.data.SparseVector;

import librec.intf.IterativeRecommender;


/**
 * 
 * fBGD: A Unified Batch Gradient Approach for Positive Unlabeled Learning 
 * For recommenation and image classification
 * For word embedding task, the positive weight is also very important 
 * For different datasets, the weighting w0 (e.g., stepping by 2^n) and alpha (0-1.0) should be tuned carefully, first tune w0 then alpha following the paper. 
 * The default learning rate=0.05 and iterations=300
 * @author fajie yuan
 * 
 */
public class FBGD_MF extends IterativeRecommender {


	private DenseMatrix P_grad;
	private DenseMatrix Q_grad;

	private double alpha_p = 1, alpha_n = 0;
	private double[] Wi;
	private double w0 = 1000;// c_0 in He paper
	private double alpha = 0.4;
	private double epsilon=0.0000001;
//	private SSparseMatrix Prediction;

	public FBGD_MF(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);
		isRankingPred = true;
		w0 = algoOptions.getFloat("-w0");
		alpha = algoOptions.getFloat("-alpha");
	}

	@Override
	protected void initModel() throws Exception {
		super.initModel();// Gaussian distribution (0, 0.1)
		userCache = trainMatrix.rowCache(cacheSpec);
		itemCache = trainMatrix.columnCache(cacheSpec);
		
		// Set the Wi as a decay function w0 * pi ^ alpha
		double sum = 0, Z = 0;
		double[] p = new double[numItems];
		for (int i = 0; i < numItems; i++) {
			p[i] = trainMatrix.column(i).size();
			sum += p[i];
		}
		// convert p[i] to probability
		for (int i = 0; i < numItems; i++) {
			p[i] /= sum;
			p[i] = Math.pow(p[i], alpha);
			Z += p[i];
		}
		// assign weight
		Wi = new double[numItems];
		for (int i = 0; i < numItems; i++)
			Wi[i] = w0 * p[i] / Z;
	}

	@Override
	protected void buildModel() throws Exception {
		long startTime = 0;
		P_grad = new DenseMatrix(P);
		P_grad.init(0);
		Q_grad = new DenseMatrix(Q);
		Q_grad.init(0);
		for (int iter = 1; iter <= numIters; iter++) {			
			if (iter == 1) {
				startTime = System.currentTimeMillis();
			}
			loss = 0;
			

			double[][] S_qff_arr = new double[numFactors][numFactors];

			for (int f = 0; f < numFactors; f++) { 
				for (int f_ = 0; f_ < numFactors; f_++) {
					double S_qff_ = 0;
					for (int i = 0; i < numItems; i++) {
						alpha_n = Wi[i];
						S_qff_ += alpha_n * Q.get(i, f) * Q.get(i, f_);
					}
					S_qff_arr[f][f_] = S_qff_;
				}
			}

			// here the context is the user, i.e. u=c
			for (int u = 0; u < numUsers; u++) {
				double[] rateItem=new double[numItems];
				
				List<Integer> itemList = userCache.get(u).getIndexList();
				if (itemList.size() == 0)
					break; // user has no ratings
				for (int i : itemList) {
					rateItem[i]=predict(u, i);
				}
				for (int f = 0; f < numFactors; f++) { 
					SparseVector pu = userCache.get(u);
					int[] is = pu.getIndex();// positive item array

					double JP_Derivative = 0;
					double JA_Derivative = 0;
					double vals = 0;

					for (int pos = 0; pos < is.length; pos++) {

						double x_upos =rateItem[is[pos]];
						
						alpha_n = Wi[is[pos]];
						JP_Derivative += 2 * (alpha_p - alpha_n) * (x_upos - alpha_p / (alpha_p - alpha_n))
								* Q.get(is[pos], f);
						vals = (alpha_p - alpha_n) * (x_upos - alpha_p / (alpha_p - alpha_n))
								* (x_upos - alpha_p / (alpha_p - alpha_n));
						loss += vals;
					}
					JP_Derivative += 2 * regU * P.get(u, f);

					for (int f_ = 0; f_ < numFactors; f_++) {
						JA_Derivative += 2 * S_qff_arr[f][f_] * P.get(u, f_);
					}

					double grad = JP_Derivative + alpha_n * JA_Derivative;
					
					double newgrad=P_grad.get(u, f);
					newgrad+=grad*grad;
					P_grad.set(u, f, newgrad);
					newgrad=(1.0 / (Math.sqrt(newgrad)+epsilon)) * grad;
					

					P.add(u, f, -lRate * newgrad);//
				}
			}
			// Loss
			for (int f = 0; f < numFactors; f++) {
				for (int f_ = 0; f_ < numFactors; f_++) {
					double vals = 0;
					for (int u = 0; u < numUsers; u++) {
						vals += P.get(u, f) * P.get(u, f_);// eq(12)
					}
					loss += vals * S_qff_arr[f][f_];
				}
			}

			// Repeat y
			double[][] S_pff_arr = new double[numFactors][numFactors];
			for (int f = 0; f < numFactors; f++) { 
				for (int f_ = 0; f_ < numFactors; f_++) {
					double S_pff_ = 0;
					for (int u = 0; u < numUsers; u++) {
						S_pff_ += P.get(u, f) * P.get(u, f_);
					}
					S_pff_arr[f][f_] = S_pff_;
				}
			}
			
			for (int i = 0; i < numItems; i++) {
				List<Integer> userList = itemCache.get(i).getIndexList();
				double[] rateUser=new double[numUsers];
				if (userList.size() == 0)
					break; 
				for (int u : userList) {

					rateUser[u]=predict(u, i);
				}
				for (int f = 0; f < numFactors; f++) {

					SparseVector qi = itemCache.get(i);
					int[] us = qi.getIndex();// users who have rated item i
					alpha_n = Wi[i];
					double JP_Derivative = 0;
					double JA_Derivative = 0;
					double vals = 0;
					for (int u = 0; u < us.length; u++) {
					
						double xui =rateUser[us[u]];
						JP_Derivative += 2 * (alpha_p - alpha_n) * (xui - alpha_p / (alpha_p - alpha_n))
								* P.get(us[u], f);
						// caculate loss
						vals = (alpha_p - alpha_n) * (xui - alpha_p / (alpha_p - alpha_n))
								* (xui - alpha_p / (alpha_p - alpha_n));
						loss += vals;
					}
					JP_Derivative += 2 * regI * Q.get(i, f);
		
					for (int f_ = 0; f_ < numFactors; f_++) {
						JA_Derivative += 2 * S_pff_arr[f][f_] * Q.get(i, f_);
					}
					double grad = JP_Derivative + alpha_n * JA_Derivative;
					
					double newgrad=Q_grad.get(i, f);
					newgrad+=grad*grad;
					Q_grad.set(i, f, newgrad);
					newgrad=(1.0 / (Math.sqrt(newgrad)+epsilon)) * grad;

					Q.add(i, f, -lRate * newgrad);
				}
			}
			// calculate loss
			for (int f = 0; f < numFactors; f++) {
				for (int f_ = 0; f_ < numFactors; f_++) {
					double vals = 0;
					for (int i = 0; i < numItems; i++) {
						alpha_n = Wi[i];
						vals += Q.get(i, f) * Q.get(i, f_);
					}
					loss += vals * S_pff_arr[f][f_];
				}
			}
			if (isConverged(iter))
				break;
			if (iter == 1) {
				long endTime = System.currentTimeMillis();
				System.out.println("Evgm_mf-time:" + (endTime - startTime) + "milliseconds");
			}
		}
	}

	@Override
	public String toString() {
		return Strings.toString(
				new Object[] { binThold, w0, alpha, numFactors, initLRate, maxLRate, regU, regI, numIters, initStd },
				",");
	}

}
