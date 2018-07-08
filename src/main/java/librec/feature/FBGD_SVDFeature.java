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

package librec.feature;


import data.gla.uni.data.structure.complementary.SSparseMatrix;
import data.gla.uni.data.structure.complementary.SSparseVector;
import librec.data.DenseMatrix;
import librec.data.SparseMatrix;
import librec.data.SparseVector;
import happy.coding.io.Strings;

/**
 * The code can be used for context-aware recommendation, image classification with sparse features, word embedding with prior knowledge 
 * fBGD: A Unified Batch Gradient Approach for Positive Unlabeled Learning 
 * For different datasets, the weighting w0 (e.g., stepping by 2^n) and alpha (0-1.0) should be tuned Carefully, first tune w0 then alpha following the paper. 
 * The default learning rate=0.05 and iterations=300
 * We use a simplified weight in the code, which shows same behavior
 * For better understanding the code, we have omitted the biased terms, which have very minor influence in practice.
 * The notations are not exactly the same with the paper. E.g., C in recommender system represents context or user, I represents item
 * @author fajie yuan
 * 
 */
public class FBGD_SVDFeature extends SFMachine_Base {

	private double alpha_p = 1, alpha_n = 0;
	private int V_numRows;
	private int numContexts;
	private DenseMatrix V_grad;

	private double[] Wi;
	private double w0 = 1000;// c_0 in He paper
	private double alpha = 0;
	private double epsilon=0.0000001;
	
	public FBGD_SVDFeature(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);
		isRankingPred = true;
		w0 = algoOptions.getFloat("-w0");
		alpha = algoOptions.getFloat("-alpha");

	}

	@Override
	protected void initModel() throws Exception {
		super.initModel();// Gaussian distribution (0, 0.1)
		V_numRows = V.numRows();
		userCache = trainMatrix.rowCache(cacheSpec);
		itemCache = trainMatrix.columnCache(cacheSpec);
		numContexts = numUsers;
	
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
		SSparseMatrix Prediction = new SSparseMatrix(numUsers, numItems);
		V_grad = new DenseMatrix(V);
		V_grad.init(0);
		for (int iter = 1; iter <= numIters; iter++) {
			if (iter == 1) {
				startTime = System.currentTimeMillis();
			}
			
			loss = 0;
		
			DenseMatrix pxf = new DenseMatrix(numContexts, numFactors);// it
																		// should
																		// be
																		// numofContext,
																		// |C|=|I|
			DenseMatrix qyd = new DenseMatrix(numItems, numFactors);
			// calculate pxf
			for (int f = 0; f < numFactors; f++) {
				for (int c = 0; c < numContexts; c++) {//
					// SparseVector sparse_C=x_cList.get(c);
					SSparseVector sparse_c = ZX.getRow(c);
					double sum_PC = 0;
					for (int l = 0; l < sparse_c.nonZeroCount(); l++) {
						int idx = sparse_c.indexList().get(l);
						sum_PC += V.get(idx, f) * sparse_c.getValue(idx);
					}
					pxf.set(c, f, sum_PC);
				}
			}
			for (int f = 0; f < numFactors; f++) {
				for (int i = 0; i < numItems; i++) {
					// SparseVector sparse_I=x_iList.get(i);
					SSparseVector sparse_i = ZY.getRow(i);
					double sum_PI = 0;
					for (int l = 0; l < sparse_i.nonZeroCount(); l++) {
						int idx = sparse_i.indexList().get(l);
						sum_PI += V.get(idx, f) * sparse_i.getValue(idx);
					}
					qyd.set(i, f, sum_PI);
				}
			}

			double[][] S_qff_arr = new double[numFactors][numFactors];//Sq(d,d')
			for (int f = 0; f < numFactors; f++) { // f* in the paper
				for (int f_ = 0; f_ < numFactors; f_++) {// f in the paper
					double S_qff_ = 0;
					for (int i = 0; i < numItems; i++) {
						alpha_n = Wi[i];
						S_qff_ += alpha_n * qyd.get(i, f) * qyd.get(i, f_);
					}
					S_qff_arr[f][f_] = S_qff_;
				}
			}

			for (int lstar = 0; lstar < PC; lstar++) {
				
				SSparseVector sparse_lstar = ZX.getCol(lstar);
				int nonZeroCount = sparse_lstar.nonZeroCount();
				for (int c_index = 0; c_index < nonZeroCount; c_index++) {
					int c_index_ = sparse_lstar.indexList().get(c_index);
					SparseVector pc = userCache.get(c_index_);
					int[] is = pc.getIndex();// positive item array
					for (int pos = 0; pos < is.length; pos++) {
						Prediction.setValue(c_index_, is[pos], predict(c_index_, is[pos]));//Save time
					}
				}
				for (int f = 0; f < numFactors; f++) {
					double JP_Derivative = 0;
					double JA_Derivative = 0; 
					double vals = 0;
					for (int c_index = 0; c_index < nonZeroCount; c_index++) {

						int c_index_ = sparse_lstar.indexList().get(c_index);
						SparseVector pc = userCache.get(c_index_);
						int[] is = pc.getIndex();// positive item array
						for (int pos = 0; pos < is.length; pos++) {
							int i = is[pos];// item i [0, |PI|]
							alpha_n = Wi[i];
							
							double x_cpos = Prediction.getValue(c_index_, i);
							double y_OneDerivative = qyd.get(i, f) * ZX.getValue(c_index_, lstar);
							JP_Derivative += 2 * (alpha_p - alpha_n) * (x_cpos - alpha_p / (alpha_p - alpha_n))
									* y_OneDerivative;// eq.(2)
							vals = (alpha_p - alpha_n) * (x_cpos - alpha_p / (alpha_p - alpha_n))
									* (x_cpos - alpha_p / (alpha_p - alpha_n));
							loss += vals;
						}
						for (int f_ = 0; f_ < numFactors; f_++) {
							JA_Derivative += 2 * S_qff_arr[f][f_] * pxf.get(c_index_, f_)
									* ZX.getValue(c_index_, lstar);// eq.(18)
						}
					}
					JP_Derivative += 2 * regU * V.get(lstar, f);// regularization,
																	// empirically
																	// can be
																	// omited

					double grad = (JP_Derivative + JA_Derivative);
																		
					
					double newgrad=V_grad.get(lstar, f);
					newgrad+=grad*grad;
					V_grad.set(lstar, f, newgrad);
					newgrad=(1.0 / (Math.sqrt(newgrad)+epsilon)) * grad;
					


					V.add(lstar, f, -lRate * newgrad);
					// update pxf w_new-w_old=-lRate * grad
					for (int c_index = 0; c_index < nonZeroCount; c_index++) {
						int c_index_ = sparse_lstar.indexList().get(c_index);
						double phi_fc = pxf.get(c_index_, f) + ZX.getValue(c_index_, lstar) * (-lRate * newgrad);
						pxf.set(c_index_, f, phi_fc);
					}
				}
			}

			// calculate JA(\Theta) loss, does not affact the training result
			for (int f = 0; f < numFactors; f++) {
				for (int f_ = 0; f_ < numFactors; f_++) {
					double vals = 0;
					for (int c = 0; c < numContexts; c++) {	
						vals += pxf.get(c, f) * pxf.get(c, f_);
					}
					loss += vals * S_qff_arr[f][f_];
				}
			}

			double[][] S_pff_arr = new double[numFactors][numFactors];
			for (int f = 0; f < numFactors; f++) { // f* in the paper
				for (int f_ = 0; f_ < numFactors; f_++) {// f in the paper
					double S_pff_ = 0;
					for (int c = 0; c < numContexts; c++) {
						S_pff_ += pxf.get(c, f) * pxf.get(c, f_);
					}
					S_pff_arr[f][f_] = S_pff_;
				}
			}

			for (int lstar = PC; lstar < V_numRows; lstar++) {
				// find which c have lstar
				SSparseVector sparse_lstar = ZY.getCol(lstar);
				int nonZeroCount = sparse_lstar.nonZeroCount();
				for (int i_index = 0; i_index < nonZeroCount; i_index++) {
					int i_index_ = sparse_lstar.indexList().get(i_index);
					SparseVector pi = itemCache.get(i_index_);
					int[] cs = pi.getIndex();// positive item array
					for (int pos_c = 0; pos_c < cs.length; pos_c++) {
						Prediction.setValue(cs[pos_c], i_index_, predict(cs[pos_c], i_index_));
					}
				}

				for (int f = 0; f < numFactors; f++) {
					double JP_Derivative = 0;
					double JA_Derivative = 0;
					double vals = 0;
					for (int i_index = 0; i_index < nonZeroCount; i_index++) {
						int i_index_ = sparse_lstar.indexList().get(i_index);
						alpha_n = Wi[i_index_];
						SparseVector pi = itemCache.get(i_index_);
						int[] cs = pi.getIndex();// positive item array
						for (int pos_c = 0; pos_c < cs.length; pos_c++) {// the
																			// context
																			// that
																			// assigned
																			// a
																			// positive
																			// feedback
																			// to
																			// item
																			// i_index
							int c = cs[pos_c];
							
							double x_cpos = Prediction.getValue(c, i_index_);
							double y_OneDerivative = pxf.get(c, f) * ZY.getValue(i_index_, lstar);
							JP_Derivative += 2 * (alpha_p - alpha_n) * (x_cpos - alpha_p / (alpha_p - alpha_n))
									* y_OneDerivative;

							vals = (alpha_p - alpha_n) * (x_cpos - alpha_p / (alpha_p - alpha_n))
									* (x_cpos - alpha_p / (alpha_p - alpha_n));
							loss += vals;
						}
						for (int f_ = 0; f_ < numFactors; f_++) {
							JA_Derivative += 2 * S_pff_arr[f][f_] * qyd.get(i_index_, f_)
									* ZY.getValue(i_index_, lstar) * alpha_n;// eq.(18)
						}
					}

					JP_Derivative += 2 * regI * V.get(lstar, f);// regularization,
																	// empirically
																	// can be
																	// omited
					double grad = (JP_Derivative + JA_Derivative);

					double newgrad=V_grad.get(lstar, f);
					newgrad+=grad*grad;
					V_grad.set(lstar, f, newgrad);
					newgrad=(1.0 / (Math.sqrt(newgrad)+epsilon)) * grad;
					

					V.add(lstar, f, -lRate * newgrad);

					// update qyd w_new-w_old=-lRate * grad
					for (int i_index = 0; i_index < nonZeroCount; i_index++) {
						int i_index_ = sparse_lstar.indexList().get(i_index);
						double psi_fi = qyd.get(i_index_, f) + ZY.getValue(i_index_, lstar) * (-lRate * newgrad);
						qyd.set(i_index_, f, psi_fi);
					}
				}
			}
			
			for (int f = 0; f < numFactors; f++) {
				for (int f_ = 0; f_ < numFactors; f_++) {
					double vals = 0;
					for (int i = 0; i < numItems; i++) {
						alpha_n = Wi[i];
						vals += alpha_n * qyd.get(i, f) * qyd.get(i, f_);// eq(12)
					}
					loss += vals * S_pff_arr[f][f_];
				}
			}

			if (isConverged(iter))
				break;
			if (iter == 1) {
				long endTime = System.currentTimeMillis();
				System.out.println("Evgm_mfsi-time:" + (endTime - startTime) + "milliseconds");
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
