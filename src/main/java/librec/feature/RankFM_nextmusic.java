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

import java.math.BigDecimal;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;

import happy.coding.io.Strings;
import happy.coding.math.Randoms;
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.SparseMatrix;
import librec.data.SparseVector;
import librec.fajie.MProUtil;
import librec.intf.IterativeRecommender;


/**
 * 
 * Rendle et al., <strong>
 * <strong>Pairwise ranking Factorization machine</strong>,
 * <strong>Performance can be largely improved by LambdaFM</strong>,
 * @author fajieyuan
 * 
 * For music recommendation smax = numUsers * 300;
 */
public class RankFM_nextmusic extends FactMachine_UID {
	public static double max=Integer.MAX_VALUE;
	public static String colon=":";
	public static String comma=",";
	private int lossf;
	private int numU;
	public RankFM_nextmusic(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);
		isRankingPred = true;
		lossf = algoOptions.getInt("-lossf");
		numU=algoOptions.getInt("-numU",300);
		
		PC = algoOptions.getInt("-pc");
		PI = algoOptions.getInt("-pi");
	}
	
	@Override
	protected void initModel() throws Exception {
		super.initModel();
		userCache = trainMatrix.rowCache(cacheSpec);
	}


	@Override
	protected void buildModel() throws Exception {

		DenseVector grad=new DenseVector(x_size);
		DenseVector grad_visited=new DenseVector(x_size);
		SparseVector x_i;
		SparseVector x_j;
		for (int iter = 1; iter <= numIters; iter++) {
			loss = 0;
			for (int s = 0, smax = numUsers * numU; s < smax; s++) {
				// randomly draw (u, i, j)
				int u = 0, i = 0, j = 0;
				
				while (true) {
					u = Randoms.uniform(numUsers);
					SparseVector pu = userCache.get(u);//Super Slow, strongly suggest to write your own code in this step

					if (pu.getCount() == 0)
						continue;

					int[] is = pu.getIndex();
					i = is[Randoms.uniform(is.length)];//Tag

					do {
						j = Randoms.uniform(numItems);//-------------1//For OCCF
					} 
					while (pu.contains(j));//--------------------2 OCCF
					break;
				}

				x_i=new SparseVector(x_size);
				x_j=new SparseVector(x_size);
				
				String userfield=rateDao.getUserId(u);
				int uid=Integer.parseInt(userfield.split("-")[0]);
				int premusicid=Integer.parseInt(userfield.split("-")[1]);
				
				String itemfield_i=rateDao.getItemId(i);//positive
				int nextmusicid_i=Integer.parseInt(itemfield_i.split("-")[0]);
				int nextartistid_i=Integer.parseInt(itemfield_i.split("-")[1]);
				
				String itemfield_j=rateDao.getItemId(j);
				int nextmusicid_j=Integer.parseInt(itemfield_j.split("-")[0]);
				int nextartistid_j=Integer.parseInt(itemfield_j.split("-")[1]);
				//1-1000,2-16057,1,1241361476000
				x_i.set(uid, 1);//user
				x_i.set(premusicid, 1);
				x_i.set(nextmusicid_i + PC, 1);
				x_i.set(nextartistid_i + PC, 1);
				
				x_j.set(uid, 1);//user
				x_j.set(premusicid, 1);
				x_j.set(nextmusicid_j + PC, 1);
				x_j.set(nextartistid_j + PC, 1);
								
                double si=trainMatrix.get(u, i);
                double sj=trainMatrix.get(u, j);
                double sij=si-sj;
                sij=1;//----------------------3OCCF
				
				double xui = predict(x_i);
				DenseVector sum_pos=sum( x_i);
				double xuj = predict(x_j);
				DenseVector sum_neg=sum(x_j);
				// update factors
				double xuij = xui - xuj;
				double Sij = sij>0? 1: (sij==0? 0: -1); 
				double pij_real=0.5*(1+Sij);//from ground truth
				double pij=g(xuij);
				double cmg=getGradMag(lossf, xuij);
				loss+=-pij_real*Math.log(pij)-(1-pij_real)*Math.log(1-pij);//Ranknet loss
				
				if (k1 == 1) {
					for (int c = 0; c < x_i.size(); c++) {
						int idx = x_i.getIndexList().get(c);
						grad.set(idx, 0);
						grad_visited.set(idx, 0);
					}
					for (int c = 0; c < x_j.size(); c++) {
						int idx = x_j.getIndexList().get(c);
						grad.set(idx, 0);
						grad_visited.set(idx, 0);
					}
					for (int c = 0; c < x_i.size(); c++) {
						int idx = x_i.getIndexList().get(c);
						grad.add(idx, x_i.get(idx));
					}
					for (int c = 0; c < x_j.size(); c++) {
						int idx = x_j.getIndexList().get(c);
						grad.add(idx, -x_j.get(idx));
					}
					//Updata w_positive including the common part e.g. bu
					//e.g.,postive w1 w3  negative w1 w4
					for (int c = 0; c < x_i.size(); c++) {
						int idx = x_i.getIndexList().get(c);
						if(grad_visited.get(idx)==0)
						{//Update like bu, bi
							w.add(idx,
									lRate
											* (cmg* (grad.get(idx) )-  regK1
													* w.get(idx)));
							grad_visited.set(idx, 1);
//							loss += regK1*w.get(idx)*w.get(idx);
						}
					}
					//Updata w_negative
					for (int c = 0; c < x_j.size(); c++) {
						int idx = x_j.getIndexList().get(c);
						if(grad_visited.get(idx)==0)
						{//Update like bu, bj
							w.add(idx,
									lRate
											* (cmg* (grad.get(idx)) -  regK1
													* w.get(idx)));
							grad_visited.set(idx, 1);
//							loss += regK1*w.get(idx)*w.get(idx);
						}
					}
				}
				//Update v_ij
				if (k2 == 1) {
					for (int f = 0; f < numFactors; f++) {
						for (int c = 0; c < x_i.size(); c++) {
							int idx = x_i.getIndexList().get(c);
							grad.set(idx, 0);
							grad_visited.set(idx, 0);
						}
						for (int c = 0; c < x_j.size(); c++) {
							int idx = x_j.getIndexList().get(c);
							grad.set(idx, 0);
							grad_visited.set(idx, 0);
						}
						for (int c = 0; c < x_i.size(); c++) {
							int idx = x_i.getIndexList().get(c);
							grad.add(
									idx,
									sum_pos.get(f) * x_i.get(idx)
											- V.get(idx, f) * x_i.get(idx)
											* x_i.get(idx));
						}
						for (int c = 0; c < x_j.size(); c++) {
							int idx = x_j.getIndexList().get(c);
							grad.add(
									idx,
									-(sum_neg.get(f) * x_j.get(idx) - V.get(
											idx, f)
											* x_j.get(idx)
											* x_j.get(idx)));
						}
						
						for (int c = 0; c < x_i.size(); c++) {
							int idx=x_i.getIndexList().get(c);
							if(grad_visited.get(idx)==0)
							{	
								V.add(idx,
										f,
										lRate
												* (cmg * grad.get(idx) -  regK2
														* V.get(idx, f)));
								grad_visited.set(idx, 1);
//								loss+=regK2*V.get(idx, f)*V.get(idx, f);
							}
						}
						for (int c = 0; c < x_j.size(); c++) {
							int idx=x_j.getIndexList().get(c);
							if(grad_visited.get(idx)==0)
							{
								V.add(idx,
										f,
										lRate
												* (cmg * grad.get(idx) - regK2
														* V.get(idx, f)));
								grad_visited.set(idx, 1);
//								loss+=regK2*V.get(idx, f)*V.get(idx, f);
							}
						}
					}
				}
			}
			if (isConverged(iter))
				break;
		}
	}

	@Override
	public String toString() {
		return Strings.toString(new Object[] {lossf, binThold, numFactors, initLRate, maxLRate, k0, k1 , k2,regK0, regK1, regK2, numIters, numU }, ",");
	}
	protected double predict(SparseVector x) throws Exception {
		return super.predict(x);
	}
	private DenseVector sum(SparseVector x) {
		DenseVector sum=new DenseVector(numFactors);
		// TODO Auto-generated method stub
		for (int f = 0; f < numFactors; f++) {
			double sum_f = 0;
			sum.set(f, 0);
			for (int i = 0; i < x.size(); i++) {
				int idx = x.getIndexList().get(i);
				double d = V.get(idx, f) * x.get(idx);
				sum_f += d;
				sum.set(f, sum_f);
			}
		}
		return sum;
	}

}
