package librec.ranking;

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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import happy.coding.io.Strings;
import happy.coding.math.Randoms;
import librec.data.DenseVector;
import librec.data.SparseMatrix;
import librec.data.SparseVector;
import librec.intf.IterativeRecommender;


/**
 * 
 * word2vec parameter learning explained
 *  @author fajie yuan
 *  Have not test it
 * 
 */
public class Skipgram extends IterativeRecommender {
	private float rho;
	private double[] iidRelativeRank;
	private int lossf;
	private int negsize;
	  private double[] Wi;
	public Skipgram(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);
		isRankingPred = true;
		rho = algoOptions.getFloat("-rho");
		negsize= algoOptions.getInt("-negsize",25);
	}
	
	@Override
	protected void initModel() throws Exception {
		super.initModel();
		
		userCache = trainMatrix.rowCache(cacheSpec);
		itemBias = new DenseVector(numItems);
		// initialize user bias
		itemBias.init();
		
		//calculate popularity 
	    final double[] RankingPro;
		RankingPro=new double[numItems];
		for (int i = 0; i < numItems; i++) {
			RankingPro[i] = trainMatrix.column(i).size();// popularity
																// distribution,e.g.,[0.001,0.002,0.0023,...]
		}
		
		
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
			p[i] = Math.pow(p[i], rho);
			Z += p[i];
		}
		// assign weight
		Wi = new double[numItems];
		for (int i = 0; i < numItems; i++)
			Wi[i] =  p[i] / Z;

	}

	@Override
	protected void buildModel() throws Exception {

		for (int iter = 1; iter <= numIters; iter++) {

			loss = 0;
			//	 To test the feature influence of lastfm and yahoo, 100---300
			for (int s = 0, smax = numUsers * 300; s < smax; s++) {

				// randomly draw (u, i)
				int u = 0, i = 0, j = 0;
                List<Integer> W_neg=new ArrayList<Integer>();
				while (true) {
					u = Randoms.uniform(numUsers);
					SparseVector pu = userCache.get(u);

					if (pu.getCount() == 0)
						continue;

					int[] is = pu.getIndex();
					i = is[Randoms.uniform(is.length)];

					while(W_neg.size()<negsize)
					{
					j = Randoms.discrete(Wi);//sampling based on popularity
						if(!pu.contains(j))
							W_neg.add(j);
					}
					break;
				}
			
				// update parameters
				double xui = predict(u, i);
				loss += -Math.log(g(xui));
				double grad_p=g(xui)-1;//WO in word2vector explaining  E/h
				List<Double> grad_neg=new ArrayList<Double>();
				for (int k = 0; k < negsize; k++) {
					int neg=W_neg.get(k);
					double xuj = predict(u, neg);
					grad_neg.add(g(xuj));
					loss+=-Math.log(g(-xuj));
				}
				for (int f = 0; f < numFactors; f++) {
					double puf = P.get(u, f);
					double qif = Q.get(i, f);
					double qjf =0;
					double grad_allneg=0;
					for (int k = 0; k < negsize; k++) {
						int neg=W_neg.get(k );
					    qjf = Q.get(neg, f);
					    grad_allneg+=grad_neg.get(k)*qjf;
					}
					P.add(u, f, -lRate * (grad_p*qif+grad_allneg));//EQ(61) grad_p*qif+grad_allneg
					Q.add(i, f, -lRate * (grad_p*puf));
					//iterate all negative according to EQ(56)
					for (int k = 0; k < negsize; k++) {
						int neg=W_neg.get(k);
						Q.add(neg, f, -lRate * grad_neg.get(k)*puf);
					}
				}
			}
			if (isConverged(iter))
				break;
		}
	}


	@Override
	public String toString() {
		return Strings.toString(new Object[] { rho,negsize, binThold, numFactors, initLRate, maxLRate, regU, regI, numIters }, ",");
	}
}
