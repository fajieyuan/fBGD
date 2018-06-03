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

import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import happy.coding.io.Strings;
import happy.coding.math.Randoms;
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.SparseMatrix;
import librec.data.SparseVector;
import librec.intf.IterativeRecommender;


/**
 * 
 *LambdaFM: Learning Optimal Ranking with Factorization Machines Using Lambda Surrogates
 * A matrix factorization version FM version is online https://github.com/fajieyuan/LambdaFM
 * Note that the implementation is very slow because of the framework
 * There are many ways to optimze the code - sorry no time
 * @author fajie yuan
 * 
 */
public class LambdaMF_D extends IterativeRecommender {
	private double[] iidRelativeRank;
	private float rho;
	private int lossf;
	private int n;
	public LambdaMF_D(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);

		isRankingPred = true;
		rho = algoOptions.getFloat("-rho");
		lossf = algoOptions.getInt("-lossf");
		n= algoOptions.getInt("-n",10);
//		initByNorm = true;
	}
	
	@Override
	protected void initModel() throws Exception {
		super.initModel();
		
		userCache = trainMatrix.rowCache(cacheSpec);
//		MProUtil.getMap();
		itemBias = new DenseVector(numItems);
		// initialize user bias
		itemBias.init();
	}

	@Override
	protected void buildModel() throws Exception {

		for (int iter = 1; iter <= numIters; iter++) {

			loss = 0;
			// To test the feature influence of lastfm and yahoo, 100---300
			for (int s = 0, smax = numUsers * 300; s < smax; s++) {

				// randomly draw (u, i, j)
				int u = 0, i = 0, j = 0, g=0;

				while (true) {
					u = Randoms.uniform(numUsers);
					SparseVector pu = userCache.get(u);//quite slow

					if (pu.getCount() == 0)
						continue;

					int[] is = pu.getIndex();
					i = is[Randoms.uniform(is.length)];

					do {
						j = ChooseNeg(n,u);
					} while (pu.contains(j));

					break;
				}

				// update parameters
				double xui = predict(u, i);
				double xuj = predict(u, j);
				double xuij = xui - xuj;
				double vals = -Math.log(g(xuij));
				loss += vals;

//				double cmg = g(-xuij);
				double cmg=getGradMag(lossf, xuij);


				for (int f = 0; f < numFactors; f++) {
					double puf = P.get(u, f);
					double qif = Q.get(i, f);
					double qjf = Q.get(j, f);
					P.add(u, f, lRate * (cmg * (qif - qjf) - regU * puf));
					
					Q.add(i, f, lRate * (cmg * puf - regI * qif));
					Q.add(j, f, lRate * (cmg * (-puf) - regI * qjf));
					
					loss += regU * puf * puf + regI * qif * qif + regI * qjf * qjf;
				}
			}

			if (isConverged(iter))
				break;

		}
	}

	private int ChooseNeg(int size, int u) throws Exception {
		// TODO Auto-generated method stub
		if (size > numItems) {
			throw new IllegalArgumentException();
		}
		final double[] RankingPro;
		RankingPro = new double[numItems];
		Arrays.fill(RankingPro, -100.0);//For comparision below, otherwise element =0 , some scores are negative 

		for (int i = 0; i < size; i++) {
			int iid = Randoms.uniform(numItems);
			RankingPro[iid] = predict(u, iid);
		}
		
		Integer[] iidRank = new Integer[numItems];
		for (int i = 0; i < numItems; ++i)
			iidRank[i] = i;
		Arrays.sort(iidRank, new Comparator<Integer>() {
			@Override
			public int compare(Integer o1, Integer o2) {
				return (RankingPro[o1] > RankingPro[o2] ? -1
						: (RankingPro[o1] < RankingPro[o2] ? 1 : 0));
			}
		});
		double sum = 0.0;
		iidRelativeRank = new double[numItems];
		for (int i = 0; i < size; ++i) {
			int iid = iidRank[i];// iidRank [2360, 1248, 626, 2385, 2543] means
									// item 2360 rank first
//			iidRelativeRank[iid] = 1.0 - i * 1.0 / size;----------good result
			iidRelativeRank[iid] = Math.exp(-(i + 1) / (size*rho));
			sum += iidRelativeRank[iid];
		}
		Map<Integer,Integer> map=new HashMap<Integer, Integer>();
		double[] iidRelativeRank_small=new double[size];
		int k=0;
		for (int i = 0; i < iidRelativeRank.length; i++) {
			if (iidRelativeRank[i] != 0) {// non-zero elements=size
				iidRelativeRank[i] = iidRelativeRank[i] / sum;
				iidRelativeRank_small[k] = iidRelativeRank[i];
				map.put(k,  i);
				k++;
			}
		}
		int index=Randoms.discrete(iidRelativeRank_small);//It is quicker
		return map.get(index);
	}

	@Override
	public String toString() {
		return Strings.toString(new Object[] { lossf,binThold,rho, numFactors, initLRate, maxLRate, regU, regI, numIters }, ",");
	}

}
