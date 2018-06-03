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

import java.util.List;

import librec.data.DenseVector;
import librec.data.SparseMatrix;
import librec.data.SparseVector;
import librec.intf.IterativeRecommender;
import happy.coding.io.Strings;
import happy.coding.math.Randoms;

/**
 * 
 * Biase MF with 0/1 or Frequency 
 * Emploit Boostrap sampling
 * 
 * @author fajieyuan
 * 
 */
public class wMF01_allnegative extends IterativeRecommender {
	private double alpha_p = 1, alpha_n = 0.6;
	private float rho = 1;
	public wMF01_allnegative(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);
		isRankingPred = true;
		alpha_n = algoOptions.getFloat("-alpha_n");
		rho=algoOptions.getFloat("-rho");
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
		long startTime=0;
		for (int iter = 1; iter <= numIters; iter++) {
			if(iter==1)
			{
				 startTime = System.currentTimeMillis();
			}
			loss = 0;
			for (int u = 0; u < numUsers; u++) {
				SparseVector pu = userCache.get(u);// positive items for u
				int[] is = pu.getIndex();// positive item array
				for (int pos = 0; pos < is.length; pos++) {
					//Update positive
					int i=is[pos];
//					double rui = trainMatrix.get(u, i);////MF-Freq
					double rui =1;//MF-01
					double pui = predict(u, i);
					double eui = rui - pui;

					loss += eui * eui;
					// update factors
					for (int f = 0; f < numFactors; f++) {
						double puf = P.get(u, f), qif = Q.get(i, f);

						P.add(u, f, lRate * (eui * qif - regU * puf));
						Q.add(i, f, lRate * (eui * puf - regI * qif));

						loss += regU * puf * puf + regI * qif * qif;
					}
				}
//				for (int j = 0; j < numItems; j++) {
				int n=(int)Math.ceil(is.length*rho);
				for (int numSample = 0; numSample <n; numSample++) {
					int j=Randoms.uniform(numItems);
					
					if (pu.contains(j))
						continue;
					//Update negative
					double ruj = trainMatrix.get(u, j);

//					double ruj = 0;
					double puj = predict(u, j);
					double euj = ruj - puj;

					loss += euj * euj;
					// update factors
					for (int f = 0; f < numFactors; f++) {
						double puf = P.get(u, f), qjf = Q.get(j, f);
						P.add(u, f, lRate * (euj * qjf* alpha_n - regU * puf));
						Q.add(j, f, lRate * (euj * puf* alpha_n - regI * qjf));
						loss += regU * puf * puf + regI * qjf * qjf;
					}
				}
			}
			if (isConverged(iter))
				break;
			if(iter==1)
			{
				long endTime = System.currentTimeMillis();
				System.out.println("GD_MFSI-time:"+(endTime - startTime)+"milliseconds");
			}
		}
	}

	@Override
	public String toString() {
		return Strings.toString(new Object[] { alpha_n,rho, binThold, numFactors, initLRate, maxLRate, regU, regI, numIters }, ",");
	}
}
