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
public class MF01 extends IterativeRecommender {

	
	public MF01(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);

		isRankingPred = true;
//		initByNorm = false;
		
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
			for (int s = 0, smax = numUsers * 100; s < smax; s++) {

				// randomly draw (u, i, j)
				int u = 0, i = 0, j = 0;

				while (true) {
					u = Randoms.uniform(numUsers);
					SparseVector pu = userCache.get(u);

					if (pu.getCount() == 0)
						continue;

					int[] is = pu.getIndex();
					i = is[Randoms.uniform(is.length)];

					do {
						j = Randoms.uniform(numItems);
					} while (pu.contains(j));

					break;
				}
				//Update negative
				double ruj = trainMatrix.get(u, j);
				double puj = predict(u, j, false);
				double euj = ruj - puj;

				loss += euj * euj;
				// update factors
				for (int f = 0; f < numFactors; f++) {
					double puf = P.get(u, f), qjf = Q.get(j, f);

					P.add(u, f, lRate * (euj * qjf - regU * puf));
					Q.add(j, f, lRate * (euj * puf - regI * qjf));

					loss += regU * puf * puf + regI * qjf * qjf;
				}
				//Update positive
				double rui = trainMatrix.get(u, i);////MF-Freq
//				double rui =1;//MF-01
				double pui = predict(u, i, false);
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

			if (isConverged(iter))
				break;

		}
	}

	@Override
	public String toString() {
		return Strings.toString(new Object[] { binThold, numFactors, initLRate, maxLRate, regU, regI, numIters }, ",");
	}
//	protected double predict(int u, int j) {
////		return DenseMatrix.rowMult(P, u, Q, j)+itemBias.get(j);
//		return DenseMatrix.rowMult(P, u, Q, j);
////		return itemBias.get(j) ;
//	}
}
