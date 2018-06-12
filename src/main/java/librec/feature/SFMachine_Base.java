package librec.feature;


import java.util.ArrayList;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;

import data.gla.uni.data.structure.complementary.SSparseMatrix;
import data.gla.uni.data.structure.complementary.SSparseVector;
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.MatrixEntry;
import librec.data.SparseMatrix;
import librec.intf.IterativeRecommender;

/**
 * 
 * a simplified version of FM by only considering interaction between context
 * and item, we do not consider the iteractions between two context or two item
 * variables.  Note when there is only user and item, SFMachine=MF
 * Rendle, Steffen. "Factorization machines with libfm." ACM
 * Transactions on Intelligent Systems and Technology (TIST) 3.3 (2012): 57.
 * Different from SFMachine, we use sparse matrix to express ZX (Vsize C) and ZY (Isize I)
 * @author fajieyuan
 */

public class SFMachine_Base extends IterativeRecommender {

	protected DenseVector sum_PC;
	protected DenseVector sum_PI;
//	protected int PC = 943;
//	protected int PI = 1683;
//	protected int PC = 8021;
//	protected int PI = 9080;
//	protected int PC = 143;
//	protected int PI = 1536;
//	protected int PC = 675;
//	protected int PI = 3403;
	
//	protected List<SparseVector> x_iList;
//	protected List<SparseVector> x_cList;
//	protected SSparseMatrix ZX;
//	protected SSparseMatrix ZY;
	public SFMachine_Base(SparseMatrix rm, SparseMatrix tm, int fold) {
		super(rm, tm, fold);
		PC = algoOptions.getInt("-pc");
		PI = algoOptions.getInt("-pi");
	}

	protected void initModel() throws Exception {

//		System.out.println(PC);
		Table<Integer, Integer, Double> XdataTable = HashBasedTable.create();
//		V = new DenseMatrix(PC + PI, numFactors);
		V = new DenseMatrix(PC + PI+2, numFactors); //usually a big V is no problem, we here add 2 consider
		//some times the index of  data is from 1.
		// initialize model
		if (initByNorm) {
			V.init(initMean, initStd);
		} else {
			V.init(); // P.init(smallValue);
		}

		sum_PC = new DenseVector(numFactors);
		sum_PI = new DenseVector(numFactors);

		sum_PC.init(0);
		sum_PI.init(0);

		// ------------------

		int I=numItems;
		ZY=new SSparseMatrix(I,V.numRows());
		for (int i = 0; i < I; i++) {
			String o_i=rateDao.getItemId(i);
			int index_f = 0;
			if(o_i.contains("-"))
			{
				String[] temp=o_i.split("-");
				int size=temp.length;
				for (int field = 0; field < size; field++) {
					 index_f=Integer.parseInt(temp[field]);
					 ZY.setValue(i, index_f+PC, 1);		 
					//Only for writing paper, when test features,e.g., user-previous item,
					 //we want to test the performacne if there is no previous item,i.e.,x(c. last item id,0)
//					 if(field==1)
//						 ZY.setValue(i, index_f+PC, 0);
				}
			}else
			{
				index_f=Integer.parseInt(o_i);
				ZY.setValue(i, index_f + PC, 1);// be careful
//				ZY.setValue(i,i + PC, 1);
			}
		}
		int C=numUsers;
		ZX=new SSparseMatrix(C,V.numRows());

		for (int c = 0; c < C; c++) {
			int index_f = 0;
			String o_c=rateDao.getUserId(c);//o_c structure should be like 5-12-455-11, here o_c is original userid
			if (o_c.contains("-")) {
				String[] temp=o_c.split("-");
				int size=temp.length;
				
				for (int field = 0; field < size; field++) {
					 index_f=Integer.parseInt(temp[field]);
					 ZX.setValue(c, index_f, 1);
					 //Only for writing paper, when test features,e.g., user-previous item,
					 //we want to test the performacne if there is no previous item,i.e.,x(c. last item id,0)
//					 if(field==1)
//						 ZX.setValue(c, index_f, 0);
				}
				
			}
			else
			{
				 index_f=Integer.parseInt(o_c);
				 ZX.setValue(c, index_f, 1);// be careful
			}
		}

	}

	/**
	 * x----feature,e.g.,user, item, distance
	 */
	@Override
	protected void buildModel() throws Exception {

		for (int iter = 1; iter <= numIters; iter++) {
			loss = 0;
			for (MatrixEntry me : trainMatrix) {
				
				int cid = me.row(); // user
				double ruj = me.get();//
				int jid = me.column(); // item


				double pred = predict(cid, jid);
				double euj = (ruj - pred);
				loss += euj * euj;


				SSparseVector x_c = ZX.getRow(cid);
				SSparseVector x_i =  ZY.getRow(jid);
				
				
				for (int f = 0; f < numFactors; f++) {

					for (int c = 0; c < x_c.nonZeroCount(); c++) {
						int cdx = x_c.indexList().get(c);
//						int cdx = x_c.getIndexList().get(c);
						double grad = sum_PI.get(f);
						V.add(cdx,
								f,
								lRate
										* (euj * grad - regK2
												* V.get(cdx, f)));
						// loss += regK2 * V.get(cdx, f) * V.get(cdx, f);
					}
					for (int i = 0; i < x_i.nonZeroCount(); i++) {
//						int idx = x_i.getIndexList().get(i);
						int idx = x_i.indexList().get(i);
						double grad = sum_PC.get(f);
						V.add(idx,
								f,
								lRate
										* (euj * grad -  regK2
												* V.get(idx, f)));
						// loss += regK2 * V.get(idx, f) * V.get(idx, f);
					}
				}
			}
			if (isConverged(iter))
				break;

		}
	}

	protected double predict(int c, int i) throws Exception {
		double pre = 0;
		pre = sumsum(c, i);
		return pre;
	}

	protected double sumsum(int c, int i) {
		// TODO Auto-generated method stub
		double result = 0;
		SSparseVector PCV=ZX.getRow(c);
		SSparseVector PIV=ZY.getRow(i);

		int PCs =PCV.nonZeroCount();
		int PIs = PIV.nonZeroCount();
		ArrayList<Integer> pivindexlist=PIV.indexList();//slow should use sparsevector directly
		ArrayList<Integer> pcvindexlist=PCV.indexList();
		for (int f = 0; f < numFactors; f++) {
			double sum_j = 0;
			double sum_con = 0;
			sum_PC.set(f, 0);
			sum_PI.set(f, 0);

			for (int j = 0; j <  PIs; j++) {
				int jidx=pivindexlist.get(j );
				sum_j += V.get(jidx, f)*ZY.getValue(i, jidx);
			}
			sum_PI.set(f, sum_j);
			for (int con = 0; con < PCs; con++) {// context
				int cidx=pcvindexlist.get(con );
				sum_con += V.get(cidx, f) *ZX.getValue(c, cidx);
				sum_PC.set(f, sum_con);
			}
			result += sum_con* sum_j;
		}
		return result;
	}
}
