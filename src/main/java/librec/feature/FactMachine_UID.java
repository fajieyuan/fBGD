package librec.feature;



import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.MatrixEntry;
import librec.data.SparseMatrix;
import librec.data.SparseVector;
import librec.fajie.MProUtil;
import librec.intf.IterativeRecommender;
/**
 * 
 * Rendle, Steffen. "Factorization machines with libfm." ACM Transactions on Intelligent Systems and Technology (TIST) 3.3 (2012): 57.
 * @author fajieyuan
 */

public class FactMachine_UID  extends IterativeRecommender{
	protected  double w_0=0;
	protected  SparseVector x;
	protected  DenseVector sum;
	protected  DenseVector sum_sqr;
//	protected  double regK0=0.1,regK1=0.1,regK2=0.001;//parameters
    double max=Double.MAX_VALUE;
	public FactMachine_UID(SparseMatrix rm, SparseMatrix tm, int fold) {
		super(rm, tm, fold);
	}
	protected void initModel() throws Exception {
	
		V = new DenseMatrix(super.x_size, numFactors);
		// initialize model
		if (initByNorm) {
			V.init(initMean, initStd);
		} else {
			V.init(); // P.init(smallValue);
		}
		w = new DenseVector(x_size);
		sum = new DenseVector(numFactors);
		sum_sqr = new DenseVector(numFactors);

		w.init(0);
		sum.init(0);
		sum_sqr.init(0);


	}
	/**
	 * x----feature,e.g.,user, item, distance
	 */
	@Override
	protected void buildModel() throws Exception {

	
		for (int iter = 1; iter <= numIters; iter++) {
			loss = 0;
			for (MatrixEntry me : trainMatrix) {
				x=new SparseVector(super.x_size);
				int u = me.row(); // user 
				double ruj = me.get();//
				int j = me.column(); // item
				x.set(u, 1);
				x.set(j+super.user_size, 1);

				String uid=rateDao.getUserId(u);
				String iid=rateDao.getItemId(j);
				
				double ui_dis=MProUtil.uidmap.get(uid+MProUtil.slash+iid)==null?max:MProUtil.uidmap.get(uid+MProUtil.slash+iid);
				int dis_ui=MProUtil.Norm(ui_dis);
				x.set(super.user_size+super.item_size+dis_ui,1);
				
				double pred = predict(x);
	
				double euj = (ruj - pred);
				loss += euj * euj;
				
				if (k0 == 1) {
					w_0 += lRate * (2 * euj - 2 * regK0 * w_0);
					loss+=regK0*w_0*w_0;
				}
				
				// update factors
				if (k1 == 1) {
					for (int i = 0; i < x.size(); i++) {
						int idx=x.getIndexList().get(i);
						w.add(idx,
								lRate
										* (2 * euj * x.get(idx) - 2 * regK1
												* w.get(idx)));
						loss+=regK1*w.get(idx)*w.get(idx);
					}
				}
				if (k2 == 1) {
					for (int f = 0; f < numFactors; f++) {
						for (int i = 0; i < x.size(); i++) {
							int idx=x.getIndexList().get(i);
							double grad = sum.get(f) * x.get(idx)
									- V.get(idx, f) * x.get(idx)
									* x.get(idx);
							V.add(idx,
									f,
									lRate
											* (2 * euj * grad - 2 * regK2
													* V.get(idx, f)));
							loss+=regK2*V.get(idx, f)*V.get(idx, f);
						}
					}
				}
			}
			loss *= 0.5;
			if (isConverged(iter))
				break;

		}
	}

	protected double predict(SparseVector x) throws Exception {
		double pre = 0;
		if (k0 == 1) {
			pre += w_0;
		}
		if (k1 == 1) {
			pre += sum( x);
		}
		if (k2 == 1) {
			pre += sumsum(x);
		}
		return pre;
	}
	private double sumsum( SparseVector x) {
		// TODO Auto-generated method stub
		double result = 0;
		for (int f = 0; f < numFactors; f++) {
			double sum_f = 0;
			double sum_sqr_f = 0;
			sum.set(f, 0);
			sum_sqr.set(f, 0);
             //	91480	1.000000                    SparseVector bug show three  values but only two non-zero,cannot identify 0 automatically
			for (int i = 0; i < x.size(); i++) {

				int idx = x.getIndexList().get(i);//Be careful here  x.add(100, 0) x.getIndexList() will think 0 also exists, kinda bug
				double d = V.get(idx, f) * x.get(idx);
				sum_f += d;
				sum_sqr_f += d * d;
				sum.set(f, sum_f);
				sum_sqr.set(f, sum_sqr_f);
			}
			result += 0.5 * (sum.get(f) * sum.get(f) - sum_sqr.get(f));
		}
		return result;
	}
	private double sum(SparseVector x) {
		// TODO Auto-generated method stub
		double sum = 0;
		for (int i = 0; i <  x.size(); i++) {
			int idx=x.getIndexList().get(i);
			sum += w.get(idx) * x.get(idx);
		}
		return sum;
	}
}
