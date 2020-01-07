package hex.gam.MatrixFrameUtils;

import hex.gam.GAMModel.GAMParameters.BSType;
import hex.gam.GamSplines.CubicRegressionSplines;
import water.MRTask;
import water.fvec.Frame;

public class GenerateGamMatrixOneColumn extends MRTask<GenerateGamMatrixOneColumn> {
  BSType _splineType;
  int _numKnots;  // number of knots
  double[] _knots;  // value of knots to use if specified by user
  double[][] _bInvD; // store inv(B)*D
  
  public GenerateGamMatrixOneColumn(BSType splineType, int numKnots, double[] knots, Frame gamx) {
    _splineType = splineType;
    _numKnots = numKnots;
    CubicRegressionSplines crSplines = new CubicRegressionSplines(numKnots, knots, gamx.vec(0).max(), gamx.vec(0).min());
    _knots = crSplines._knots;
    _bInvD = crSplines.genreateBIndvD(crSplines._hj);
  }
}


