package hex.gam;

import hex.ModelBuilder;
import hex.ModelCategory;
import hex.glm.GLMModel.GLMParameters.Family;
import hex.glm.GLMModel.GLMParameters.Link;
import jsr166y.RecursiveAction;
import jsr166y.ForkJoinTask;
import water.Key;
import water.exceptions.H2OModelBuilderIllegalArgumentException;
import water.fvec.Frame;
import water.fvec.Vec;
import hex.gam.GAMModel.GAMParameters.BSType;
import hex.gam.MatrixFrameUtils.GenerateGamMatrixOneColumn;
import water.DKV;

import java.util.Arrays;


public class GAM extends ModelBuilder<GAMModel, GAMModel.GAMParameters, GAMModel.GAMModelOutput> {

  @Override
  public ModelCategory[] can_build() { return new ModelCategory[]{ModelCategory.Regression}; }

  @Override
  public boolean isSupervised() { return true; }

  @Override
  public BuilderVisibility builderVisibility() { return BuilderVisibility.Experimental; }
  
  @Override public boolean havePojo() { return false; }
  @Override public boolean haveMojo() { return false; }

  public GAM(boolean startup_once) {
    super(new GAMModel.GAMParameters(), startup_once);
  }
  public GAM(GAMModel.GAMParameters parms) { super(parms);init(false); }
  public GAM(GAMModel.GAMParameters parms, Key<GAMModel> key) { super(parms, key); init(false); }

  @Override public void init(boolean expensive) {
    super.init(expensive);
    if (expensive) {  // add custom check here
      if (error_count() > 0)
        throw H2OModelBuilderIllegalArgumentException.makeFromBuilder(GAM.this);
      
      if (!_parms._family.equals(Family.gaussian)) 
        error("_family", "Only gaussian family is supported for now.");
      if (!_parms._link.equals(Link.identity) && !_parms._link.equals(Link.family_default))
        error("_link", "Only identity or family_default link is supported for now.");
      if (_parms._gam_X==null)
        error("_gam_X", "must specify columns indices to apply GAM to.  If you don't have any, use GLM.");
      if (_parms._k==null) {  // user did not specify any knots, we will use default 10, evenly spread over whole range
        int numKnots = _train.numRows() < 10?(int)_train.numRows():10;
        _parms._k = new int[_parms._gam_X.length];  // different columns may have different 
        Arrays.fill(_parms._k, numKnots);
      }
    }
  }

  @Override
  public void checkDistributions() {  // will be called in ModelBuilder.java
    if (!_response.isNumeric()) {
      error("_response", "Expected a numerical response, but instead got response with " + _response.cardinality() + " categories.");
    }
  }

  @Override
  protected boolean computePriorClassDistribution() {
    return false; // no use, we don't output probabilities
  }

  @Override
  protected int init_getNClass() {
    return 1; // only regression is supported for now
  }

  @Override
  protected GAMDriver trainModelImpl() {
    return new GAMDriver();
  }
  
  @Override 
  protected int nModelsInParallel(int folds) {
    return nModelsInParallel(folds, 2);
  }

  @Override protected void checkMemoryFootPrint_impl() {
    ;
  }
  private class GAMDriver extends Driver {
    /***
     * This method will take the _train that contains the predictor columns and response columns only and add to it
     * the following:
     * 1. For each predictor included in gam_x, expand it out to calculate the f(x) and attach to the frame.
     * @return
     */
    Frame adaptTrain() {
      Frame orig = _parms.train();  // contain all columns, _train contains only predictors and responses
      int numGamFrame = _parms._gam_X.length;
      Key<Frame>[] gamFramesKey = new Key[numGamFrame];  // store the Frame keys of generated GAM column
      RecursiveAction[] generateGamColumn = new RecursiveAction[numGamFrame];
      for (int index=0; index < numGamFrame; index++) {
        final Frame predictVec = new Frame(orig.vec(_parms._gam_X[index]));  // extract the vector to work on
        final int numKnots = _parms._k[index];  // grab number of knots to generate
        final BSType splineType = _parms._bs[index];
        final int tIndex = index;
        generateGamColumn[tIndex] = new RecursiveAction() {
          @Override
          protected void compute() {
            Frame oneAugmentedColumn = new GenerateGamMatrixOneColumn(splineType, numKnots, null, predictVec).doAll(numKnots,Vec.T_NUM,predictVec).outputFrame(null, null, null);
            gamFramesKey[tIndex] = oneAugmentedColumn._key;
            DKV.put(oneAugmentedColumn);
          }
        };
      }
      ForkJoinTask.invokeAll(generateGamColumn);
      return _train;
    }
    
    @Override
    public void computeImpl() {
      init(true);     //this can change the seed if it was set to -1
      if (error_count() > 0)   // if something goes wrong, let's throw a fit
        throw H2OModelBuilderIllegalArgumentException.makeFromBuilder(GAM.this);
      
      _job.update(0, "Initializing model training");
      
      buildModel(); // build gam model 
    }

    public final void buildModel() {
      GAMModel model = new GAMModel(dest(), _parms, new GAMModel.GAMModelOutput(GAM.this));
      model.delete_and_lock(_job);

     Frame newTFrame = adaptTrain();  // get frames with correct predictors and spline functions
    }
  }
}
