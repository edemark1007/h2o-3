package hex.gam;

import hex.*;
import hex.glm.GLMModel.GLMParameters.Family;
import hex.glm.GLMModel.GLMParameters.GLMType;
import hex.glm.GLMModel.GLMParameters.Link;
import hex.glm.GLMModel.GLMParameters.Solver;
import water.Key;
import water.fvec.Frame;
import water.util.Log;

import java.io.Serializable;
import java.util.Arrays;

public class GAMModel extends Model<GAMModel, GAMModel.GAMParameters, GAMModel.GAMModelOutput> {
  
  @Override public ModelMetrics.MetricBuilder makeMetricBuilder(String[] domain) {
    return null;
    //return new MetricBuilderGAM(domain);
  }

  public GAMModel(Key<GAMModel> selfKey, GAMParameters parms, GAMModelOutput output) {
    super(selfKey, parms, output);
    assert(Arrays.equals(_key._kb, selfKey._kb));
  }

  ModelMetricsSupervised makeModelMetrics(Frame origFr, Frame adaptFr, String description) {
    Log.info("Making metrics: " + description);
    ModelMetrics.MetricBuilder mb = scoreMetrics(adaptFr);
    ModelMetricsSupervised mm = (ModelMetricsSupervised) mb.makeModelMetrics(this, origFr, adaptFr, null);
    mm._description = description;
    return mm;
  }
  

  @Override
  protected double[] score0(double[] data, double[] preds) {
    return new double[0];
  }

  @SuppressWarnings("WeakerAccess")
  public static class GAMParameters extends Model.Parameters {
    // the following parameters will be passed to GLM algos
    public boolean _standardize = false; // pass to GLM algo
    public Family _family;
    public Link _link; 
    public Solver _solver = Solver.AUTO;
    public double _tweedie_variance_power;
    public double _tweedie_link_power;
    public double _theta; // 1/k and is used by negative binomial distribution only
    public double _invTheta;
    public double [] _alpha;
    public double [] _lambda;
    public Serializable _missing_values_handling = MissingValuesHandling.MeanImputation;
    public double _prior = -1;
    public boolean _lambda_search = false;
    public int _nlambdas = -1;
    public boolean _non_negative = false;
    public boolean _exactLambdas = false;
    public double _lambda_min_ratio = -1; // special
    public boolean _use_all_factor_levels = false;
    public int _max_iterations = -1;
    public boolean _intercept = true;
    public double _beta_epsilon = 1e-4;
    public double _objective_epsilon = -1;
    public double _gradient_epsilon = -1;
    public double _obj_reg = -1;
    public boolean _compute_p_values = false;
    public boolean _remove_collinear_columns = false;
    public String[] _interactions=null;
    public StringPair[] _interaction_pairs=null;
    public boolean _early_stopping = true;
    public Key<Frame> _beta_constraints = null;
    public Key<Frame> _plug_values = null;
    // internal parameter, handle with care. GLM will stop when there is more than this number of active predictors (after strong rule screening)
    public int _max_active_predictors = -1;
    public boolean _stdOverride; // standardization override by beta constraints

    // the following parameters are for GAM
    public int[] _k; // array storing number of knots per basis function
    public String[] _gam_X; // array storing which predictor columns are needed
    public BSType[] _bs; // array storing basis functions, only support cr for now
    public double[] _scale;  // array storing scaling values to control wriggliness of fit
    public GLMType _glmType = GLMType.gam; // internal parameter
    
    public String algoName() { return "GAM"; }
    public String fullName() { return "General Additive Model"; }
    public String javaName() { return GAMModel.class.getName(); }
    public enum MissingValuesHandling {
      MeanImputation, PlugValues, Skip
    }

    @Override
    public long progressUnits() {
      return 1;
    }

    public long _seed = -1;
    
    public enum BSType {
      cr  // will support more in the future
    }
  }
  
  public static class GAMModelOutput extends Model.Output {
    String[] _coefficient_names;
    public int _best_lambda_idx; // lambda which minimizes deviance on validation (if provided) or train (if not)
    public int _lambda_1se = -1; // lambda_best + sd(lambda); only applicable if running lambda search with nfold
    public int _selected_lambda_idx; // lambda which minimizes deviance on validation (if provided) or train (if not)
    double[] _global_beta;
    private double[] _zvalues;
    private double _dispersion;
    private boolean _dispersionEstimated;

    public double dispersion(){ return _dispersion;}
    
    public GAMModelOutput(GAM b) { super(b); }

    @Override public ModelCategory getModelCategory() {
      return ModelCategory.Regression;
    } // will add others later
  }
}
