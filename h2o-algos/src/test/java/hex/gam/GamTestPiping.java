package hex.gam;

import hex.gam.GAMModel.GAMParameters.BSType;
import hex.glm.GLMModel;
import org.junit.BeforeClass;
import org.junit.Test;
import water.Scope;
import water.TestUtil;
import water.fvec.Frame;

/***
 * Here I am going to test the following:
 * - model matrix formation with centering
 */
public class GamTestPiping extends TestUtil {
  @BeforeClass
  public static void setup() {
    stall_till_cloudsize(1);
  }
  
  @Test
  public void testAdaptFrame() {
    try {
      Scope.enter();
      Frame train = parse_test_file("./smalldata/gam_test/gamDataRegressionOneFun.csv");
      Scope.track(train);
      Frame trainCorrectOutput = parse_test_file("./smalldata/gam_test/gamDataRModelMatrixCenterDataOneFun.csv");
      Scope.track(trainCorrectOutput);
      GAMModel.GAMParameters parms = new GAMModel.GAMParameters();
      parms._bs = new BSType[]{BSType.cr};
      parms._k = new int[]{6};
      parms._response_column = train.name(2);
      parms._ignored_columns = new String[]{train.name(0)}; // row of ids
      parms._gam_X = new String[]{train.name(1)};
      parms._train = train._key;
      parms._family = GLMModel.GLMParameters.Family.gaussian;
      parms._link = GLMModel.GLMParameters.Link.family_default;

      GAMModel model = new GAM(parms).trainModel().get();
      Scope.track_generic(model);
    } finally {
      Scope.exit();
    }
  }
}
