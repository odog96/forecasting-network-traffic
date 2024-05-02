import os
import cdsw
#from joblib import dump, load

import json

@cdsw.model_metrics
def add(args):
  result = args["a"] + args["b"]
  return result


#add({"a": 3, "b": 5})
