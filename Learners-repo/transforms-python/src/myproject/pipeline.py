from transforms.api import Pipeline

from myproject import CeVAE
# from myproject import NetworkDeconfounder
# from myproject import TARNet
# from myproject import Lasso
my_pipeline = Pipeline()
# my_pipeline.discover_transforms(TARNet)
# my_pipeline.discover_transforms(NetworkDeconfounder)
my_pipeline.discover_transforms(CeVAE)
# my_pipeline.discover_transforms(Lasso)
