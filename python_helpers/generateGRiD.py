from GRiD.URDFParser import URDFParser
from GRiD.GRiDCodeGenerator import GRiDCodeGenerator

parser = URDFParser()
URDF_PATH = "GRiD\\URDFBenchmarks\\panda.urdf"
robot = parser.parse(URDF_PATH)

#codegen = GRiDCodeGenerator(robot,False,True)
codegen = GRiDCodeGenerator(robot,False,True, FILE_NAMESPACE = "panda_grid")
codegen.gen_all_code(include_homogenous_transforms = True)