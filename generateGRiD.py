from GRiD.URDFParser import URDFParser
from GRiD.GRiDCodeGenerator import GRiDCodeGenerator

parser = URDFParser()
URDF_PATH = "GRiD\\URDFBenchmarks\\iiwa.urdf"
robot = parser.parse(URDF_PATH)

codegen = GRiDCodeGenerator(robot,False,True)
codegen.gen_all_code(include_homogenous_transforms = True)