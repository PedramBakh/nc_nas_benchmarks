from tabular_benchmarks import FCNetProteinStructureBenchmark
from tabular_benchmarks.nas_cifar10 import NASCifar10A

#b = FCNetProteinStructureBenchmark(data_dir="./fcnet_tabular_benchmarks/")
b = NASCifar10A(data_dir="/home/student/pedram-local/nas_benchmarks/")
cs = b.get_configuration_space()
config = cs.sample_configuration()

print("Numpy representation: ", config.get_array())
print("Dict representation: ", config.get_dictionary())

max_epochs = 108
y, cost = b.objective_function(config, budget=max_epochs)
print(y, cost)


