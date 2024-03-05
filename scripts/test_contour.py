from utils.similarity_strategy import *

sim_context = SimilarityContext()
sim_context.set_strategy(FeatureBasedMatchingStrategy())
in_file1 = '../data/objects/ENDSTOP/img/lab/mdi_5_0.npy'
sim_s = []
for i in range(6):
    in_file2 = f'../data/objects/ENDSTOP/img/gen/di_5_{i}.npy'
    sim_s.append(sim_context.compare_images(in_file1, in_file2, a_is_real=True))
print(sim_s)
