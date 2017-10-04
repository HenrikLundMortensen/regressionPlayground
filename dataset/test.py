from ase.io import read

a = read('all_O10_unique.traj',index=':')
print([x.get_potential_energy() for x in a])
print(a[0].get_positions())
