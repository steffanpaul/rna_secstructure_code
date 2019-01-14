Experiment I did to see if using an excessively large MLP layer  allowed the model to cheat significantly and just not learn any structure. This proved to be the case in an experiment I ran on the simple toy hairpin stuff, but with RFAM simulations the larger MLPs tended to continue to do better. 

The .sto and .hdf5 files for riboswitch and trna are in a separate file called overkill in the larger rna_sectstructure folder. These include:

- riboswitch_100k.sto
- riboswitch_100k_d5.hdf5
- trna_100k_d5.hdf5
- trnasim_100k.sto