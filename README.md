# Learning RNA Secondary Structure

### Motivation:
This project seeks to explore the ability of Deep Neural Networks (DNNs) to learn RNA secondary structure from only sequence data. DNNs have been shown to be extremely powerful when modelling RNA sequence data in applications such as predicting RNA Binding Protein binding events. Additionally, it has been suggested that DNNs make accurate predictions on binding events that involve the secondary structure of the molecule, without supplying the structure as input to the DNN. Thus these DNNs are able to learn secondary structure while only being fed the sequence data. 

Interogating what a DNN has learned is incredibly difficult. While these models might predict very powerfully for problems that involve secondary structure, we cannot make statements about what the model has learned simply from its prediction scores. To interpret a model's learned information, I developed a tool called Second Order Mutagenesis, which systematically mutates pairs of nucelotides in a sequence and tests the trained DNN on these mutants to identify loss and rescuses in scores that can be interpretted as base pairs.

### Overview of repository 
This repository contains numerous experiments performed on a) Toy RNA simulations and b) real RNA family simulations from RFAM (http://rfam.xfam.org). Second Order Mutagenesis is used to decipher how much of the secondary structure a model has learned.

Experiments are performed on representative examples of three classes of Neural Network Architecture:
- Multi-layer Perceptrons
- Convolutional Neural Networks
- Recurrent Neural Networks.

These experiments provide insight into the different abilities of DNN classes to learn RNA secondary structure and build intuition for possible incorporating Neural Network models into RNA homology search tools.
