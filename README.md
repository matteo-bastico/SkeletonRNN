# SkeletonRNN

<p>
SkeletonRNN is a Recurrent Deep Neural Network to estimate miss-detected skeleton 3D joints.  
</p>

<p style="text-align: center">
    <img src="images/architecture.png" style="height: 300px">
</p>

# Dataset for SkeletonRNN training and testing

<p>
The train set contains 1035 sequences of 
complete skeletons which are augmented during 
the training to simulate loss of joints. The test 
set contains 259 sequences of skeletons 
with missing points and the ground-truth is 
provided. Both dataset are in the Data folder.

Data are stored in .npy files. 
Each of them contains a list of skeleton 
sequences saved as Numpy array 
with shape (L, N, D) where L is the sequence 
length, N is the number of skeleton 
points (18 for Intel RealSense) and D is the 
dimensionality (3). In the testing dataset 
missing points are represented with [-1, -1, -1]. 
Training skeletons are all complete, 
i.e. without missing points.
</p>