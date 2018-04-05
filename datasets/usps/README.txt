# matlab/octave storage, 
# use scipy.io.loadmat to load in Python

contains two variables:
X - 9298x256 double - image data
	each row is from a 16x16 image
y - 9298x1 double - ground-truth data (labels)
	each row is a integer 1-10 (stored as double) 
	Note: 1=0, 2=1,..., 10=9
