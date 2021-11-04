### In this project we will experiment with random hyperplanes 
### for classification. Your program will take a dataset as input and
### produce new features following the procedure below. The input is in
### the same format as for previous assignments.

Input 
1. training dataset with labels
2. test dataset with labels that may be 0 
3. number of new features k

For i = 0 to k do:
	a. Create random vector w where each wj is uniformly sampled between -1 and 1.
	
	b. Let xj be our training data points. Determine the largest and smallest wTxj
	across all xj. Select w0 randomly between [smallest wTxj, largest wTxj].

	c. Project training data X (each row is datapoint xj) onto w. 
	Let projection vector zi be Xw + w0 (here X has dimensions n by m and w is m by 1).
	Append (1+sign(zi))/2 as new column to the right end of Z. Remember that zi is
	a vector and so (1+sign(zi))/2 is 0 if the sign is -1 and 1 otherwise.
	
	d. Project test data X' (each row is datapoint xj) onto w. 
	Let projection vector z'i be X'w + w0. Append (1+sign(z'i))/2 as new column to 
	the right end of Z'. We create the test data in exactly the same way as we do
	the training except that we do it on X' the test data instead of X the training data.
	
1. Run hinge loss on Z and predict on Z' after standardizing the data. Remember to 
   standardize using the column lengths from the training data only and not the test 
   data.
2. Do values of k=10, 100, 1000, and 10000.
3. How does the error compare to hinge loss on original data X and X' for each k? 
   Don't forget to standardize the data before applying hinge loss. This will improve 
   the accuracy and speed of your search.

The output of your program are two files. In the first file called 
"original_output.txt" are the predicted labels of the original and standardized test 
data with hinge loss. In the second file called "01space_output.txt" are predicted 
labels of hinge loss in the zero-one feature space.


