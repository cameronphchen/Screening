% Change the filenames if you've saved the files under different names
% On some platforms, the files might be saved as 
% train-images.idx3-ubyte / train-labels.idx1-ubyte
images = loadMNISTImages('/Users/ChimatChen/Dropbox/Research/OMPvsScreening/data/train-images-idx3-ubyte');
labels = loadMNISTLabels('/Users/ChimatChen/Dropbox/Research/OMPvsScreening/data/train-labels-idx1-ubyte');
 
imagesc(reshape(images(:,1),28,28))
colormap(gray)
% We are using display_network from the autoencoder code
%display_network(images(:,1:100)); % Show the first 100 images
%disp(labels(1:10));
