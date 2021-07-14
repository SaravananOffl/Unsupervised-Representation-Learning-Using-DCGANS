echo "Downloading the SHVN Dataset ...." 
mkdir data/shvn
wget http://ufldl.stanford.edu/housenumbers/train_32x32.mat -O ./data/shvn/train.mat
wget http://ufldl.stanford.edu/housenumbers/test_32x32.mat -O ./data/shvn/test.mat
