echo "Downloading the Celeb A Dataset ...." 
wget https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip
mkdir data/celeba_dataset 
unzip celeba.zip -d data/celeba_dataset/
rm celeba.zip
