## Build

````
cd [wherever this repo is]
mvn -Pspark-deploy clean package
````

## Deployment

### Set up files on edge node.

On Linux:

````
curl -L -O http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar
tar -xf 256_ObjectCategories.tar
mkdir -p 256_ObjectCategories/train 256_ObjectCategories/test 256_ObjectCategories/valid
find ./256_ObjectCategories/ -regextype posix-extended -type f -regex ".*/[0-9]{3}\..+" -print | shuf | head -n 5000 | xargs -I {} mv {} ./256_ObjectCategories/valid/
find ./256_ObjectCategories/ -regextype posix-extended -type f -regex ".*/[0-9]{3}\..+" -print | shuf | head -n 6000 | xargs -I {} mv {} ./256_ObjectCategories/test/
find ./256_ObjectCategories/ -regextype posix-extended -type f -regex ".*/[0-9]{3}\..+" -print | xargs -I {} mv {} ./256_ObjectCategories/train/
find ./256_ObjectCategories/ -regextype posix-extended -type d -regex ".*/[0-9]{3}\..+" -delete
````

Copy to HDFS.

````
hadoop fs -put ./256_ObjectCategories
````

### Copy app to edge node.

````
scp dl4j-cnn/target/dl4j-cnn-1.0.0-jar-with-dependencies.jar [cluster]:
````

If using the web UI:

````
scp dl4j-ui/target/dl4j-ui-1.0.0-jar-with-dependencies.jar [cluster]:
````

### Featurize the input data.

**Note:** Optionally specify a path to the VGG model via `--modelPath /path/to/vgg16.zip`.

**Note:** This job may take on the magnitude of hours, depending on cluster size. If you'd like to use data that has already
been featurized, instructions are below.

````
GC_FLAGS="-XX:+UseG1GC -XX:MaxGCPauseMillis=200 -Dorg.bytedeco.javacpp.maxretries=100 -Dorg.bytedeco.javacpp.maxbytes=25000000000"
BIGTOP_JAVA_MAJOR=8 # ensures Java 8 on distros like CDH
spark2-submit \
--master yarn \
--deploy-mode client \
--conf spark.driver.extraJavaOptions="$GC_FLAGS" \
--conf spark.executor.extraJavaOptions="$GC_FLAGS" \
--conf spark.locality.wait=0 \
--conf spark.driver.maxResultSize=10g \
--conf spark.yarn.executor.memoryOverhead=27g \
--conf spark.yarn.driver.memoryOverhead=27g \
--conf spark.executor.cores=8 \
--conf spark.kryo.registrator=org.nd4j.Nd4jRegistrator \
--conf spark.kryoserializer.buffer.max=2047m \
--conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
--executor-memory 10g \
--driver-memory 10g \
--num-executors=5 \
--class "com.cloudera.datascience.dl4j.cnn.examples.caltech256.SaveFeaturizedData" \
dl4j-cnn-1.0.0-jar-with-dependencies.jar \
--numClasses 257 \
--outputLayer fc2 \
--imagePath hdfs:///path/to/256_ObjectCategories/ \
--savePath hdfs:///path/to/256_ObjectCategories_Featurized_FC2/
````

### \[Optional\] Use pre-computed feature data

Computing the features is compute intensive due to the size of the VGG16
model. The featurized data, saved in parquet format, can be downloaded
here: [Caltech256_FeaturizedFC2](https://storage.googleapis.com/dl4j-256-objectcategories/256_ObjectCategories_Featurized_FC2.tar)

Extract this folder and put it into HDFS.

````
curl -L -O https://storage.googleapis.com/dl4j-256-objectcategories/256_ObjectCategories_Featurized_FC2.tar
tar -xf 256_ObjectCategories_Featurized_FC2.tar
hadoop fs -put 256_ObjectCategories_Featurized_FC2/
````

### Train a model

#### \[Optional\] Start the web UI

Specify the port via `-p [PORT]`

````
java -jar dl4j-ui-1.0.0-jar-with-dependencies.jar -p 9000

````

Once you have the data featurized and saved in HDFS, you can train a model.

*Note*: Substitute the IP address of your machine in the command below in order to view the UI.
Optionally, don't provide a `--ui` argument to skip it entirely.

````
GC_FLAGS="-XX:+UseG1GC -XX:MaxGCPauseMillis=200 -Dorg.bytedeco.javacpp.maxretries=100 -Dorg.bytedeco.javacpp.maxbytes=13000000000"
BIGTOP_JAVA_MAJOR=8 # ensures Java 8 on distros like CDH
spark2-submit \
--master yarn \
--deploy-mode client \
--conf spark.driver.extraJavaOptions="$GC_FLAGS" \
--conf spark.executor.extraJavaOptions="$GC_FLAGS" \
--conf spark.locality.wait=0 \
--conf spark.driver.maxResultSize=10g \
--conf spark.yarn.executor.memoryOverhead=15g \
--conf spark.yarn.driver.memoryOverhead=15g \
--conf spark.executor.cores=8 \
--conf spark.kryo.registrator=org.nd4j.Nd4jRegistrator \
--conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
--conf spark.kryoserializer.buffer.max=2047m \
--executor-memory 10g \
--driver-memory 10g \
--num-executors=5 \
--class "com.cloudera.datascience.dl4j.cnn.examples.caltech256.TrainFeaturized" \
dl4j-cnn-1.0.0-jar-with-dependencies.jar \
--inputLayer fc2 \
--train hdfs:///path/to/256_ObjectCategories_Featurized_FC2/train/ \
--valid hdfs:///path/to/256_ObjectCategories_Featurized_FC2/valid/ \
--epochs 20 \
--rate 0.01 \
--batchSize 32 \
--averagingFrequency 5 \
--ui "http://0.0.0.0:9000/" \
--momentum 0.9 \
--reg 0.05 \
--validationInterval 5 \
--updater NESTEROVS
````

## Reference

The following examples from [dl4j-examples](https://github.com/deeplearning4j/dl4j-examples)
were helpful references for the transfer learning and MNIST exercise:

* [MNIST](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-spark-examples/dl4j-spark/src/main/java/org/deeplearning4j/mlp/MnistMLPExample.java)
* [VGG Transfer Learning](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/transferlearning/vgg16/EditAtBottleneckOthersFrozen.java)
* [Spark VGG Transfer](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-spark-examples/dl4j-spark/src/main/java/org/deeplearning4j/transferlearning/vgg16/FitFromFeaturized.java)
