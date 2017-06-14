# Overview

Deeplearning4j is a suite of projects (as well as a project itself)
that are aimed at providing a comprehensive solution to deep learning
on the JVM. It was created in 2014 and is supported commercially by a
relatively small startup [Skymind](https://skymind.ai/). DL4J is written mostly in Java,
but uses a number of C++ libraries under the hood so that it can be
competitive in terms of numerical computing performance.

## Supported Languages

* Java (1st class citizen)
* Scala (2nd class citizen)
* Python (2nd class citizen)

## Project components

### Datavec

Formerly "Canova", Datavec is the ETL framework developed for DL4J and
allows users to easily manipulate DL4J data structures to do common ETL
tasks like loading images, data normalization, etc...

### Nd4j

Nd4j is the numerical computing library developed for DL4J and aims to
be "numpy for the JVM". They provide options for different BLAS backends,
including CPU blas libraries like OpenBLAS or GPU backends like CUDA.
Normally performing native operations from inside a JVM would still be
expensive due to the costs of copying data through the JNI. However,
ND4J manages its own off-heap memory using the JavaCPP library, and
avoids these expenive data copy operations altogether. This [provides
some headache](https://deeplearning4j.org/spark#why-memory_only_ser-or-memory_and_disk_ser-are-recommended) when integrating with Spark since Spark can't accurately
estimate the size of Java objects which contain most of its data off-heap.

### Deeplearning4j

The main project containing the deep learning functionality. Users can
build deep learning models using DL4J's builder framework, which should
feel familiar and similar to other libraries.

### Arbiter

Evaluating and tuning (e.g. hyperparameter optimization) for deep
learning models. Seems a bit more raw, less activity on this project.

Right now, HPO supports grid search and random search. [This discussion](https://github.com/deeplearning4j/Arbiter/issues/45)
on other methods seems to have died out.

## Musings

* This project aims to provide a full feature suite of not only deep
learning (CNN, RNN, RL, etc...) but also for the entire end-to-end process
required to put DL in production environments for enterprise companies.
* They move incredibly fast given the number of developers actually working
on it. This is possible due in large part to the fact that they are not
constrained by Apache governance. They welcome community members to talk directly to them, and seem
very supportive of quickly responding to feature requests or bug fixes
provided that it is "worth their time."
* Because they work so quickly, I have found their documentation to be
both incomplete and incorrect. This is somewhat expected, but as far as
adoption goes I think this is a large barrier for most users. To that end,
they do not seem to (nor do they seem to want to) cater to beginners. I
believe their target audience is enterprise companies with a base of
seasoned Java developers who can dive into their complex code base and
figure it out themselves.
* When documentation is lacking, the solution is generally to go look at
source code, though I still found their source code to be lacking sufficient
documentation. The learning process has thus far been slow.
* The development of DL4J is incredibly active. Though there are many,
many subprojects within DL4J, you can see that the most recent commit
for almost every one of these is within the last week, usually within
the last day.
* DL4J contains a very complex set of suite of projects which seems to
make managing dependencies a nightmare. To that end, in their gitter
channel they do not seem to be even willing to talk to users that don't
have a firm grasp on maven.
* While DL4J delivers on pretty much all of the necessary components in
the DL world, I think it will be quite difficult for a typical R/Python
data scientist to adopt these tools. Just to set up the MNIST example using Spark I
found myself pouring through Java source code that integrates with Spark.
I have an above average understanding of Spark internals and still it was
quite time-consuming for me to diagnose the exact effects of simple
implementation details.
* For widely used projects, e.g. Tensorflow, there is an abundance of
tutorials, blog posts, StackOverflow support, etc... For DL4J these things
are lacking. The alternatives seem to be looking directly at source code
which is indeed helpful but challenging for a typical data scientist, or
to talk directly to developers through their gitter channel. This may be
ok for users who have a strong incentive to learn this framework but may
lead others to settle on a framework with less overhead.
* Some of these issues may be relieved by the [planned integration with
Keras](https://github.com/fchollet/keras/issues/6042).
* DL4J is going to be a good solution for users that want to do more
involved, large-scale deep learning on the JVM, and don't mind getting
their hands dirty with the sort of chaos that comes with a fast-changing
project.


## Internals

### Training on Spark

dl4j supports training on Spark using [synchronous parameter averaging](https://deeplearning4j.org/spark#how).

Consider the following configuration:

````scala
  val tm = new ParameterAveragingTrainingMaster.Builder(1)
    .averagingFrequency(10)
    .workerPrefetchNumBatches(2)
    .batchSizePerWorker(16)
    .build()
````

In this configuration we will train on an `RDD[org.nd4j.linalg.dataset.DataSet]`,
where each of the RDD elements contain a single example (`Builder(1)`).
We will divide the entire RDD up into mini-batches consisting of 16 images
and partition those minibatches into the number of partitions in our RDD.
One epoch of training will process all of the mini-batches. Further, we
will only perform parameter averaging after every 10 minibatch updates
on our workers. We have 60000 examples split into 3750 minibatches of
size 16 images. Those minibatches are divided across 8 partitions and we
want to average parameters every 10 minibatches. So in each epoch we
perform `3750 / 8 / 10 = 46` reduce operations. So each epoch requires a
map-reduce of 46 RDDs in serial.

The distributed training is done using Spark's driver to synchronize
updates. Before an epoch begins, the training data is batched up into
minibatches and written to some storage system like HDFS. Then, they
repartition the batches so that all the allocated cores are doing even
work. The begin training an epoch and in each core a new model is trained
on one or more minibatches. After some number of mini batches, the models
from each core are synced together via a call to `aggregate`. This happens
until all the minibatches have been consumed.

This happens to be pretty slow - other DL frameworks have found ways to
remove the Spark driver from the parameter synchronization steps. For
this reason, the DL4J devs have recently implemented parameter server
architecture using Aeron so that parameter synchronization can happen
outside of Spark. This is new/raw and not fully supported yet.
