#! /usr/bin/env bash

dir=`dirname $0`

cp /dli/assessment_results/train.out $dir/train.out

# Get the training and validation accuracy of the
# last step of the training.
val_acc=$(cat train.out | grep "Validation Accuracy" | tail -n 1 | awk -F"Validation Accuracy =" '{print $2}' | awk '{print $1}')
train_acc=$(cat train.out | grep "Training Accuracy" | tail -n 1 | awk -F"Training Accuracy =" '{print $2}' | awk '{print $1}' | tr -d ',')

# If these weren't found, set a negative number
# so we know something went wrong.

if [ -z "$val_acc" ]
then
    val_acc="-1.0"
fi

if [ -z "$train_acc" ]
then
    train_acc="-1.0"
fi

# Elapsed training time through the last epoch.
script_time=$(cat train.out | grep "Cumulative Time" | tail -1 | awk '{ print $7 }' | tr -d ',')

if [ -z "$script_time" ]
then
    script_time="-1.0"
fi

# Determine whether the student attempted to cheat
# by making sure that they did not edit the callback
# to return a false amount of elapsed time. We'll
# check for a couple key phrases that we
# know should be there.
cheated="no"

grep -Fq "epoch_time = time.time() - t0" $dir/../task/assessment.py

if [ $? -ne 0 ]
then
    cheated="yes"
fi

grep -Fq "total_time += epoch_time" $dir/../task/assessment.py

if [ $? -ne 0 ]
then
    cheated="yes"
fi

# Also check to make sure they didn't just copy a training
# script from a previous run. Those previous training scripts
# used Fashion MNIST, so let's make sure CIFAR-10 is actually
# the dataset that was used here.

grep -Fq "CIFAR10" $dir/../task/assessment.py

if [ $? -ne 0 ]
then
    cheated="yes"
fi

# ...and just in case CIFAR10 was in the script, but commented out,
# make sure FashionMNIST is nowhere in the script.
grep -Fq "FashionMNIST" $dir/../task/assessment.py

if [ $? -eq 0 ]
then
    cheated="yes"
fi

# Check whether the script ran out of memory; this is commonly
# a sign that the user did not correctly assign ranks to GPUs.

oom="no"

grep -Fq "out of memory" train.out

if [ $? -eq 0 ]
then
    oom="yes"
fi

echo "cheated $cheated oom $oom script_time $script_time val_acc $val_acc train_acc $train_acc"

