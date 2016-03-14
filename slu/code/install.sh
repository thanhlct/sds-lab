#!/bin/bash

echo "Example script how to install TF"

if [ ! -d env ] ; then
    echo "Creating virtualenv for python3.5"
    virtualenv --python=python3 env
fi

. env/bin/activate

if [[ $(uname)=Darwin ]] ; then
    echo "Using python 3.5"
    pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.7.1-cp35-none-any.whl
elif [[ $(uname)=Linux ]] ; then
    echo "Using python 3.4"
    pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.7.1-cp34-none-linux_x86_64.whl
else
    echo "Install tensorflow yourself"
fi

deactivate
