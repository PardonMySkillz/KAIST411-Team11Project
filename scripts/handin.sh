#!/bin/bash
cd $(dirname ${BASH_SOURCE[0]})/..

./scripts/log.sh

tar -zcvf $TEAM.tar.gz ./lib/model.py ./lib/mytorch/{activation,batchnorm,conv,pad,pooling}.py src Makefile log
