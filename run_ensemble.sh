SELF=$(dirname "$(realpath $0)")

docker run -it \
    -v $SELF:/src \
    dsfhe49854/vl-bert \
    python3 /src/ensemble.py /src /src/test_set_enselble.csv