IMG_LIST=$1
OUT_DIR=$2
echo $1
echo $2


SELF=$(dirname "$(realpath $0)")
TARGET=$(eval "cat $1 | wc -l")
RESULT_COUNT=$(eval "find $2 -name *.json | wc -l")
sleep 3s

while [ $RESULT_COUNT -lt $TARGET ]; do
    echo $RESULT_COUNT $TARGET
    python3 $SELF/web_enetity.py detect_dataset $1 $2
    RESULT_COUNT=$(eval "find $2 -name *.json | wc -l")
done
