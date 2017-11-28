for DATASET in paris
do
python evaluation.py --dataset $DATASET --query_expansion
python evaluation.py --dataset $DATASET --query_expansion --weighting 'gaussian'
python evaluation.py --dataset $DATASET --query_expansion --weighting 'l2norm'
python evaluation.py --dataset $DATASET --query_expansion --weighting 'SALGAN'
done
