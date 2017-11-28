for DATASET in instre oxford paris
  do
    python evaluation.py --dataset $DATASET --query_expansion
    for MASK in gaussian l2norm bms itti salnet SALGAN SAM_VGG16 SAM_ResNet
      do
        python evaluation.py --dataset $DATASET --query_expansion --weighting $MASK
      done
  done
