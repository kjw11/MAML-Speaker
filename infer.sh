#!/bin/bash

# infer options
model_path=/work5/cslt/kangjiawen/031320-domain-glzt/masf_v2/log/05102020/3
vectors=/work5/cslt/kangjiawen/122019-tf-kaldi/experiments/data_cn/models/vox-rn-sp-aam
ark_file=/work5/cslt/kangjiawen/122019-tf-kaldi/experiments/data_cn/models/vox-rn-sp-aam/xvector.txt #input vecotrs
out_file=/work5/cslt/kangjiawen/masf_v2/output/05102020/3 #output file
infered_file=$out_file/output.ark #output vectors

# dev options
srcdir=/work5/cslt/kangjiawen/031320-domain-glzt/data/genre_ism_0514-3
files=$(ls $srcdir)
train=./CNdata/train #PLDA training set
dev=$out_file/dev #PLDA output set

stage=1
cmd=/work9/cslt/kangjiawen/temp/kaldi-cnceleb/egs/wsj/s5/utils/run.pl

if [ $stage -le 1 ]; then
  # clear old file
  rm -rf $out_file
  # get output vectors
  ./infer.py --model_path $model_path --ark_file $ark_file --out_file $out_file || exit 1;

  # get scp
  copy-vector ark:$infered_file ark,scp:$(realpath $out_file)/xvector.ark,$(realpath $out_file)/xvector.scp || exit 1;
  rm $infered_file
  sort $out_file/xvector.scp > $out_file/xvector_sort.scp
  mv $out_file/xvector_sort.scp $out_file/xvector.scp
  # split training set from all 1000 speakers vectors
  head -n 111257 $out_file/xvector.scp > $out_file/train_xvector.scp
  sed -i '/interview/d;/singing/d;/movie/d' $out_file/train_xvector.scp

  echo "Infered all vectors"
fi

if [ $stage -le 2 ]; then
# Compute Cosine scores

  # For each genre
  for name in $files ;do
    rm -rf $srcdir/$name/cosine_scores
    mkdir -p $srcdir/$name/cosine_scores
    rm -rf $srcdir/$name/log
    mkdir -p $srcdir/$name/log

    scores_dir=$srcdir/$name/cosine_scores
    enroll_data=$srcdir/$name/enroll
    trials=$srcdir/$name/test/trials

  $cmd $scores_dir/log/cosine_scoring.log \
   cat $trials \| awk '{print $1" "$2}' \| \
   ivector-compute-dot-products - \
    "ark:ivector-mean ark:$enroll_data/spk2utt scp:$out_file/xvector.scp ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-normalize-length scp:$out_file/xvector.scp ark:- |" \
     $scores_dir/cosine_scores || exit 1;

    eer=$(paste $trials  $scores_dir/cosine_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
    echo "$name Cosine EER: $eer%"

  done
fi

if [ $stage -le 3 ]; then
  rm -rf $dev 
  mkdir $dev
  # Compute the mean.vec used for centering.
  $cmd $dev/log/compute_mean.log \
    ivector-mean scp:$out_file/train_xvector.scp \
    $dev/mean.vec || exit 1;

  # Use LDA to decrease the dimensionality prior to PLDA.
  lda_dim=128
  $cmd $dev/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$out_file/train_xvector.scp ark:- |" \
    ark:$train/utt2spk_smv $dev/lda.mat || exit 1;

  # Train the PLDA model.
  $cmd $dev/log/plda.log \
    ivector-compute-plda ark:$train/spk2utt_smv \
    "ark:ivector-subtract-global-mean scp:$out_file/train_xvector.scp ark:- | transform-vec $dev/lda.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $dev/plda || exit 1;
fi

if [ $stage -le 4 ]; then
  vec_dir=$out_file/xvector.ark
  nj=8

  for name in $files ;do
    rm -rf $scores
    scores=$srcdir/$name/scores
    enroll_dir=$srcdir/$name/enroll
    trials=$srcdir/$name/test/trials
    trl_name=`basename $trials`
    trl_dir=`dirname $trials`
    $cmd JOB=1:$nj $scores/log/lda_plda_scoring.JOB.log \
      ivector-plda-scoring --normalize-length=true \
        --num-utts=ark:$enroll_dir/num_utts.ark \
        "ivector-copy-plda --smoothing=0.0 $dev/plda - |" \
        "ark:ivector-mean ark:$enroll_dir/spk2utt ark:$vec_dir ark:- | ivector-subtract-global-mean $dev/mean.vec ark:- ark:- | transform-vec $dev/lda.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "ark:ivector-subtract-global-mean $dev/mean.vec ark:$vec_dir ark:- | transform-vec $dev/lda.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "cat '$trl_dir/$nj/$trl_name.JOB' | cut -d\  --fields=1,2 |" $scores/lda_plda_scores.JOB || exit 1;

    for n in $(seq $nj); do
      cat $scores/lda_plda_scores.$n
    done > $scores/lda_plda_scores
    for n in $(seq $nj); do
      rm $scores/lda_plda_scores.$n
    done

    eer=$(paste $trials $scores/lda_plda_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
    echo "LDA_PLDA EER: $eer%"
  done
fi


