#!/usr/bin/env bash

# This script uses weight transfer as a transfer learning method to transfer
# already trained neural net model on train data to new data from another source
#
# Model preparation: The last layer (prefinal and output layer) from
# already-trained wsj model is removed and 3 randomly initialized layer
# (new tdnn layer, prefinal, and output) are added to the model.
#
# Training: The transferred layers are retrained with smaller learning-rate,
# while new added layers are trained with larger learning rate using rm data.
# The chain config is as in run_tdnn_5n.sh and the result is:
#System tdnn_5n tdnn_wsj_rm_1a
#WER      2.71     1.68
set -e

# configs for 'chain'
stage=0
train_stage=-10
get_egs_stage=-10

exp_fold=exp_hindi_model_1

dir=$exp_fold/chain/tdnn
xent_regularize=0.2
adapt="hindi_adapt_UP" #
test_sets="ASER_UP_Hindi_test"
# configs for transfer learning
src_mdl=$dir/final.mdl # Input chain model
                                                   # trained on source dataset (wsj).
                                                   # This model is transfered to the target domain.

src_mfcc_config=conf/mfcc_hires.conf # mfcc config used to extract higher dim
                                                  # mfcc features for ivector and DNN training
                                                  # in the source domain.
src_ivec_extractor_dir=$exp_fold/nnet3/extractor  # Source ivector extractor dir used to extract ivector for
                         # source data. The ivector for target data is extracted using this extractor.
                         # It should be nonempty, if ivector is used in the source model training.

common_egs_dir=
primary_lr_factor=0 # The learning-rate factor for transferred layers from source
                       # model. e.g. if 0, the paramters transferred from source model
                       # are fixed.
                       # The learning-rate factor for new added layers is 1.0.

nnet_affix=_online_iitm
# End configuration section.

echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

required_files="$src_mfcc_config $src_mdl"
use_ivector=false
ivector_dim=$(nnet3-am-info --print-args=false $src_mdl | grep "ivector-dim" | cut -d" " -f2)
if [ -z $ivector_dim ]; then ivector_dim=0 ; fi

if [ ! -z $src_ivec_extractor_dir ]; then
  if [ $ivector_dim -eq 0 ]; then
    echo "$0: Source ivector extractor dir '$src_ivec_extractor_dir' is specified "
    echo "but ivector is not used in training the source model '$src_mdl'."
  else
    required_files="$required_files $src_ivec_extractor_dir/final.dubm $src_ivec_extractor_dir/final.mat $src_ivec_extractor_dir/final.ie"
    use_ivector=true
  fi
else
  if [ $ivector_dim -gt 0 ]; then
    echo "$0: ivector is used in training the source model '$src_mdl' but no "
    echo " --src-ivec-extractor-dir option as ivector dir for source model is specified." && exit 1;
  fi
fi

for f in $required_files; do
  if [ ! -f $f ]; then
    echo "$0: no such file $f." && exit 1;
  fi
done

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 4" if you have already
# run those things.

gmm="tri3_2750_55000"
ali_dir=$exp_fold/${gmm}_ali_${adapt}_sp
treedir=$exp_fold/chain/tree_a_sp
lang=data/lang_chain_$adapt


cp -r data/lang_generic/all-stories_0 $lang
ali_rqu=4
if [ $ali_rqu -le 3 ]; then

mfccdir=dumpdir/mfcc_data

./steps/make_mfcc.sh --cmd "run.pl" --nj "5" data/$adapt $exp_fold/make_mfcc/$adapt $mfccdir/$adapt || exit 1;
./steps/compute_cmvn_stats.sh data/$adapt $exp_fold/make_mfcc/$adapt $mfccdir/$adapt || exit 1;
#./utils/fix_data_dir.sh data/$adapt || exit 1;

./steps/align_si.sh --nj 5 --cmd run.pl data/${adapt} $lang $exp_fold/$gmm $ali_dir

fi
#exit 0

local/online/run_nnet2_adapt.sh  --stage $stage \
                                 --ivector-dim $ivector_dim \
                                 --nnet-affix "$nnet_affix" \
                                 --mfcc-config $src_mfcc_config \
				 --train-sets $adapt \
				 --test-sets $test_sets \
				 --exp-fold $exp_fold \
                                 --extractor $src_ivec_extractor_dir || exit 1;


src_mdl_dir=`dirname $src_mdl`
ivec_opt=""
if $use_ivector;then ivec_opt="--online-ivector-dir ${exp_fold}/nnet2${nnet_affix}/ivectors_adapt" ; fi

if [ $stage -le 4 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  nj=1 #$(cat $ali_dir/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/${adapt} \
    $lang $exp_fold/$gmm $exp_fold/tri3b_lats || exit 1;
  rm $exp_fold/tri3b_lats/fsts.*.gz 2>/dev/null || true # save space # old scripts use GMM for alignment?!
 # steps/nnet3/align_lats.sh --nj 5 --cmd "$train_cmd" $ivec_opt \
 #   --generate-ali-from-lats true \
 #   --acoustic-scale 1.0 --extra-left-context-initial 0 --extra-right-context-final 0 \
 #   --frames-per-chunk 150 \
 #   --scale-opts "--transition-scale=1.0 --self-loop-scale=1.0" \
 #   data/${adapt}_hires $lang $src_mdl_dir $exp_fold/tri3b_lats || exit 1;
 # rm $exp_fold/tri3b_lats/fsts.*.gz # save space

fi

if [ $stage -le 5 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  #rm -r $lang 2>/dev/null || true
  #cp -r data/lang $lang
  #silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  #nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  #steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
   echo "Already done, replace with proper stage from 1c script later"
fi

if [ $stage -le 6 ]; then
  # Build a tree using our new topology.
#  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
#    --leftmost-questions-truncate -1 \
#    --cmd "$train_cmd" 1200 data/$adapt $lang $ali_dir $treedir || exit 1;
   echo "Not building new trees"
fi

if [ $stage -le 7 ]; then
  echo "$0: Create neural net configs using the xconfig parser for";
  echo " generating new layers, that are specific to dev set from iitm challenge. These layers ";
  echo " are added to the transferred part of the training data only network.";
  num_targets=$(tree-info --print-args=false $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
  mkdir -p $dir
  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  relu-batchnorm-dropout-layer name=tdnn-target input=Append(tdnnf13.noop@-3,tdnnf13.noop) dim=1024
  ## adding the layers for chain branch
  relu-batchnorm-dropout-layer name=prefinal-chain input=tdnn-target dim=1024 target-rms=0.5
  output-layer name=output include-log-softmax=false dim=$num_targets max-change=1.5
  relu-batchnorm-dropout-layer name=prefinal-xent input=tdnn-target dim=1024 target-rms=0.5
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5
EOF
  steps/nnet3/xconfig_to_configs.py --existing-model $src_mdl \
    --xconfig-file  $dir/configs/network.xconfig  \
    --config-dir $dir/configs/

  # Set the learning-rate-factor to be primary_lr_factor for transferred layers "
  # and adding new layers to them.
  $train_cmd $dir/log/generate_input_mdl.log \
    nnet3-copy --edits="set-learning-rate-factor name=* learning-rate-factor=$primary_lr_factor" $src_mdl - \| \
      nnet3-init --srand=1 - $dir/configs/final.config $dir/input.raw  || exit 1;
fi

if [ $stage -le 8 ]; then
  echo "$0: generate egs for chain to train new model on rm dataset."
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/rm-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi
  ivector_dir=
  if $use_ivector; then ivector_dir="${exp_fold}/nnet2${nnet_affix}/ivectors_adapt" ; fi

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --trainer.input-model $dir/input.raw \
    --feat.online-ivector-dir "$ivector_dir" \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.01 \
    --chain.apply-deriv-weights false \
    --egs.dir "$common_egs_dir" \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width 150 \
    --trainer.num-chunk-per-minibatch=128 \
    --trainer.frames-per-iter 1000000 \
    --trainer.num-epochs 4 \
    --trainer.optimization.num-jobs-initial=1 \
    --trainer.optimization.num-jobs-final=1 \
    --trainer.optimization.initial-effective-lrate=0.00005 \
    --trainer.optimization.final-effective-lrate=0.000005 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs true \
    --feat-dir data/${adapt}_hires \
    --tree-dir $treedir \
    --lat-dir $exp_fold/tri3b_lats \
    --dir $dir || exit 1;
fi

if [ $stage -le 9 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  ivec_opt=""
  if $use_ivector;then
    ivec_opt="--online-ivector-dir ${exp_fold}/nnet2${nnet_affix}/ivectors_${test_sets}"
  fi
  utils/mkgraph.sh --self-loop-scale 1.0 $lang $dir $dir/graph
  steps/nnet3/decode_lowmem.sh --acwt 1.0 --post-decode-acwt 10.0 \
    --scoring-opts "--min-lmwt 10 --max-lmwt 50" \
    --nj 5 --cmd "$decode_cmd" $ivec_opt \
    $dir/graph data/${test_sets}_hires $dir/decode_${test_sets}_retrain_1a || exit 1;

fi
wait;
exit 0;
