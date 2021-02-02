#!/usr/bin/env bash
# 1) _1c uses source chain-trained TDNN model instead of GMM model
# to generate alignments for adapt ASER data using source model.
# 2) It uses source model tree-dir and generates new alignments and lattices for adapt
# 3) It also trains phone LM using weighted combination of alignemts from source
#    and adapt, which is used in chain denominator graph.
#    Since we use phone.txt from source dataset, this can be helpful in cases
#    where there is a few training data in the target domain and some 4-gram phone
#    sequences have no count in the target domain. #Step 3 not done. keeping in comments
# 4) It transfers all layers in already-trained model and
#    re-train the last layer using target dataset, instead of replacing it
#    with new randomly initialized output layer.

# This script uses weight transfer as Transfer learning method

set -e

# configs for 'chain'
stage=5
train_stage=-4
get_egs_stage=-10

exp_fold=exp_hindi_model_1
adapt_no_sp="UP_RJ_new_Hindi_CS_train" #"UP_RJ_Hindi_CS_train"
adapt=${adapt_no_sp}_sp #hindi_adapt_RJ
test_sets="UP_RJ_new_valid_all"
## ASER_RJ_Hindi_test" # "DFS_2019 Nasik_2018 Ooloi_Ahmednagar_2019 DF_Ballaravada_2019"
dir=$exp_fold/chain/tdnn

# configs for transfer learning

common_egs_dir=
b1_lr_factor=0.01 # learning-rate factor for blocks in the TDNN architecture
b2_lr_factor=0.0
b3_lr_factor=0.0
b4_lr_factor=0.00125

nnet_affix=_online_iitm

#phone_lm_scales="1,10" # comma-separated list of positive integer multiplicities
                       # to apply to the different source data directories (used
                       # to give the adapt data a higher weight).

# model and dirs for source model used for transfer learning
src_mdl=$dir/final.mdl # input chain model
                                                    # trained on source dataset (wsj) and
                                                    # this model is transfered to the target domain.

src_mfcc_config=conf/mfcc_hires.conf # mfcc config used to extract higher dim
                                                  # mfcc features used for ivector training
                                                  # in source domain.
src_ivec_extractor_dir=$exp_fold/nnet3/extractor/  # source ivector extractor dir used to extract ivector for
                         # source data and the ivector for target data is extracted using this extractor.
                         # It should be nonempty, if ivector is used in source model training.

src_lang=data/lang_generic_newtopo/ #lang_generic # source lang directory used to train source model.
                                # new lang dir for transfer learning experiment is prepared
                                # using source phone set phone.txt and lexicon.txt in src lang dir and
                                # word.txt target lang dir.
src_dict=data/local/dictionary  # dictionary for source dataset containing lexicon.txt,
                                            # nonsilence_phones.txt,...
                                            # lexicon.txt used to generate lexicon.txt for
                                            # src-to-tgt transfer.

src_tree_dir=$exp_fold/chain/tree_a_sp # chain tree-dir for src data;
                                         # the alignment in target domain is
                                         # converted using src-tree

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

# dirs for src-to-tgt transfer experiment
lang_dir=data/lang_generic/all-stories_0   # lang dir for target data.# In harsha's experiment both lang_dir and src_lang_dir are same with new topo in src_lang_dir
lang_src_tgt=data/lang_iitm_hindi_adapt #data/lang_adapt_${adapt}_test_$test_sets # This dir is prepared using phones.txt and lexicon from
lat_dir=$exp_fold/chain_lats_iitm
## Replacing LM files directly, ensure lang_dir is proper ####
# %%%%%%%%%%%%%%%%%% ^^^^^^^^^^^^^^^ Check the LM used ^^^^^^^^^^^^ ############### Best procedure is to just copy lang_generic/all-stories from asr folder in ../asr/data/
if [ $stage -le 0 ]; then
   echo "$0: Augmenting retrain + valid set using speed perturbation"
   ./utils/data/perturb_data_dir_speed_3way.sh --always-include-prefix true data/$adapt_no_sp data/$adapt
   echo "Check arguments in utils/custom_harsha_valid.sh. Now finding speed perturbed train+hindiCS versions only. Loss to be computed only on valid set"
   mkdir -p data/${adapt}_valid_nsp
   for file in {utt2spk,spk2utt,text,wav.scp};do cat data/$adapt/$file data/$test_sets/$file > data/${adapt}_valid_nsp/$file;done
   ./utils/fix_data_dir.sh data/${adapt}_valid_nsp
   echo "Using lang_dir variable for LM"
   cp -r $lang_dir $lang_src_tgt
#exit 0
fi
adapt="${adapt}_valid_nsp"

#for b2_lr_factor in {0.005,0.0025};do
#for b3_lr_factor in {0.005,0.0025};do
cp -r exp_hindi_model_UP_RJ_ready exp_hindi_model_1
required_files="$src_mfcc_config $src_mdl $lang_src_tgt/phones.txt $src_dict/lexicon.txt $src_tree_dir/tree"
#echo $required_files
use_ivector=false
ivector_dim=$(nnet3-am-info --print-args=false $src_mdl | grep "ivector-dim" | cut -d" " -f2)
if [ -z $ivector_dim ]; then ivector_dim=0 ; fi

if [ ! -z $src_ivec_extractor_dir ]; then
  if [ $ivector_dim -eq 0 ]; then
    echo "$0: Source ivector extractor dir '$src_ivec_extractor_dir' is "
    echo "specified but ivector is not used in training the source model '$src_mdl'."
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
    echo "$0: no such file $f" && exit 1;
  fi
done

local/online/run_nnet2_adapt.sh  --stage $stage \
                                 --ivector-dim $ivector_dim \
                                 --nnet-affix "$nnet_affix" \
                                 --mfcc-config $src_mfcc_config \
                                 --extractor $src_ivec_extractor_dir \
                                 --exp-fold $exp_fold \
                                 --train-sets $adapt \
                                 --test-sets $test_sets    || exit 1;

src_mdl_dir=`dirname $src_mdl`
ivec_opt=""
if $use_ivector;then ivec_opt="--online-ivector-dir ${exp_fold}/nnet2${nnet_affix}/ivectors_adapt" ; fi

if [ $stage -le 4 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/nnet3/align_lats.sh --nj 5 --cmd "$train_cmd" $ivec_opt \
    --generate-ali-from-lats true \
    --acoustic-scale 1.0 --extra-left-context-initial 0 --extra-right-context-final 0 \
    --frames-per-chunk 140 \
    --scale-opts "--transition-scale=1.0 --self-loop-scale=1.0" \
    data/${adapt}_hires $lang_src_tgt $src_mdl_dir $lat_dir || exit 1;
  rm $lat_dir/fsts.*.gz # save space
fi
#exit 0
if [ $stage -le 5 ]; then
  $train_cmd $dir/log/generate_input_mdl.log \
    nnet3-am-copy --raw=true --edits="set-learning-rate-factor name=* learning-rate-factor=$b1_lr_factor; set-learning-rate-factor name=tdnn1 learning-rate-factor=$b1_lr_factor;set-learning-rate-factor name=tdnnf2 learning-rate-factor=$b1_lr_factor ;set-learning-rate-factor name=tdnnf3 learning-rate-factor=$b1_lr_factor;set-learning-rate-factor name=tdnnf4 learning-rate-factor=$b1_lr_factor; set-learning-rate-factor name=tdnnf5 learning-rate-factor=$b1_lr_factor; set-learning-rate-factor name=tdnnf6 learning-rate-factor=$b2_lr_factor;set-learning-rate-factor name=tdnnf7 learning-rate-factor=$b2_lr_factor ;set-learning-rate-factor name=tdnnf8 learning-rate-factor=$b2_lr_factor;set-learning-rate-factor name=tdnnf9 learning-rate-factor=$b2_lr_factor;set-learning-rate-factor name=tdnnf10 learning-rate-factor=$b3_lr_factor;set-learning-rate-factor name=tdnnf11 learning-rate-factor=$b3_lr_factor ;set-learning-rate-factor name=tdnnf12 learning-rate-factor=$b3_lr_factor;set-learning-rate-factor name=tdnnf13 learning-rate-factor=$b3_lr_factor;set-learning-rate-factor name=prefinal-l learning-rate-factor=$b4_lr_factor;set-learning-rate-factor name=prefinal-chain learning-rate-factor=$b4_lr_factor ;set-learning-rate-factor name=prefinal-xent learning-rate-factor=$b4_lr_factor;set-learning-rate-factor name=output learning-rate-factor=$b4_lr_factor; set-learning-rate-factor name=output-xent learning-rate-factor=$b4_lr_factor" \
      $src_mdl $dir/input.raw || exit 1;
fi

if [ $stage -le 6 ]; then
  echo "Not doing this, seems to weigh the adapt and train data phone LMs if trees of train and adapt model are different and creates a new one; no need for my case; $0: compute {den,normalization}.fst using weighted phone LM."
 # steps/nnet3/chain/make_weighted_den_fst.sh --cmd "$train_cmd" \
 #   --num-repeats $phone_lm_scales \
 #   --lm-opts '--num-extra-lm-states=200' \
 #   $src_tree_dir $lat_dir $dir || exit 1;
fi

if [ $stage -le 7 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/rm-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi
  # exclude phone_LM and den.fst generation training stage
  if [ $train_stage -lt -4 ]; then train_stage=-4 ; fi

  ivector_dir=
  if $use_ivector; then ivector_dir="${exp_fold}/nnet2${nnet_affix}/ivectors_adapt" ; fi

  # we use chain model from source to generate lats for target and the
  # tolerance used in chain egs generation using this lats should be 1 or 2 which is
  # (source_egs_tolerance/frame_subsampling_factor)
  # source_egs_tolerance = 5

 # %%%%%%% For tree_dir argument no denominator fst part from stage 0 in this script is done so just passing lat_dir, default: $src_tree_dir if new tree is built #### 
  chain_opts=(--chain.alignment-subsampling-factor=1 --chain.left-tolerance=1 --chain.right-tolerance=1)
  steps/nnet3/chain/train.py --stage $train_stage ${chain_opts[@]} \
    --cmd "$decode_cmd" \
    --trainer.input-model $dir/input.raw \
    --feat.online-ivector-dir "$ivector_dir" \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize 0.001 \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --egs.dir "$common_egs_dir" \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width 140 \
    --trainer.num-chunk-per-minibatch=64 \
    --trainer.frames-per-iter 1000000 \
    --trainer.num-epochs 2 \
    --trainer.optimization.num-jobs-initial=1 \
    --trainer.optimization.num-jobs-final=1 \
    --trainer.optimization.initial-effective-lrate=0.0005 \
    --trainer.optimization.final-effective-lrate=0.00005 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs false \
    --feat-dir data/${adapt}_hires \
    --tree-dir $lat_dir \
    --lat-dir $lat_dir \
    --dir $dir || exit 1;
fi
#     --chain.xent-regularize 0.1 is default # Using the get_egs in chain/ folder
#     --chain.l2-regularize 0.00005 is the default
echo "Do decoding with decode_run script, for consistency"
#./decode_run2.sh
#mv exp_hindi_model_1 exp_hindi_model_UP_RJ_b1_${b1_lr_factor}_b2_${b2_lr_factor}_b3_${b3_lr_factor}_b4_${b4_lr_factor}
#done
#done
exit 0
#if [ $stage -le 8 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.

#utils/mkgraph.sh --self-loop-scale 1.0 $lang_src_tgt $dir $dir/graph

#for test in $test_sets;do
#  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
#    --nj 5 --cmd "$decode_cmd" --online-ivector-dir ${exp_fold}/nnet2${nnet_affix}/ivectors_$test \
#    $dir/graph data/${test}_hires $dir/decode_${adapt}_${test}

#  cat $dir/decode_${adapt}_${test}/scoring_kaldi/best_wer

#done
#fi
#exit 0;

