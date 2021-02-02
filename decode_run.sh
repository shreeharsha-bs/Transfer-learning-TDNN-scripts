. ./cmd.sh
. ./path.sh

stage=1
#expall="exp_hindi_model_RJ_ILR_0.01_out_0.0 exp_hindi_model_RJ_ILR_0.01_out_0.005 exp_hindi_model_RJ_ILR_0.01_out_1.0 exp_hindi_model_RJ_ILR_0.02_out_0.0 exp_hindi_model_RJ_ILR_0.02_out_0.005 exp_hindi_model_RJ_ILR_0.02_out_1.0 exp_hindi_model_RJ_ILR_0.05_out_0.0 exp_hindi_model_RJ_ILR_0.05_out_0.005 exp_hindi_model_RJ_ILR_0.05_out_1.0 exp_hindi_model_RJ_ILR_0.0_out_1.0 exp_hindi_model_RJ_ILR_0.1_out_1.0"
expall="exp_hindi_model_UP_ILR_0.01_out_0.005"
#expall="exp_hindi_model_1"
#expall="exp_hindi_model_UP_b1_0.01_b2_0.0025_b3_0.0025_b4_0.00125_new_global_lr exp_hindi_model_UP_b1_0.01_b2_0.0025_b3_0.005_b4_0.00125_new_global_lr exp_hindi_model_UP_b1_0.01_b2_0.005_b3_0.0025_b4_0.00125_new_global_lr exp_hindi_model_UP_b1_0.01_b2_0.005_b3_0.005_b4_0.00125_new_global_lr exp_hindi_model_UP_b1_0.01_b2_0_b3_0_b4_0.00125_new_global_lr exp_hindi_model_UP_best exp_hindi_model_UP_exc_b1_0.01_b2_0_b3_0_b4_0.00125_shuffle_config_12_13 exp_hindi_model_vtlp_1.2sp"
for expdir in $expall;do
echo $expdir
test_sets="ASER_RJ_split_valid"
datadir="data"
tree_dir=$expdir/chain/tree_a_sp
dir=$expdir/chain/tdnn

# Prepare language model
if [ $stage -le 1 ]; then
echo "Creating generic LM as specified in prep_LM.sh"
./prep_LM.sh blank data/local/dictionary_phone/ 2 # Default for WER: ./prep_LM.sh new_UP2 data/local/dictionary/ to get 3 gram (3 is default no need to specify))
fi

if [ $stage -le 2 ]; then # Start from next stage if G.fst and L.fst are already created
lang=$datadir/lang_generic_newtopo
# mv $lang ${lang}.bak # uncomment for moving old lang, if switching to hindi
echo "$0: creating lang directory $lang with chain-type topology"
# Create a version of the lang/ directory that has one state per phone in the
# topo file. [note, it really has two states.. the first one is only repeated
# once, the second one has zero or more repeats.]
if [ -d $lang ]; then
        if [ $lang/L.fst -nt $datadir/lang_generic/L.fst ]; then
                echo "$0: $lang already exists, not overwriting it; continuing"
                else
                echo "$0: $lang already exists and seems to be older than $datadir/lang_generic..."
                echo " ... not sure what to do.  Exiting."
                exit 1;
        fi
 else
        cp -r $datadir/lang_generic $lang
        silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
        nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
        # Use our special topology... note that later on may have to tune this
        # topology.
        steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi
fi
exit 0
if [ $stage -le 3 ]; then

utils/lang/check_phones_compatible.sh \
$datadir/lang_generic/phones.txt $lang/phones.txt || exit 1

echo "Creating graph"
utils/mkgraph.sh --self-loop-scale 1.0 $datadir/lang_generic/all-stories_0 $tree_dir $tree_dir/graph || exit 1;
 # use the same lang here as the one created finally in prep_LM.sh # variable is lang_dir2 in prep_LM
fi

if [ $stage -le 4 ]; then
echo "Extracting test set ivectors for decoding"
for data in $test_sets;do
local/nnet3/run_ivector_common_test_IITM.sh \
  --stage 0 --nj 5 \
  --test-sets $data --data-folder $datadir --exp-folder $expdir
done
fi


if [ $stage -le 5 ]; then

chunk_width=140,100,160
frames_per_chunk=$(echo $chunk_width | cut -d, -f1) # 140 chunks are used at a time to decode, haven't tested for other widths, shouldn't matter much

echo "Started decoding test set"
for data in $test_sets; do
        #nspk=$(wc -l <$datadir/${data}_hires/spk2utt)
        #nspk=5
        steps/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --extra-left-context 0 --extra-right-context 0 \
          --extra-left-context-initial 0 \
          --extra-right-context-final 0 \
          --frames-per-chunk $frames_per_chunk \
          --nj 5 --cmd "run.pl"  --num-threads 1 \
          --online-ivector-dir $expdir/nnet3/ivectors_${data}_hires \
          $tree_dir/graph $datadir/${data}_hires ${dir}/decode_${data}_word || exit 1
done
fi
done
