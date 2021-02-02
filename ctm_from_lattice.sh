#!/bin/bash
. ./cmd.sh
[ -f path.sh ] && . ./path.sh


lang_dir='data/lang_generic/HI-stories_0/'
stage=1

lmwt=31
wip=-0.5

decode_dir="exp_hindi_model_UP_best/chain/tdnn/decode_t2/" 
audio_name=$1

if [ $stage -le 1 ]; then
lattice-push "ark:gunzip -c $decode_dir/lat.*.gz|" ark:-| lattice-add-penalty --word-ins-penalty=$wip ark:- ark:$decode_dir/lat_push_align_init.lats

lattice-align-words-lexicon $lang_dir/phones/align_lexicon.int $decode_dir/../final.mdl ark:$decode_dir/lat_push_align_init.lats ark:$decode_dir/lat_push_align.lats # || { echo "$0: Lattice push/align failed"; exit 1; }

lattice-to-phone-lattice $decode_dir/../final.mdl ark:$decode_dir/lat_push_align.lats ark:$decode_dir/lat_push_align_phones.lats

#rm $decode_dir/lat_push_align_init.lats # To save space on ssd # not for ram usage
lattice-to-ctm-conf --decode-mbr=false --frame-shift=0.03 --inv-acoustic-scale=$lmwt ark:$decode_dir/lat_push_align.lats - | utils/int2sym.pl -f 5 $lang_dir/words.txt - > $decode_dir/words.ctm 2> /dev/null  # || { echo "$0: Obtaining words.ctm failed"; exit 1; }
lattice-to-ctm-conf --decode-mbr=false --frame-shift=0.03 --inv-acoustic-scale=$lmwt ark:$decode_dir/lat_push_align_phones.lats - | utils/int2sym.pl -f 5 $lang_dir/phones.txt - > $decode_dir/phones.ctm #|| { echo "$0: lattice-to-phone-lattice failed"; exit 1; }

fi

#rm $decode_dir/lat_push_align.lats # To save space on ssd # not for ram usage

if [ $stage -le 2 ]; then
echo "Recording wise CTM for $audio_name"
mkdir -p results
grep $audio_name $decode_dir/words.ctm > results/${audio_name}_words.ctm
grep $audio_name $decode_dir/phones.ctm > results/${audio_name}_phones.ctm

fi
