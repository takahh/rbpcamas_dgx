cdg
cd shared_20
. batch_no_tape_aug.sh
cd ../shared_13
. batch_no_tape_aug.sh
cd ../shared_8
. batch_no_tape_aug.sh
cd ../shared_4
. batch_no_tape_aug.sh


cd ../unknown_broad_share
. batch_0.sh
. batch_1.sh
. batch_2.sh
. batch_3.sh
. batch_4.sh

cd ../shared_6_increased_data
. batch_no_tape_aug.sh

#cd ../../unknown
. batch_0.sh
. batch_1.sh
. batch_2.sh
. batch_3.sh
. batch_4.sh

#  run 9 tasks (no lncRNA)

. batch0.sh
. batch1.sh
. batch2.sh
. batch3.sh
. batch4.sh
. batch5.sh
. batch6.sh
. batch7.sh
. batch8.sh
. batch9.sh
. batch10.sh
. batch11.sh
. batch12.sh
. batch13.sh
. batch14.sh
. batch15.sh
. batch16.sh
. batch17.sh
. batch18.sh
. batch19.sh
. batch20.sh
. batch21.sh
. batch22.sh
. batch23.sh
. batch24.sh
. batch25.sh
. batch26.sh
. batch27.sh
. batch28.sh
. batch29.sh

#### submit job before the current jobs end

cd /gs/hs0/tga-science/kimura/transformer_tape_dnabert/python/single_node/past
python change_batchfiles.py
cdg
cd no_tape_aug
. bat*sh
cd ../only_lncRNA
. bat*sh
cd ../tape_aug_augmultiply
. bat*sh
cd ../tape_aug_clip
. bat*sh
cd ../tape_aug_no_2dsfot
. bat*sh
cd ../tape_aug_only_pro
. bat*sh
cd ../tape_aug_only_rna
. bat*sh
cd ../tape_no_aug
. bat*sh
cd ../no_tape_no_aug
. bat*sh
cd ../tape_aug
. bat*sh


cd no_tape_aug
rmlog
cd ../only_lncRNA
rmlog
cd ../tape_aug
rmlog
cd ../tape_aug_augmultiply
rmlog
cd ../tape_aug_clip
rmlog
cd ../tape_aug_no_2dsfot
rmlog
cd ../tape_aug_only_pro
rmlog
cd ../tape_aug_only_rna
rmlog
cd ../tape_no_aug
rmlog
cd ../no_tape_no_aug
rmlog



