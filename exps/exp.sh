# hrnet 200*50 = 10000 iter
# train 1024 crop 512
# valid 1024 whole infer
# test  1024 sliding infer

# label smooth
python train.py --dataset HZ_Merge --base-size 1024 --crop-size 512 \
--epochs 50 --batch-size 4 \
--loss-type label_smooth --merge-all-buildings \
--lr 5e-3 --lr-scheduler poly \
--seg_model hrnet --hrnet-width 48

# ce_thre
python train.py --dataset HZ_Merge --base-size 1024 --crop-size 512 \
--epochs 50 --batch-size 4 \
--loss-type ce_thre --merge-all-buildings \
--lr 5e-3 --lr-scheduler poly \
--seg_model hrnet --hrnet-width 48

# hrnet_ce 10767MiB
python train.py --dataset HZ_Merge --base-size 1024 --crop-size 512 \
--epochs 50 --batch-size 4 \
--loss-type ce --merge-all-buildings \
--lr 5e-3 --lr-scheduler poly \
--seg_model hrnet --hrnet-width 64

# hrnet_psp  12283MiB
python train.py --dataset HZ_Merge --base-size 1024 --crop-size 512 \
--epochs 20 --batch-size 4 \
--loss-type ce --merge-all-buildings \
--lr 1e-3 --lr-scheduler poly \
--seg_model hrnet_psp --hrnet-width 64

# hrnet_pam  12479MiB
python train.py --dataset HZ_Merge --base-size 1024 --crop-size 512 \
--epochs 20 --batch-size 4 \
--loss-type ce --merge-all-buildings \
--lr 1e-3 --lr-scheduler poly \
--seg_model hrnet_pam --hrnet-width 64

# hrnet_ocr 11197MiB
python train.py --dataset HZ_Merge --base-size 1024 --crop-size 512 \
--epochs 20 --batch-size 4 \
--loss-type ce --merge-all-buildings \
--lr 1e-3 --lr-scheduler poly \
--seg_model hrnet_ocr --hrnet-width 64
