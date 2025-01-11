python train_multigpu.py --data_path ../../data --neg_num 15 --data_ratio 100 --model_save_path model_save/laic-crime-100-15 --epochs 8 --batch_size 16 --test_batch_size 64 --wd 1e-3 --log True --model_save True
python predict.py --data_path ../../data --test_batch_size 64 --model_file model_save/laic-crime-100-10/Epoch-6.pt --result_path result/laic-crime-100-10-Epoch-6
