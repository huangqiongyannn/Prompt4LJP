python train_multigpu.py --data_path data --neg_num 8 --data_ratio 100 --num_conti1 6 --num_conti2 6 --num_conti3 6 --epochs 8 --batch_size 16 --test_batch_size 64 --wd 1e-3 --log True --model_save True
python predict.py --data_path data/test --test_batch_size 64 --model_file model_save/crime-100-8-6/Epoch-7.pt --result_path result/crime-100-8-6-Epoch-7
python train_multigpu.py --data_path data --neg_num 5 --data_ratio 100 --num_conti1 6 --num_conti2 6 --num_conti3 6 --epochs 8 --batch_size 16 --test_batch_size 64 --wd 1e-3 --log True --model_save True
python predict.py --data_path data/test --test_batch_size 64 --model_file model_save/crime-100-5-6/Epoch-7.pt --result_path result/crime-100-5-6-Epoch-7
python train_multigpu.py --data_path data --neg_num 3 --data_ratio 100 --num_conti1 6 --num_conti2 6 --num_conti3 6 --epochs 8 --batch_size 16 --test_batch_size 64 --wd 1e-3 --log True --model_save True
python predict.py --data_path data/test --test_batch_size 64 --model_file model_save/crime-100-3-6/Epoch-7.pt --result_path result/crime-100-3-6-Epoch-7



python train_multigpu.py --data_path data --neg_num 10 --data_ratio 100 --num_conti1 6 --num_conti2 6 --num_conti3 6 --epochs 8 --batch_size 16 --test_batch_size 64 --wd 1e-3 --log True --model_save True
python predict.py --data_path data/test --test_batch_size 64 --model_file model_save/crime-100-1-6/Epoch-7.pt --result_path result/crime-100-1-6-Epoch-7