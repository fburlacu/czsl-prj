
echo [1/4] Running GDE - top 1 - seq 
python -m classification --dataset 3_attributes --experiment_name GDE --modality_IW image --test_phase test --seq --topk 1 --result_path results_top1_seq.json

echo [2/4] Running - top 1
python -m classification --dataset 3_attributes --experiment_name LDE --modality_IW image --test_phase test --topk 1 --result_path results_top1.json

echo [3/4] Running - top 3
python -m classification --dataset 3_attributes --experiment_name LDE --modality_IW image --test_phase test --topk 3 --result_path results_top3.json

echo [4/4] Running - top 3
python -m classification --dataset 3_attributes --experiment_name GDE --modality_IW image --test_phase test --seq --topk 3 --result_path results_top3_seq.json

echo [1/4] Running GDE - top 1 - seq 
python -m classification --dataset 3_attributes_old --experiment_name GDE --modality_IW image --test_phase test --seq --topk 1 --result_path results_top1_seq_old_dataset.json

echo [2/4] Running - top 1
python -m classification --dataset 3_attributes_old --experiment_name LDE --modality_IW image --test_phase test --topk 1 --result_path results_top1_old_dataset.json

echo [3/4] Running - top 3
python -m classification --dataset 3_attributes_old --experiment_name LDE --modality_IW image --test_phase test --topk 3 --result_path results_top3_old_dataset.json

echo [4/4] Running - top 3
python -m classification --dataset 3_attributes_old --experiment_name GDE --modality_IW image --test_phase test --seq --topk 3 --result_path results_top3_seq_old_dataset.json

pause