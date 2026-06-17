
echo [1/2] Running GDE - Image Modality - Closed world
python -m classification --dataset 3_attributes --experiment_name GDE --modality_IW image --test_phase test  --result_path results_GDE_image_individual.json

echo [2/2] Running LDE - Image Modality - Closed world
python -m classification --dataset 3_attributes --experiment_name LDE --modality_IW image --test_phase test  --result_path results_LDE_image_individual.json

pause