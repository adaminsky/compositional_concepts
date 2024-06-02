synthetic () {
    python3 synthetic_experiments.py --data_dir cached_data/clevr/ --dataset CLEVR1 --model CLIP --segment_method none --center --learn_method $1
}

truth () {
    python3 truth_topics.py --data_dir cached_data/truth --dataset truth_topics --subspace_dim 50 --method $1
}

cub () {
    python3 image_experiments/Image_concept_learning_new.py --data_dir cached_data/ --dataset_name CUB_subset --concept_num_per_attr 10 --num_attrs 3 --projection_size 200 --epochs 500 --cosine_sim --lr 0.02 --split_method $1
    python3 image_experiments/Image_concept_learning_new.py --data_dir cached_data/ --dataset_name CUB_subset --concept_num_per_attr 10 --num_attrs 3 --projection_size 200 --epochs 500 --cosine_sim --lr 0.02 --split_method $1 --do_classification
}

for method in gt pca ace dictlearn seminmf ct random ours
do
    echo "Running $method..."
    synthetic $method &> /dev/null &
    truth $method &> /dev/null &
    cub $method &> /dev/null &
    wait
done

python3 create_tables.py --markdown
