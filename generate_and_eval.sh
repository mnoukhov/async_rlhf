if [ -z "${OUTPUT_DIR}" ]; then
    OUTPUT_ARG = "--model_name_or_path $OUTPUT_DIR"
else
    OUTPUT_ARG = ""
fi

python generate_for_eval.py --config configs/generate_tldr.yml $OUTPUT_ARG $@ && \
accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=$GPU load_and_eval.py --config configs/evaluate_tldr.yml $OUTPUT_ARG $@
