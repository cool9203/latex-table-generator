SEED=99 LOG_LEVEL=INFO \
    uv run src/latex_table_generator \
    --count 100 \
    --output_path ./outputs/full-random-merge \
    --v_contents "./contents.txt" \
    --h_contents "./contents.txt" \
    --vh_contents "./contents.txt" \
    --image_path "./steels" \
    --render_headers "./render_headers" \
    --horizontal_count 1 3 \
    --vertical_count 1 3 \
    --skew_angle -3 3 \
    --new_image_size 2480 3508 \
    --min_crop_size 0.3 \
    --rows_range 1 20 \
    --add_space_row_percentage 0.3 \
    --dropout_percentage 0.3 \
    --merge_methods "horizontal" "vertical" "hybrid" \
    --html_label_cell_merge \
    --tqdm
