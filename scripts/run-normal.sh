LOG_LEVEL=INFO \
    uv run src/latex_table_generator \
    --input_path \
        /mnt/c/Users/ychsu/Downloads/沛波標記data/要標記資料/20241213_需Label鋼材Data/單一正規 \
        /mnt/c/Users/ychsu/Downloads/沛波標記data/要標記資料/20241216_需Label鋼材Data/1_單一正規 \
        /mnt/c/Users/ychsu/Downloads/沛波標記data/要標記資料/20241218_需Label鋼材Data/單一正規 \
        /mnt/c/Users/ychsu/Downloads/沛波標記data/要標記資料/20241223_需Label鋼材Data/1_單正規表格 \
        /mnt/c/Users/ychsu/Downloads/沛波標記data/要標記資料/20241225_需Label鋼材Data/1_單正規表格 \
    --output_path ./outputs \
    --v_contents "./contents.txt" \
    --h_contents "./contents.txt" \
    --vh_contents "./contents.txt" \
    --image_path "./steels" \
    --css "./css.txt" \
    --render_headers "./render_headers" \
    --horizontal_count 1 3 \
    --vertical_count 1 3 \
    --skew_angle -3 3 \
    --add_space_row_percentage 0.3 \
    --dropout_percentage 0.3 \
    --merge_methods "none" \
    --tqdm
