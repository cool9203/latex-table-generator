SEED=99 LOG_LEVEL=INFO \
    uv run src/latex_table_generator \
    --count 100 \
    --output_path ./outputs/full-random-merge \
    --v_contents "工作筋" "鋼材筋" "寬止筋" "含上層工作筋" "增築" "上增築" "穿樑補強" "角隅增強" "角隅增強*120" \
        "已扣約100支至x向" "CS花台" "柱擴柱" "門窗補強" "電梯剪力牆" "車道" "車道預留*10" "備料5支" "備料7支" "備料10支" \
        "另外綑綁" "甘蔗筋" \
    --h_contents "開口補強" "請依區別分開包裝" "請依施工內容分開包裝" "分開綑綁" "雜項鋼筋" "增築" \
    --vh_contents "工作筋" "鋼材筋" "寬止筋" "含上層工作筋" "增築" "上增築" "穿樑補強" "角隅增強" "角隅增強*120" \
        "CS花台" "柱擴柱" "門窗補強" "電梯剪力牆" "車道" "車道預留*10" "甘蔗筋" "開口補強" "增築" \
    --image_path "./steels" \
    --horizontal_count 1 3 \
    --vertical_count 1 3 \
    --skew_angle -3 3 \
    --new_image_size 2480 3508 \
    --min_crop_size 0.3 \
    --rows_range 1 20 \
    --tqdm
